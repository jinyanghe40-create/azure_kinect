#!/usr/bin/env python3
"""
RealSense（D435/D435i/D415/D455）取帧 → 点云

为什么单独一个文件：整条几何管线是【相机无关】的 ——
    depth_to_points(depth, intr) → 点云 → pc_pipeline.run_pipeline(pts, args)
这个文件就是那层"适配器"，换相机只换它，算法一行不用改。

用法:
    python3 realsense_capture.py --list              # 列出接上的相机
    python3 realsense_capture.py --info              # 打印内参
    python3 realsense_capture.py --save out.ply      # 抓一帧存点云（给 pc_pipeline 用）
    python3 realsense_capture.py --save-depth d.npy  # 只存深度图
    python3 realsense_capture.py --fake --save t.ply  # ★ 没相机也能跑通全流程（合成深度）

Jetson 上的安装（板子）:
    # Jetson 内核没有 RealSense 需要的补丁，必须用 RSUSB 后端源码编译
    git clone https://github.com/IntelRealSense/librealsense.git
    cd librealsense && mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DFORCE_RSUSB_BACKEND=true \
             -DBUILD_PYTHON_BINDINGS=true -DPYTHON_EXECUTABLE=$(which python3)
    make -j4 && sudo make install
    sudo cp ../config/99-realsense-libusb.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules && sudo udevadm trigger
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "common"))
sys.path.insert(0, COMMON)      # depth_to_cloud 在 common/ 里（相机无关）
from depth_to_cloud import depth_to_points, write_ply   # noqa: E402


# =====================================================================
class RealSenseSource:
    """封装 RealSense：拿到的永远是 (depth_mm, color_bgr, intr, xyz)"""

    def __init__(self, depth_size=(640, 480), color_size=(1280, 720),
                 fps=30, align_to_color=False, flip=False, rotate=0):
        # rotate: 相机装歪了时用。0/90/180/270 度（顺时针）。
        #         图像、深度图、点云、内参【一起转】，保证 2D 框和 3D 坐标
        #         始终在同一个坐标系里，配对不会错。
        # flip=True 等价于 rotate=180（旧的写法，保留兼容）。
        import pyrealsense2 as rs          # ★ 延迟导入：没装库也能 --fake
        self.rs = rs
        self.align_to_color = align_to_color
        self.rotate = (180 if flip and not rotate else rotate) % 360
        self.flip = self.rotate == 180      # 兼容旧的判断

        # ★ 为什么要重试 + 换分辨率兜底：
        #   librealsense 在 Jetson（RSUSB 后端）上有个已知毛病 ——
        #   `pipeline.start()` 有时会报
        #       Failed to resolve the request: Format: Z16, width: 640, height: 480
        #   但**同样的配置再试一次就成功了**（设备刚枚举完时状态没准备好）。
        #   实测：同一个进程里连试 5 次 640x480 全部成功，
        #   而一个新进程的第一次调用常常失败。
        #   所以这里做两件事：① 失败重试  ② 重试还不行就换分辨率。
        tried = [tuple(depth_size)]
        for alt in ((848, 480), (640, 360), (480, 270)):
            if tuple(alt) != tuple(depth_size):
                tried.append(alt)

        last_err = None
        for res in tried:
            for attempt in range(3):
                self.pipe = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.depth, res[0], res[1],
                                  rs.format.z16, fps)
                cfg.enable_stream(rs.stream.color, color_size[0],
                                  color_size[1], rs.format.bgr8, fps)
                try:
                    self.profile = self.pipe.start(cfg)
                    if tuple(res) != tuple(depth_size):
                        print(f"[RealSense] ⚠️ {depth_size[0]}x{depth_size[1]} "
                              f"打不开，改用 {res[0]}x{res[1]}")
                    elif attempt > 0:
                        print(f"[RealSense] ℹ️ 第 {attempt+1} 次尝试才成功"
                              f"（RSUSB 的偶发问题，正常）")
                    self.depth_size = tuple(res)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    try:
                        self.pipe.stop()
                    except Exception:
                        pass
                    time.sleep(0.4)
            if last_err is None:
                break
        if last_err is not None:
            raise RuntimeError(
                f"深度 {depth_size} 和几个备用分辨率都打不开。\n"
                f"  最后一次错误: {last_err}\n"
                f"  试: ① 相机拔插一次 ② python3 test_streams.py 看哪一路有问题")

        # ★★ 为什么默认【不对齐】：
        #   rs.align() 在 Jetson 上是 CPU 逐像素重采样，实测【固定 440ms/帧】，
        #   跟分辨率无关（1280x720 和 424x240 都是 440ms）。
        #   而它的唯一作用就是"让深度和彩色像素一一对应" ——
        #   我们其实只需要在【把 3D 点投回彩色图】时做一次深度→彩色的
        #   刚体变换（一个 3x3 矩阵乘法，微秒级）就够了。
        #   所以默认关掉对齐，省下这 440ms。
        self.aligner = rs.align(rs.stream.color) if align_to_color else None
        # 深度→彩色的外参（R,t）。对齐关掉时，投回彩色图要用它。
        self.extr = None
        self.intr_color = None
        if not align_to_color:
            try:
                dp = self.profile.get_stream(rs.stream.depth) \
                    .as_video_stream_profile()
                cp = self.profile.get_stream(rs.stream.color) \
                    .as_video_stream_profile()
                e = dp.get_extrinsics_to(cp)
                self.extr = (np.array(e.rotation, np.float64).reshape(3, 3),
                             np.array(e.translation, np.float64))
            except Exception:
                self.extr = None
        # 深度尺度（默认 0.001 m/单位）
        self.depth_scale = (self.profile.get_device()
                            .first_depth_sensor().get_depth_scale())
        time.sleep(1.0)                     # 让自动曝光稳定
        for _ in range(5):                  # 丢掉前几帧
            self.pipe.wait_for_frames()

    def get(self, stride=1):
        """返回 (depth_m, color_bgr, intr_dict, xyz_map)

        ★ stride>1 时【先抽样再反投影】。之前是先把 1280x720（92 万点）
          全算一遍、调用方再扔掉重算 —— 每帧白分配 44MB、白算 92 万点。
          注意：返回的 intr 始终是【全分辨率】的（用它把 3D 点投回彩色图），
          抽样版内参只在函数内部用于反投影点云。
        """
        fs = self.pipe.wait_for_frames()
        if self.aligner is not None:
            fs = self.aligner.process(fs)
        df = fs.get_depth_frame()
        cf = fs.get_color_frame()
        if not df or not cf:
            return None
        raw = np.asarray(df.get_data())
        # ★ 16 位深度图里 0xFFFF(65535) 是"没有有效深度"的哨兵值，
        #   直接乘尺度会变成 65.535m 这种假距离。必须显式置 0。
        depth = raw.astype(np.float32) * self.depth_scale
        depth[raw >= 65535] = 0.0
        color = np.asarray(cf.get_data())                # BGR uint8
        i = df.profile.as_video_stream_profile().get_intrinsics()
        # 对齐关掉时，上面拿的是【深度】内参；彩色内参单独取出来给出参
        j = cf.profile.as_video_stream_profile().get_intrinsics()
        intr_color = {"fx": float(j.fx), "fy": float(j.fy),
                      "cx": float(j.ppx), "cy": float(j.ppy),
                      "width": int(j.width), "height": int(j.height)}
        # ★★ 只转【彩色图】（给检测器用），点云保持【原始】坐标系不动。
        #   原因：外参 (R,t) 是"原始深度系 → 原始彩色系"的变换。如果连点云
        #   一起转，外参就对不上了（之前 --flip 时类别偶尔才对上就是这个原因）。
        #   正确做法：用原始外参投出【原始图的 (u,v)】，再把这个坐标
        #   按同样的旋转换算到【旋转后的图】上，去和检测框比。
        k = (self.rotate // 90) if self.rotate else 0
        if k:
            color = np.rot90(color, k=-k).copy()
        i2 = {"fx": float(i.fx), "fy": float(i.fy),
              "cx": float(i.ppx), "cy": float(i.ppy),
              "w": int(i.width), "h": int(i.height)}
        intr = {"fx": i2["fx"], "fy": i2["fy"],
                "cx": i2["cx"], "cy": i2["cy"],
                "width": int(i2["w"]), "height": int(i2["h"]),
                "model": str(i.model)}

        # ★ 先抽样，再反投影（省掉 92 万点的无用计算）
        if stride and stride > 1:
            depth = depth[::stride, ::stride].copy()
            intr_pts = dict(intr)
            intr_pts["fx"] /= stride
            intr_pts["fy"] /= stride
            intr_pts["cx"] /= stride
            intr_pts["cy"] /= stride
            intr_pts["width"] = depth.shape[1]
            intr_pts["height"] = depth.shape[0]
        else:
            intr_pts = intr
        pts, valid, xyz = depth_to_points(depth, intr_pts)
        # intr      = 点云所在的坐标系的内参（对齐关掉时=深度内参）
        # intr_color= 彩色图内参 + extr：把 3D 点投回彩色图用
        self.intr_color = intr_color
        self.rotate_k = k              # 彩色图转了几个 90°（点云没转）
        return depth, color, intr, (pts, valid, xyz)

    def close(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


def list_devices():
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("❌ 还没装 RealSense SDK（pyrealsense2）")
        print()
        print("  在板子上按顺序跑（① 和 ③ 要 sudo，② 不要）:")
        print("     ① sudo bash ~/work/realsense/install_deps.sh     # 装编译依赖，约 5 分钟")
        print("     ② bash ~/work/realsense/build.sh                 # 编译，40~90 分钟，别断电")
        print("     ③ cd ~/librealsense/build && sudo make install && sudo ldconfig")
        print("        sudo cp ~/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/")
        print("        sudo udevadm control --reload-rules && sudo udevadm trigger")
        print()
        print("  相机硬件本身认到了吗？可以先: python3 ../common/cam_check.py")
        return 1
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        print("❌ 没找到 RealSense。查：")
        print("   ① 插的是蓝色 USB 3.0 口吗")
        print("   ② lsusb | grep -i intel   应该看到 8086:0b07 (D435)")
        print("   ③ udev 规则装了吗: sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/")
        return 1
    for d in devs:
        n = d.get_info(rs.camera_info.name)
        s = d.get_info(rs.camera_info.serial_number)
        fw = d.get_info(rs.camera_info.firmware_version)
        usb = d.get_info(rs.camera_info.usb_type_descriptor)
        print(f"  ✅ {n}  序列号={s}  固件={fw}  USB={usb}")
        if usb.startswith("2"):
            print("     ⚠️ 当前是 USB 2.0！深度分辨率和帧率会受限，请插蓝色 USB3.0 口")
    return 0


# =====================================================================
def fake_depth(intr, seed=0):
    """合成一帧深度（桌面 + 两个方块），没相机时用来验证整条链路"""
    rng = np.random.default_rng(seed)
    H, W = intr["height"], intr["width"]
    z = np.full((H, W), 1.20, np.float32)                # 桌面在 1.2m
    z += rng.normal(0, 0.0015, z.shape).astype(np.float32)
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    for (ox, oy) in ((-0.15, 0.0), (0.15, 0.05)):
        u0 = int(cx + (ox - 0.125) * fx / 1.0)
        u1 = int(cx + (ox + 0.125) * fx / 1.0)
        v0 = int(cy + (oy - 0.125) * fy / 1.0)
        v1 = int(cy + (oy + 0.125) * fy / 1.0)
        u0, u1 = max(0, u0), min(W, u1)
        v0, v1 = max(0, v0), min(H, v1)
        if u1 > u0 and v1 > v0:
            z[v0:v1, u0:u1] = 1.0                        # 方块前面在 1.0m
    color = np.full((H, W, 3), 90, np.uint8)
    return z, color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="列出相机")
    ap.add_argument("--info", action="store_true", help="打印内参")
    ap.add_argument("--save", default=None, help="存点云 .ply")
    ap.add_argument("--save-depth", default=None, help="存深度图 .npy（米）")
    ap.add_argument("--save-color", default=None, help="存彩色图 .jpg")
    ap.add_argument("--fake", action="store_true", help="不用相机，用合成深度")
    ap.add_argument("--frames", type=int, default=1, help="抓几帧（默认 1）")
    ap.add_argument("--hz", type=float, default=0.0, help=">0 则连续打印 FPS")
    ap.add_argument("--depth-size", default="640x480")
    ap.add_argument("--color-size", default="1280x720")
    a = ap.parse_args()

    if a.list:
        return list_devices()

    dw, dh = (int(x) for x in a.depth_size.lower().split("x"))
    cw, ch = (int(x) for x in a.color_size.lower().split("x"))

    src = None
    if a.fake:
        print("[合成模式] 不用相机，生成一帧假深度来验证链路")
        intr = {"fx": 600.0, "fy": 600.0, "cx": dw / 2.0 - 0.5,
                "cy": dh / 2.0 - 0.5, "width": dw, "height": dh, "model": "fake"}
    else:
        try:
            src = RealSenseSource((dw, dh), (cw, ch))
        except ImportError:
            print("❌ 没装 pyrealsense2。")
            print("   板子上要源码编译（见本文件头部说明）")
            print("   想先验证流程可以加 --fake")
            return 1
        except Exception as e:
            print(f"❌ 打不开相机: {type(e).__name__} {e}")
            print("   试: python3 realsense_capture.py --list")
            return 1

    n = 0
    t0 = time.time()
    try:
        while a.frames == 0 or n < a.frames:
            if src is not None:
                r = src.get()
                if r is None:
                    continue
                depth, color, intr, (pts, valid, xyz) = r
            else:
                depth, color = fake_depth(intr)
                pts, valid, xyz = depth_to_points(depth, intr)

            n += 1
            if n == 1 or a.hz > 0:
                v = valid.sum()
                # ★ valid 是【展平】后的掩码（depth_to_points 里 reshape(-1) 了）
                d = depth.reshape(-1)[valid]
                print(f"[{n}] {intr['width']}x{intr['height']}  有效点 {v} "
                      f"({100.0*v/depth.size:.0f}%)  深度 {d.min():.2f}~{d.max():.2f}m  "
                      f"内参 fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
                      f"cx={intr['cx']:.1f} cy={intr['cy']:.1f}")

            if a.save and n == 1:
                write_ply(a.save, pts[valid], color.reshape(-1, 3)[valid]
                          if color.shape[:2] == depth.shape else None)
                print(f"✅ 点云 → {a.save}  ({int(valid.sum())} 点)")
                print(f"   接着跑: python3 pc_pipeline.py --replay {a.save}")
            if a.save_depth and n == 1:
                np.save(a.save_depth, depth)
                print(f"✅ 深度图 → {a.save_depth}")
            if a.save_color and n == 1:
                import cv2
                cv2.imwrite(a.save_color, color)
                print(f"✅ 彩色图 → {a.save_color}")

            if a.hz > 0:
                time.sleep(max(0.0, 1.0 / a.hz - (time.time() - t0 - (n - 1) / a.hz)))
    except KeyboardInterrupt:
        pass
    finally:
        if src is not None:
            src.close()
    dt = time.time() - t0
    if n:
        print(f"\n[DONE] {n} 帧，{n/max(1e-6, dt):.1f} FPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
