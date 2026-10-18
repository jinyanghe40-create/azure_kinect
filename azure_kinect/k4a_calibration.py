#!/usr/bin/env python3
"""
读取 Azure Kinect 的工厂标定参数（内参 + 畸变 + 深度↔彩色外参）

为什么不需要自己拿棋盘格标定:
    Azure Kinect 出厂时每个镜头都单独标定过，参数存在相机固件里。
    工厂标定精度通常优于自己用棋盘格标的结果（尤其是深度相机，
    你根本没法用棋盘格标它——它测不了棋盘格的角点）。
    所以正确做法是: 读出来 -> 验证 -> 直接用。

用法:
    python3 k4a_calibration.py                 # 打印全部参数
    python3 k4a_calibration.py --json out.json # 存成 json 供点云生成使用
    python3 k4a_calibration.py --verify        # 只打印验证要点
"""
import argparse
import ctypes
import json
import sys
from ctypes import (c_char_p, c_float, c_int, c_size_t, c_uint32, c_void_p,
                    POINTER, byref)

# ---------------- k4a 常量 ----------------
K4A_DEPTH_MODE_NFOV_UNBINNED = 2
K4A_CALIBRATION_TYPE_DEPTH = 0
K4A_CALIBRATION_TYPE_COLOR = 1
K4A_CALIBRATION_TYPE_GYRO = 2
K4A_CALIBRATION_TYPE_ACCEL = 3

TYPE_NAMES = {0: "DEPTH", 1: "COLOR", 2: "GYRO", 3: "ACCEL"}

# ---------------- 结构体定义（与 k4a 头文件严格对应）----------------


class K4AExtrinsics(ctypes.Structure):
    _fields_ = [
        ("rotation", c_float * 9),
        ("translation", c_float * 3),
    ]


class K4AIntrinsics(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("parameter_count", c_uint32),
        ("parameters", c_float * 15),
    ]


class K4ADeviceConfiguration(ctypes.Structure):
    _fields_ = [
        ("color_format", c_int), ("color_resolution", c_int),
        ("depth_mode", c_int), ("camera_fps", c_int),
        ("synchronized_images_only", c_int),
        ("depth_delay_off_color_usec", ctypes.c_int32),
        ("wired_sync_mode", c_int),
        ("subordinate_delay_off_master_usec", c_uint32),
        ("disable_streaming_indicator", c_int),
    ]


DEPTH_PARAM_NAMES = ["cx", "cy", "fx", "fy", "k1", "k2", "k3", "k4",
                     "k5", "k6", "codx", "cody", "p2", "p1", "metric_radius"]
COLOR_PARAM_NAMES = ["cx", "cy", "fx", "fy", "k1", "k2", "k3", "k4",
                     "k5", "k6", "codx", "cody", "p2", "p1", "metric_radius"]


def load_lib():
    for nm in ("libk4a.so.1.4", "libk4a.so.1.4.2", "libk4a.so"):
        try:
            return ctypes.CDLL(nm)
        except OSError:
            continue
    raise OSError("找不到 libk4a")


def setup_proto(k4a):
    k4a.k4a_device_get_installed_count.restype = c_uint32
    k4a.k4a_device_open.restype = c_int
    k4a.k4a_device_open.argtypes = [c_uint32, POINTER(c_void_p)]
    k4a.k4a_device_get_serialnum.restype = c_int
    k4a.k4a_device_get_serialnum.argtypes = [c_void_p, c_char_p,
                                             POINTER(c_size_t)]
    k4a.k4a_device_get_calibration.restype = c_int
    k4a.k4a_device_get_calibration.argtypes = [
        c_void_p, c_int, c_int,
        POINTER(K4AIntrinsics), POINTER(K4AIntrinsics), POINTER(K4AExtrinsics)]
    k4a.k4a_device_close.restype = None
    k4a.k4a_device_close.argtypes = [c_void_p]


def intrinsics_to_dict(intr):
    n = min(int(intr.parameter_count), 15)
    names = DEPTH_PARAM_NAMES if intr.type == K4A_CALIBRATION_TYPE_DEPTH \
        else COLOR_PARAM_NAMES
    out = {"model": "brown_conrady" if n >= 14 else "unknown",
           "parameter_count": n}
    for i in range(n):
        out[names[i]] = round(float(intr.parameters[i]), 6)
    return out


def extrinsics_to_dict(ext):
    R = [round(float(v), 8) for v in ext.rotation]
    t = [round(float(v) * 1000.0, 4) for v in ext.translation]   # m -> mm
    return {"rotation_row_major": R, "translation_mm": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="保存到 json 文件")
    ap.add_argument("--verify", action="store_true", help="只打印验证要点")
    ap.add_argument("--depth-mode", type=int,
                    default=K4A_DEPTH_MODE_NFOV_UNBINNED)
    args = ap.parse_args()

    try:
        k4a = load_lib()
    except OSError as e:
        print(f"[FAIL] {e}")
        return 4
    setup_proto(k4a)

    n = k4a.k4a_device_get_installed_count()
    if n == 0:
        print("[FAIL] 没找到设备。内参存在相机固件里，必须插上相机才能读。")
        print("       （注意：读内参不需要深度能工作，只要 USB 能枚举就行）")
        return 1

    dev = c_void_p()
    if k4a.k4a_device_open(0, byref(dev)) != 0 or not dev:
        print("[FAIL] 打不开设备（供电/权限问题，见 k4a_smoke.py）")
        return 1

    try:
        buf = ctypes.create_string_buffer(256)
        need = c_size_t(len(buf))
        sn = ""
        if k4a.k4a_device_get_serialnum(dev, buf, byref(need)) == 0:
            sn = buf.value.decode(errors="replace")

        di = K4AIntrinsics()
        ci = K4AIntrinsics()
        de = K4AExtrinsics()      # depth -> color
        rc = k4a.k4a_device_get_calibration(dev, args.depth_mode,
                                            K4A_CALIBRATION_TYPE_COLOR,
                                            byref(di), byref(ci), byref(de))
        if rc != 0:
            print(f"[FAIL] get_calibration 失败 rc={rc}")
            print("       试试 --depth-mode 1 (NFOV_2x2BINNED) 或 3 (WFOV_UNBINNED)")
            return 1

        result = {
            "serial": sn,
            "depth_mode": args.depth_mode,
            "depth": intrinsics_to_dict(di),
            "color": intrinsics_to_dict(ci),
            "depth_to_color": extrinsics_to_dict(de),
        }

        if not args.verify:
            print(f"相机序列号: {sn}")
            print(f"深度模式:   {args.depth_mode} (2=NFOV_UNBINNED)")
            for key, label in (("depth", "深度相机内参"), ("color", "彩色相机内参")):
                d = result[key]
                print(f"\n--- {label} ---")
                print(f"  分辨率由模式决定；焦距/主点单位: 像素")
                for k in ("fx", "fy", "cx", "cy"):
                    if k in d:
                        print(f"    {k:>3} = {d[k]:>10.3f}")
                if "k1" in d:
                    print(f"  畸变 k1={d['k1']:+.5f} k2={d['k2']:+.5f} "
                          f"k3={d['k3']:+.5f}")
                    print(f"       p1={d['p1']:+.5f} p2={d['p2']:+.5f} "
                          f"(切向畸变)")
            e = result["depth_to_color"]
            print(f"\n--- 深度 -> 彩色 外参 ---")
            print(f"  平移 (mm): {e['translation_mm']}")
            print(f"  （这个平移量就是两个镜头的物理间距，"
                  f"典型值 20~30mm）")

        # 验证要点
        print("\n" + "=" * 60)
        print("验证要点（照着核对，不对就说明有问题）")
        print("=" * 60)
        d, c = result["depth"], result["color"]
        checks = []
        if "fx" in d and "fy" in d:
            checks.append(("深度 fx≈fy（像素应为方形）",
                           abs(d["fx"] - d["fy"]) / d["fx"] < 0.02,
                           f"fx={d['fx']:.2f} fy={d['fy']:.2f}"))
            checks.append(("深度 fx 在 500~600 之间 (NFOV)",
                           500 < d["fx"] < 600, f"fx={d['fx']:.2f}"))
        if "fx" in c and "fy" in c:
            checks.append(("彩色 fx≈fy",
                           abs(c["fx"] - c["fy"]) / c["fx"] < 0.02,
                           f"fx={c['fx']:.2f} fy={c['fy']:.2f}"))
            checks.append(("彩色 fx 在 800~1000 之间 (720p)",
                           780 < c["fx"] < 1000, f"fx={c['fx']:.2f}"))
        t = e["translation_mm"]
        tn = (t[0] ** 2 + t[1] ** 2 + t[2] ** 2) ** 0.5
        checks.append(("两镜头间距 15~40mm",
                       15 < tn < 40, f"距离={tn:.2f}mm"))
        for name, ok, detail in checks:
            print(f"  {'✅' if ok else '❌'} {name:<28} {detail}")

        print("\n用这些参数就能把深度图转点云：")
        print("  Z = depth_mm / 1000")
        print("  X = (u - cx) * Z / fx")
        print("  Y = (v - cy) * Z / fy")

        if args.json:
            with open(args.json, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n[DONE] 已保存到 {args.json}")
        return 0
    finally:
        k4a.k4a_device_close(dev)


if __name__ == "__main__":
    raise SystemExit(main())
