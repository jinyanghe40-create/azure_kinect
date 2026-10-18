#!/usr/bin/env python3
"""把测出来的面心/体心投影回彩色图 —— 一眼看出 XYZ 对不对。

为什么需要：
    "Y 的符号是不是反了"这种问题，光看数字吵不出结果。
    把 3D 点按【外参 + 彩色内参】投回照片，画个十字：
        · 十字正好落在方块那个面上 → 整条链路（深度→点云→拟合→投影）是对的
        · 十字跑到别处 → 哪一步有符号/坐标系问题
    这是最硬的判据，不依赖任何假设。

用法（板子上，需要有 X）:
    DISPLAY=:1 python3 ~/work/azure_kinect/check_projection.py \
        --rotate 180 --calib ~/work/azure_kinect/body_calib.json
    # 输出 /tmp/proj_check.jpg，拷回电脑看

★★ 参数必须和生产【完全一致】。这个工具唯一的价值就是"用同一个配置
   把 3D 点投回照片"；配置不一致时它给出的 ✅ 什么都证明不了。
   所以 rotate 和 calib 都要显式给（或者用 K4A_BODY_CALIB 环境变量）。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "common"))
REALSENSE = os.path.abspath(os.path.join(HERE, "..", "realsense"))
for p in (COMMON, HERE, REALSENSE):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse                                           # noqa: E402
import cv2                                               # noqa: E402
import pc_pipeline as P                                  # noqa: E402
from azure_capture import AzureSource                    # noqa: E402

# 复用 RealSense 版的投影函数（相机无关）
import run_realsense as RR                               # noqa: E402


def main():
    # ★ 先用自己的 parser 把 rotate/calib 抠出来 —— 下面要 sys.argv=["x"]
    #   去骗 pc_pipeline.parse_args()，不先存下来就丢了。
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--rotate", type=int, default=180,
                    choices=[0, 90, 180, 270])
    ap.add_argument("--calib", default=os.environ.get("K4A_BODY_CALIB"),
                    metavar="PATH")
    ap.add_argument("--allow-uncalibrated", dest="allow_uncalibrated",
                    action="store_true")
    ap.add_argument("--stride", type=int, default=2)
    mine, _unknown = ap.parse_known_args()

    import body_calib as BC
    calib = BC.load_or_exit(mine.calib, rotate=mine.rotate,
                            allow_missing=mine.allow_uncalibrated)

    sys.argv = ["x"]
    a = P.parse_args()
    for k, v in dict(min_dist=0.3, max_dist=1.5, voxel=0.02, edge=0.25,
                     plane_thresh=0.003, table_thresh=0.010, max_faces=3,
                     min_cluster_points=200, face_select="most_facing",
                     dump=None, edge_auto=True, depth_noise_coef=0.008,
                     json=True, synthetic=False, replay=None).items():
        setattr(a, k, v)

    src = AzureSource(verbose=True, rotate=mine.rotate, body_calib=calib)
    # ★★ 和 run_azure 一样把三个坐标系参数喂给 pc_pipeline，否则
    #   orient_toward_camera 会以为相机在原点 → 法向翻反 → 体心退错方向
    #   （偏 ~33cm），而这个工具恰好就是用来验体心的。
    a.up_axis = ([0.0, 0.0, 1.0] if calib is not None
                 else ([0.0, 1.0, 0.0] if getattr(src, "gravity_ok", False)
                       else None))
    a.cam_axis = src.R_out[:, 2].tolist()
    a.cam_origin = src.t_out.tolist()
    a.gravity = bool(getattr(src, "gravity_ok", False))
    try:
        r = src.get(stride=mine.stride)
        if r is None:
            print("❌ 取帧失败")
            return 1
        depth, color, intr, (pts, valid, xyz) = r
        res = P.run_pipeline(np.asarray(pts)[valid], a, verbose=False)
        if not res["objects"]:
            print("❌ 没检出物体 —— 方块在画面里吗？")
            return 1
        o = max(res["objects"], key=lambda x: x["n_points"])
        fc = o["face_center_m"]
        cc = o.get("cube_center_m")
        k = getattr(src, "rotate_k", 0)
        ic = src.intr_color
        R, t = src.extr

        print()
        print("=" * 66)
        print(f" 簇 {o['n_points']} 点  {o['n_faces']} 面  夹角 {o['angle_deg']}°")
        print(f" 面心 (X,Y,Z) = ({fc[0]:+.4f}, {fc[1]:+.4f}, {fc[2]:+.4f}) m"
              f"   |P| = {np.linalg.norm(fc):.4f}")
        if cc:
            print(f" 体心 (X,Y,Z) = ({cc[0]:+.4f}, {cc[1]:+.4f}, {cc[2]:+.4f}) m")

        # 投影：3D(深度系) --R,t--> 彩色系 --内参--> 像素 --旋转--> 最终图
        vis = color.copy()
        for name, p3, col in (("face", fc, (0, 0, 255)),
                              ("cube", cc, (255, 0, 255))):
            if p3 is None:
                continue
            uv = RR.project_to_image(p3, ic, (R, t))
            if uv is None:
                print(f" {name}: 投影失败（在相机后面）")
                continue
            u, v = RR.rot_uv(uv[0], uv[1], k, ic["width"], ic["height"])
            pf = getattr(src, "px_fix", (0.0, 0.0))
            u, v = u + pf[0], v + pf[1]
            print(f" {name}: 投影像素 = ({u:.1f}, {v:.1f})   "
                  f"画面 {ic['width']}x{ic['height']}    "
                  f"{'✅ 在画面内' if 0 <= u < ic['width'] and 0 <= v < ic['height'] else '❌ 出界'}")
            if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
                cv2.drawMarker(vis, (int(u), int(v)), col,
                               cv2.MARKER_CROSS, 40, 4)
                cv2.putText(vis, name, (int(u) + 22, int(v) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)

        # 顺便标出画面中心（光轴）
        cu, cv_ = ic["width"] // 2, ic["height"] // 2
        cv2.drawMarker(vis, (cu, cv_), (0, 255, 0), cv2.MARKER_TILTED_CROSS,
                       30, 2)
        cv2.putText(vis, "optical axis", (cu + 18, cv_ + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out = "/tmp/proj_check.jpg"
        cv2.imwrite(out, vis)
        print(f"\n✅ 已存 {out}（绿十字=光轴中心，红=面心，品红=体心）")
        print("   判读：红十字应该正好落在方块朝向相机的那个面上")
        print("=" * 66)
    finally:
        src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
