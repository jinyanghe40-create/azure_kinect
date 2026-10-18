#!/usr/bin/env python3
"""抓一帧"完整证据包"，供 PC 上离线复算几何。

为什么要这个：板子上算力紧、又没有窗口，调几何问题时在板子上反复跑很慢。
把【点云 + 彩色图 + 内参 + 外参】一次性抓到 PC，就能在 PC 上慢慢画图、
把拟合出来的面和真实照片叠在一起看，一眼就能判断角度对不对。

存出来的东西：
    grab_pts.npy    Nx3 float32，点云（原始深度坐标系，单位 m）
    grab_color.jpg  彩色图（已按 --rotate/--flip 转正，和检测器看到的一致）
    grab_meta.json  深度内参 / 彩色内参 / 外参 R,t / rotate_k

用法（板子上）:
    python3 grab_bundle.py --out /tmp/bundle --flip
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "common")))
from realsense_capture import RealSenseSource      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/bundle")
    ap.add_argument("--tries", type=int, default=12,
                    help="抓几帧，挑点数最多的那帧（避免抓到空洞多的）")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--rotate", type=int, default=0)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--depth-size", default="640x480")
    ap.add_argument("--color-size", default="1280x720")
    a = ap.parse_args()

    ds = tuple(int(x) for x in a.depth_size.lower().split("x"))
    cs = tuple(int(x) for x in a.color_size.lower().split("x"))
    src = RealSenseSource(depth_size=ds, color_size=cs,
                          flip=a.flip, rotate=a.rotate)

    best = None
    for t in range(max(1, a.tries)):
        r = src.get(stride=a.stride)
        if r is None:
            print(f"  第{t}帧取帧失败，跳过")
            continue
        depth, color, intr, (pts, valid, xyz) = r
        # ★ depth_to_points 返回的是【整张网格】(H*W,3)，valid 才是"有深度的"
        #   掩码。必须 pts[valid]，否则点云里混进一大堆 (0,0,0) 原点垃圾，
        #   平面拟合会被原点那片假点带偏 —— 角度就会算错。
        p = pts[valid].astype(np.float32)
        n = int(len(p))
        print(f"  第{t}帧: {n} 有效点")
        if best is None or n > best[0]:
            best = (n, p, color.copy(), dict(intr))
    if best is None:
        print("❌ 一帧都没拿到")
        return 1

    n, pts, color, intr = best
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "grab_pts.npy"), pts)

    import cv2
    cv2.imwrite(os.path.join(a.out, "grab_color.jpg"), color)

    R, t = (None, None)
    if src.extr is not None:
        R = np.asarray(src.extr[0]).tolist()
        t = np.asarray(src.extr[1]).tolist()
    meta = {
        "n_points": n,
        "stride": a.stride,
        "rotate_k": int(getattr(src, "rotate_k", 0)),
        "intr_depth": intr,                       # 点云所在坐标系（=深度）
        "intr_color": src.intr_color,             # 彩色图内参
        "extr_R": R, "extr_t": t,                 # 深度->彩色 的刚体变换
        "color_shape": list(color.shape),
        "depth_size": list(ds), "color_size": list(cs),
    }
    with open(os.path.join(a.out, "grab_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ 存到 {a.out}  点数={n}  彩色={color.shape}")
    print(f"   深度内参 fx={intr['fx']:.1f} cx={intr['cx']:.1f}")
    src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
