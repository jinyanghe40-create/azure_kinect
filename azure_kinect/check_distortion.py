#!/usr/bin/env python3
"""验证 Azure Kinect 的畸变影响：针孔模型 vs SDK 畸变校正。

为什么要这个文件：
    Azure Kinect 的深度畸变很大（实测 k1=3.43）。
    如果直接用 fx/cx 做针孔反投影，点云会**弯**，物体尺寸会偏小
    （实测 25cm 的方块量成 22.3cm，差 10%）。
    正确做法是用 SDK 的 k4a_transformation_depth_image_to_point_cloud，
    它内部走完整的 Brown-Conrady 模型（含切向畸变）。

    这个脚本把两种点云跑同一套几何管线，并排比——一眼看出差多少。

用法（板子上）:
    DISPLAY=:1 python3 ~/work/azure_kinect/check_distortion.py --true-edge 0.25
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "common"))
sys.path.insert(0, COMMON)
sys.path.insert(0, HERE)

import pc_pipeline as P                    # noqa: E402
from azure_capture import AzureSource      # noqa: E402


def geom_args(edge):
    sys.argv = ["x"]
    a = P.parse_args()
    over = dict(min_dist=0.3, max_dist=1.5, voxel=0.02, edge=edge,
                plane_thresh=0.003, table_thresh=0.010, max_faces=3,
                min_cluster_points=200, face_select="most_facing",
                dump=None, edge_auto=True, depth_noise_coef=0.008,
                json=True, synthetic=False, replay=None)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true-edge", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--frames", type=int, default=5)
    a = ap.parse_args()
    g = geom_args(a.true_edge)

    print("=" * 72)
    print(f" 畸变影响对比   真实边长 {a.true_edge*100:.2f} cm")
    print("=" * 72)
    print(f"{'点云来源':<14}{'边长(cm)':>10}{'误差':>10}{'面心(mm)':>26}"
          f"{'RMS(mm)':>9}{'点数':>8}")
    print("-" * 72)

    # 两种模式各抓一帧同一场景，跑同一套管线
    for tag, use_sdk in (("针孔模型", False), ("SDK畸变校正", True)):
        src = AzureSource(verbose=False, use_sdk_points=use_sdk)
        best = None
        try:
            for _ in range(max(1, a.frames)):
                r = src.get(stride=a.stride)
                if r is None:
                    continue
                depth, color, intr, (pts, valid, xyz) = r
                p = np.asarray(pts)[valid]
                if best is None or len(p) > len(best):
                    best = p
        finally:
            src.close()
        if best is None:
            print(f"{tag:<14}  取帧失败")
            continue
        res = P.run_pipeline(best, g, verbose=False)
        if not res["objects"]:
            print(f"{tag:<14}  没检出物体")
            continue
        o = max(res["objects"], key=lambda x: x["n_points"])
        e = o["edge_used_m"]
        err = (e - a.true_edge) * 100
        fc = np.array(o["face_center_m"]) * 1000
        print(f"{tag:<14}{e*100:>10.2f}{err:>+9.2f}%"
              f"  ({fc[0]:+7.1f},{fc[1]:+7.1f},{fc[2]:+7.1f})"
              f"{o['face_rms_mm']:>9.2f}{o['n_points']:>8}")
    print("=" * 72)
    print(" 判读：边长那一列谁更接近真实值，谁就是对的点云。")
    print("       两者面心差多少，就是畸变对位置的直接影响。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
