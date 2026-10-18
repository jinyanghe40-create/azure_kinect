#!/usr/bin/env python3
"""给几何管线逐步计时，找出真正的瓶颈"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pc_pipeline as P


def bench(pts, tag):
    print(f"\n=== {tag}：{len(pts)} 点 ===")
    args = P.parse_args.__wrapped__() if hasattr(P.parse_args, "__wrapped__") \
        else None
    # 直接构造一份默认参数
    save = sys.argv
    sys.argv = [save[0]]
    args = P.parse_args()
    sys.argv = save

    t = {}
    t0 = time.time()
    d = np.linalg.norm(pts, axis=1)
    keep = (d > args.min_dist) & (d < args.max_dist)
    p = pts[keep]
    t["1_距离滤波"] = time.time() - t0

    t0 = time.time()
    n, dd, mask = P.ransac_plane(p, thresh=args.table_thresh,
                                 min_inliers=int(0.20 * len(p)),
                                 normal_hint=np.array([0.0, 1.0, 0.0]),
                                 cos_tol=0.3)
    t["2_去桌面RANSAC"] = time.time() - t0
    if n is not None:
        p = p[~mask]

    t0 = time.time()
    min_pts = max(60, int((args.edge / args.voxel) ** 2 * 3))
    labels = P.voxel_cluster(p, voxel=args.voxel, min_points=min_pts)
    t["3_体素聚类"] = time.time() - t0

    n_cl = int(labels.max()) + 1 if len(labels) and labels.max() >= 0 else 0
    t0 = time.time()
    for cid in range(n_cl):
        cp = p[labels == cid]
        P.analyze_cluster(cp, dist_thresh=args.plane_thresh, max_faces=3)
    t["4_逐簇拟合面"] = time.time() - t0

    tot = sum(t.values())
    for k, v in t.items():
        bar = "█" * int(40 * v / max(1e-9, tot))
        print(f"  {k:<16} {v*1000:>8.1f} ms  {100*v/tot:>5.1f}%  {bar}")
    print(f"  {'合计':<16} {tot*1000:>8.1f} ms")


if __name__ == "__main__":
    # ① 合成场景
    pts, _ = P.make_synthetic_scene(n_cubes=4, edge=0.25, seed=0,
                                    noise_sigma=0.002)
    bench(pts, "合成场景 4 个 25cm 方块")

    # ② 真实规模：把点数放大到 10 万量级（模拟 640x480 对齐到 1280x720 + stride3）
    rng = np.random.RandomState(0)
    n = 100000
    x = rng.uniform(-0.8, 0.8, n)
    y = rng.uniform(-0.6, 0.6, n)
    z = 1.0 + 0.05 * np.sin(3 * x) + rng.normal(0, 0.003, n)
    bench(np.stack([x, y, z], 1), "模拟真实规模 10 万点（单平面+噪声）")
