#!/usr/bin/env python3
"""
对【真实点云】逐步计时，找出真正的瓶颈。

用法:
    python3 bench_real.py 点云.ply [--stride ... 等几何参数]

比 bench_geometry.py 更贴近实际：它会读你相机实际拍出来的点云
（含塑料袋、桌面、地板的杂乱背景），而不是合成场景。
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pc_pipeline as P            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--edge", type=float, default=0.25)
    ap.add_argument("--min-dist", type=float, default=0.2)
    ap.add_argument("--max-dist", type=float, default=1.5)
    ap.add_argument("--plane-thresh", type=float, default=0.003)
    ap.add_argument("--table-thresh", type=float, default=0.010)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--min-cluster-points", type=int, default=0)
    a = ap.parse_args()

    pts = P.read_ply(a.ply)
    print(f"=== 点云: {os.path.basename(a.ply)}  {len(pts)} 点 ===")
    print()

    # 构造一份 args（复刻 run_realsense 里传的那套）
    sys.argv = [sys.argv[0]]
    args = P.parse_args()
    args.edge = a.edge
    args.min_dist = a.min_dist
    args.max_dist = a.max_dist
    args.plane_thresh = a.plane_thresh
    args.table_thresh = a.table_thresh
    args.voxel = a.voxel
    args.min_cluster_points = a.min_cluster_points

    T = {}

    # ---- 1 距离滤波 ----
    t0 = time.time()
    d = np.linalg.norm(pts, axis=1)
    keep = (d > args.min_dist) & (d < args.max_dist)
    pts_f = pts[keep]
    T["1_距离滤波"] = time.time() - t0
    print(f"  距离滤波后: {len(pts_f)} 点 "
          f"({100.0*len(pts_f)/len(pts):.1f}%)")

    # ---- 2 去桌面 RANSAC ----
    t0 = time.time()
    n, dd, mask = P.ransac_plane(pts_f, thresh=args.table_thresh,
                                 min_inliers=int(0.20 * len(pts_f)),
                                 normal_hint=np.array([0.0, 1.0, 0.0]),
                                 cos_tol=0.3)
    T["2_去桌面RANSAC"] = time.time() - t0
    if n is not None:
        pts_o = pts_f[~mask]
        print(f"  去掉桌面: -{int(mask.sum())} 点 → {len(pts_o)} 点")
    else:
        pts_o = pts_f
        print("  没找到桌面平面")

    # ---- 3 体素聚类 ----
    min_pts = args.min_cluster_points or max(
        60, int((args.edge / args.voxel) ** 2 * 3))
    t0 = time.time()
    labels = P.voxel_cluster(pts_o, voxel=args.voxel, min_points=min_pts)
    T["3_体素聚类"] = time.time() - t0
    n_cl = int(labels.max()) + 1 if len(labels) and labels.max() >= 0 else 0
    sizes = [int((labels == c).sum()) for c in range(n_cl)]
    print(f"  聚类: {n_cl} 个簇 (min_pts={min_pts})")
    print(f"    最大的 8 个: {sorted(sizes, reverse=True)[:8]}")

    # ---- 4 逐簇拟合面（逐个计时）----
    t0 = time.time()
    per = []
    skipped = 0
    for cid in range(n_cl):
        cp = pts_o[labels == cid]
        ext = cp.max(axis=0) - cp.min(axis=0)
        if float(ext.max()) > args.edge * 4.0:
            skipped += 1
            continue
        t1 = time.time()
        P.analyze_cluster(cp, dist_thresh=args.plane_thresh, max_faces=3)
        per.append((time.time() - t1, len(cp), float(ext.max())))
    T["4_逐簇拟合面"] = time.time() - t0
    per.sort(reverse=True)
    print(f"  跳过大簇: {skipped} 个")
    print(f"  实际分析了 {len(per)} 个簇，最慢的 5 个:")
    for dt, npt, ext in per[:5]:
        print(f"    {dt*1000:7.1f} ms   {npt:6d} 点   最大边长 {ext*100:.0f}cm")

    # ---- 汇总 ----
    tot = sum(T.values())
    print()
    print("=" * 56)
    for k, v in T.items():
        bar = "█" * int(36 * v / max(1e-9, tot))
        print(f"  {k:<16} {v*1000:>8.1f} ms  {100*v/tot:>5.1f}%  {bar}")
    print(f"  {'合计':<16} {tot*1000:>8.1f} ms")
    print("=" * 56)


if __name__ == "__main__":
    main()
