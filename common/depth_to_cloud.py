#!/usr/bin/env python3
"""
深度图 -> 点云 转换器

这是你方案里"深度图转点云"那一步的完整实现。

用法:
    python3 depth_to_cloud.py --demo                        # 造一张假深度图，看效果
    python3 depth_to_cloud.py --depth d.npy --out scene.ply  # 转换真实深度图
    python3 depth_to_cloud.py --depth d.png --out scene.ply --unit mm

内参来源优先级:
    1) --calib camera_calib.json   (用 k4a_calibration.py 读出的工厂值，最准)
    2) --fx/--fy/--cx/--cy 手动指定
    3) 内置典型值（NFOV 640x576），等能读出真值后替换
"""
import argparse
import json
import math
import os
import sys

import numpy as np

# NFOV_UNBINNED (640x576) 的典型工厂值。
# ⚠️ 这是"典型值"，不是你这台相机的真值。
#    等供电恢复后跑 k4a_calibration.py 读出真值替换。
TYPICAL_NFOV = {
    "width": 640, "height": 576,
    "fx": 504.7, "fy": 504.1, "cx": 320.4, "cy": 288.3,
    "k1": 0.0934, "k2": -0.2751, "k3": 0.0948, "p1": -0.0002, "p2": 0.0001,
}


def parse_args():
    p = argparse.ArgumentParser(description="深度图 -> 点云")
    p.add_argument("--depth", default=None, help="深度图文件 (.npy/.png/.tiff)")
    p.add_argument("--out", default=None, help="输出 PLY")
    p.add_argument("--calib", default=None, help="k4a_calibration.py 生成的 json")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--unit", default="mm", choices=["mm", "m"],
                   help="深度值的单位")
    p.add_argument("--min", type=float, default=0.3, help="最近距离(m)")
    p.add_argument("--max", type=float, default=1.5, help="最远距离(m)")
    p.add_argument("--demo", action="store_true", help="造一张假深度图测试")
    p.add_argument("--no-color", action="store_true", help="不按深度上色")
    return p.parse_args()


def resolve_intrinsics(args):
    """按优先级拿到内参。"""
    src = "内置典型值（等供电后替换为工厂真值）"
    intr = dict(TYPICAL_NFOV)
    if args.calib and os.path.exists(args.calib):
        data = json.load(open(args.calib))
        d = data.get("depth", {})
        for k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"):
            if k in d:
                intr[k] = d[k]
        src = f"工厂标定 {args.calib}"
    for k in ("fx", "fy", "cx", "cy"):
        v = getattr(args, k)
        if v is not None:
            intr[k] = v
            src = "命令行指定"
    return intr, src


def make_demo_depth(intr, seed=0):
    """造一张假的深度图：一个斜放的平板 + 两个方块 + 噪声。"""
    rng = np.random.default_rng(seed)
    W, H = intr["width"], intr["height"]
    depth = np.zeros((H, W), dtype=np.float32)

    # 桌面：真实平面前向投影后，Z 随 v 线性变化（远处更远）
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float32)
    # 桌面真实深度 1.0m，相机倾斜很小 -> 用线性近似
    plane = 0.95 + 0.00035 * vv
    depth = plane.copy()

    # 两个立方体：顶面比桌面近 25cm（立方体高 25cm）
    for (x0, x1, y0, y1) in ((200, 330, 250, 380), (400, 530, 280, 410)):
        # 立方体顶面的深度 = 它所在位置桌面深度 - 0.25（再扣除近视差）
        top = (0.95 + 0.00035 * (y0 - 70)) - 0.25 + 0.00035 * (vv[y0:y1, x0:x1] - y0)
        depth[y0:y1, x0:x1] = top
    # 深度噪声（Kinect 在 1m 处约 σ=2mm）
    depth += rng.normal(0, 0.002, depth.shape)
    # 挖几个洞（模拟反光/遮挡导致的无效像素）
    for _ in range(6):
        cx0, cy0 = rng.integers(50, W - 60), rng.integers(50, H - 60)
        depth[cy0:cy0 + rng.integers(8, 25), cx0:cx0 + rng.integers(8, 25)] = 0
    return depth


def load_depth(path, unit):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        d = np.load(path).astype(np.float32)
    else:
        import cv2
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise ValueError(f"读不了 {path}")
        d = d.astype(np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    if unit == "mm":
        d = d / 1000.0
    return d


def depth_to_points(depth, intr):
    """
    核心公式。注意 u,v 是整数像素网格，不是索引顺序，
    所以用 np.meshgrid 保证 X 对应列方向。
    """
    H, W = depth.shape
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]

    # 像素网格: u 沿宽度, v 沿高度
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # 无效点: 深度 0 / NaN / 超范围
    valid = (Z.reshape(-1) > 0) & np.isfinite(pts).all(axis=1)
    return pts, valid, (X, Y, Z)


def write_ply(path, pts, colors=None):
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    with open(path, "wb") as f:
        h = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
             "property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            h += ("property uchar red\nproperty uchar green\n"
                  "property uchar blue\n")
        h += "end_header\n"
        f.write(h.encode("ascii"))
        if colors is not None:
            rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            c = np.asarray(colors, dtype=np.uint8)
            rec["r"], rec["g"], rec["b"] = c[:, 0], c[:, 1], c[:, 2]
            f.write(rec.tobytes())
        else:
            rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            f.write(rec.tobytes())


def depth_colormap(depth, dmin, dmax):
    """按深度上色：近=红，远=蓝。方便一眼看出结构。"""
    d = depth.copy()
    m = d > 0
    t = np.zeros_like(d)
    t[m] = np.clip((d[m] - dmin) / max(1e-6, dmax - dmin), 0, 1)
    r = (255 * (1 - t)).astype(np.uint8)
    g = (255 * (1 - np.abs(t - 0.5) * 2)).astype(np.uint8)
    b = (255 * t).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~m] = 0
    return rgb.reshape(-1, 3)


def main():
    args = parse_args()
    intr, src = resolve_intrinsics(args)
    print(f"[内参] {src}")
    print(f"       fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
          f"cx={intr['cx']:.1f} cy={intr['cy']:.1f}")

    if args.demo:
        depth = make_demo_depth(intr)
        out = args.out or "/tmp/demo_cloud.ply"
        print(f"[演示] 造了一张 {depth.shape[1]}x{depth.shape[0]} 假深度图")
    elif args.depth:
        depth = load_depth(args.depth, args.unit)
        out = args.out or (os.path.splitext(args.depth)[0] + ".ply")
        print(f"[读取] {args.depth}  {depth.shape[1]}x{depth.shape[0]}")
    else:
        print("[ERROR] 需要 --demo 或 --depth FILE")
        return 2

    pts, valid, (X, Y, Z) = depth_to_points(depth, intr)
    d = np.linalg.norm(pts, axis=1)
    rng_ok = valid & (d > args.min) & (d < args.max)

    print(f"\n[统计] 总像素 {len(pts)}")
    print(f"       有效深度 {int(valid.sum())} "
          f"({valid.sum()/len(pts)*100:.1f}%)")
    print(f"       在 {args.min}-{args.max}m 内 {int(rng_ok.sum())} "
          f"({rng_ok.sum()/len(pts)*100:.1f}%)")

    keep = pts[rng_ok]
    if len(keep):
        print(f"\n[范围] X ∈ [{keep[:,0].min():+.3f}, {keep[:,0].max():+.3f}] m")
        print(f"       Y ∈ [{keep[:,1].min():+.3f}, {keep[:,1].max():+.3f}] m")
        print(f"       Z ∈ [{keep[:,2].min():.3f}, {keep[:,2].max():.3f}] m")

    colors = None if args.no_color else depth_colormap(
        depth.reshape(-1), args.min, args.max)[rng_ok]
    write_ply(out, keep, colors)
    print(f"\n[DONE] 写出 {len(keep)} 点 -> {out}")
    print("       看: CloudCompare " + out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
