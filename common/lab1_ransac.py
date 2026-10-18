#!/usr/bin/env python3
"""
Lab 1: RANSAC 平面拟合 —— 亲手看它怎么工作、怎么失败

运行:
    python3 lab1_ransac.py                 # 用默认参数
    python3 lab1_ransac.py --noise 8       # 加大噪声
    python3 lab1_ransac.py --outliers 0.4  # 40% 外点
    python3 lab1_ransac.py --iters 5       # RANSAC 迭代次数太少会怎样

产出:
    终端打印拟合出的法向量 vs 真值（误差多少度）
    /tmp/lab1/*.png  三张图：点云+拟合平面、残差分布、迭代次数的影响
"""
import argparse
import math
import os

import numpy as np

OUT = "/tmp/lab1"


def ransac_plane(pts, thresh, iters, rng):
    """最小实现的 RANSAC 平面拟合，带迭代记录（教学用）。"""
    n = len(pts)
    best_cnt, best_model = 0, None
    history = []
    for i in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        nv = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(nv)
        if nn < 1e-12:
            continue
        nv = nv / nn
        d = -float(np.dot(nv, p0))
        cnt = int((np.abs(pts @ nv + d) < thresh).sum())
        history.append(cnt)
        if cnt > best_cnt:
            best_cnt, best_model = cnt, (nv, d)
    return best_model, best_cnt, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=3.0, help="平面噪声 σ (mm)")
    ap.add_argument("--outliers", type=float, default=0.15, help="外点比例 0-1")
    ap.add_argument("--iters", type=int, default=200, help="RANSAC 迭代次数")
    ap.add_argument("--thresh", type=float, default=5.0, help="内点阈值 (mm)")
    ap.add_argument("--n", type=int, default=3000, help="平面点数")
    ap.add_argument("--size", type=float, default=30.0, help="平面边长 (mm)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(OUT, exist_ok=True)

    # 真值平面: z = 0，法向 (0,0,1)
    truth_n = np.array([0.0, 0.0, 1.0])

    # 平面内点：30mm x 30mm 的方形
    half = args.size / 2000.0
    xi = rng.uniform(-half, half, args.n)
    yi = rng.uniform(-half, half, args.n)
    zi = rng.normal(0, args.noise / 1000.0, args.n)
    inliers = np.stack([xi, yi, zi], axis=1)

    # 外点：空间里随机撒
    n_out = int(args.n * args.outliers / max(1e-6, 1 - args.outliers))
    span = max(0.02, half * 1.4)
    outliers = np.stack([
        rng.uniform(-span, span, n_out),
        rng.uniform(-span, span, n_out),
        rng.uniform(-span * 1.5, span * 1.5, n_out),
    ], axis=1)

    pts = np.vstack([inliers, outliers]) if n_out else inliers
    rng.shuffle(pts)

    print(f"点云: {len(pts)} 点  (内点 {len(inliers)}, 外点 {n_out})")
    print(f"噪声 σ={args.noise}mm  外点 {args.outliers:.0%}  "
          f"阈值 {args.thresh}mm  迭代 {args.iters}\n")

    model, cnt, hist = ransac_plane(pts, args.thresh / 1000.0, args.iters, rng)
    if model is None:
        print("[FAIL] 没找到平面")
        return 1

    nv, d = model
    if np.dot(nv, truth_n) < 0:
        nv, d = -nv, -d
    err = math.degrees(math.acos(min(1.0, abs(float(np.dot(nv, truth_n))))))

    print(f"真值法向      {truth_n}")
    print(f"拟合法向      {np.round(nv, 5)}")
    print(f"法向量误差    {err:.3f}°")
    print(f"内点数        {cnt} / {len(pts)}  ({cnt/len(pts):.0%})")

    # 用内点重新拟合（PCA）—— 这一步能显著提升精度
    mask = np.abs(pts @ nv + d) < args.thresh / 1000.0
    sub = pts[mask]
    c = sub.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((sub - c).T))
    nv2 = v[:, 0]
    nv2 = nv2 / np.linalg.norm(nv2)
    if np.dot(nv2, truth_n) < 0:
        nv2 = -nv2
    err2 = math.degrees(math.acos(min(1.0, abs(float(np.dot(nv2, truth_n))))))
    print(f"\n用内点做 PCA 精修后:")
    print(f"拟合法向      {np.round(nv2, 5)}")
    print(f"法向量误差    {err2:.3f}°   (提升了 {err - err2:+.3f}°)")

    # 理论下限
    D = args.size / 1000.0
    theory = math.degrees((args.noise / 1000.0 / (D / 2)) *
                          math.sqrt(12 / max(1, cnt)))
    print(f"\n理论下限 ≈ {theory:.3f}°   (公式: σ/(D/2)·√(12/N))")

    _make_plots(pts, nv, d, truth_n, hist, args)
    print(f"\n图已保存到 {OUT}/")
    return 0


def _make_plots(pts, nv, d, truth_n, hist, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] 没装 matplotlib，跳过画图")
        return

    thresh = args.thresh / 1000.0
    dist = pts @ nv + d
    mask = np.abs(dist) < thresh

    # 图1: 3D 点云 + 拟合平面
    fig = plt.figure(figsize=(14, 4.5))

    ax = fig.add_subplot(131, projection="3d")
    ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2], s=1,
               c="tab:green", label=f"内点 {mask.sum()}")
    ax.scatter(pts[~mask, 0], pts[~mask, 1], pts[~mask, 2], s=1,
               c="tab:red", alpha=0.4, label=f"外点 {(~mask).sum()}")
    ax.set_title("RANSAC 拟合结果\n绿=内点 红=外点")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(fontsize=7, loc="upper left")

    # 图2: 残差分布
    ax2 = fig.add_subplot(132)
    ax2.hist(dist * 1000, bins=80, color="tab:blue", alpha=0.75)
    ax2.axvline(thresh * 1000, color="r", ls="--", label=f"+阈值 {args.thresh}mm")
    ax2.axvline(-thresh * 1000, color="r", ls="--")
    ax2.set_xlabel("点到拟合平面的距离 (mm)")
    ax2.set_ylabel("点数")
    ax2.set_title(f"残差分布\n噪声 σ={args.noise}mm")
    ax2.legend(fontsize=8)
    ax2.set_xlim(-max(20, args.noise * 4), max(20, args.noise * 4))

    # 图3: 迭代收敛
    ax3 = fig.add_subplot(133)
    ax3.plot(hist, lw=1, color="tab:purple")
    ax3.axhline(mask.sum(), color="g", ls="--",
                label=f"最终内点 {mask.sum()}")
    ax3.set_xlabel("RANSAC 迭代次数")
    ax3.set_ylabel("该次采样的内点数")
    ax3.set_title(f"迭代收敛过程（共 {args.iters} 次）")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "ransac.png"), dpi=110)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
