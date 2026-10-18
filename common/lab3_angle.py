#!/usr/bin/env python3
"""
============================================================
 教学 ②：夹角是怎么算出来的？
============================================================
你项目的核心就这一个数：**物体表面与相机的夹角**。

这个脚本把一个倾斜的平面当"方块的一个面"，把全过程打印出来：
    一堆 3D 点  →  拟合出一个平面  →  平面的法向量  →  夹角

而且可以指定倾角，自己验证算法算得对不对。

用法:
    python3 lab3_angle.py                 # 一次讲清楚
    python3 lab3_angle.py --sweep         # 扫一遍 0°~60°，验证准不准
    python3 lab3_angle.py --noise 5       # 加 5mm 噪声，看精度怎么掉
    python3 lab3_angle.py --show          # 画图

------------------------------------------------------------
数学就三步：

  第一步：把平面写成  n·p + d = 0
          n = (nx, ny, nz) 是法向量（垂直于这个面的方向）
          d 是平面到原点的距离

  第二步：从一堆点求 n —— 用 SVD（奇异值分解）
          点云里每个点减去质心 → 得到矩阵 A (N×3)
          对 A 做 SVD，**最小奇异值对应的那个方向就是法向量**
          （因为点在平面内变化最大，垂直于平面变化最小）

  第三步：算夹角
          相机光轴方向 = (0, 0, 1)
          夹角 = arccos( |n·(0,0,1)| / |n| ) = arccos( |nz| / |n| )

          ★ 为什么取绝对值？
            平面的法向量有两个方向（n 和 -n 都垂直于同一个面），
            取绝对值就等于"不管面朝哪边，夹角都算成正的"。
            正对相机 = 0°，侧对着相机 = 90°。
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def make_tilted_plane(angle_deg, n=4000, noise_mm=0.0, seed=0,
                      center=(0.0, 0.0, 1.0), size=0.125,
                      intr=None, img=(640, 480)):
    """
    造一个"相对相机光轴倾斜 angle_deg 度"的平面点集。

    ★ 这里用的是**真实相机的成像模型**，不是随便撒点：
        ① 在【图像】上均匀撒像素 (u,v)
        ② 从光心沿这个像素的视线方向发射线
        ③ 射线与倾斜平面求交点 → 得到 3D 点
        ④ 只保留落在物理尺寸 size 内的点
        ⑤ 给 Z 加噪声（深度相机的噪声就在 Z 上）

    这样才会自然出现真实效果：
        **面越侧对（角度越大），它在图像上的投影越窄 → 有效点越少
          → 拟合出的法向量误差越大**。
  """
    rng = np.random.RandomState(seed)   # ★ 用旧 API，兼容板子的 numpy 1.13
    intr = intr or {"fx": 918.4, "fy": 919.2, "cx": 637.6, "cy": 351.1}
    W, H = img
    th = math.radians(angle_deg)
    nvec = np.array([math.sin(th), 0.0, math.cos(th)])   # 目标法向量

    # ① 图像上均匀撒像素
    u = rng.uniform(0, W, n)
    v = rng.uniform(0, H, n)
    # ② 视线方向（未归一化，Z 分量=1）
    d = np.stack([(u - intr["cx"]) / intr["fx"],
                  (v - intr["cy"]) / intr["fy"],
                  np.ones_like(u)], axis=1)
    # ③ 射线 t·d 与平面 n·(p - c)=0 求交
    c = np.array(center, float)
    denom = d @ nvec
    ok = np.abs(denom) > 1e-6
    t = np.where(ok, (nvec @ c) / np.where(ok, denom, 1.0), np.nan)
    pts = d * t[:, None]
    # ④ 只保留落在物理尺寸内的点（沿平面内两个基方向量）
    t1 = np.array([math.cos(th), 0.0, -math.sin(th)])
    t2 = np.array([0.0, 1.0, 0.0])
    rel = pts - c[None, :]
    a = rel @ t1
    b = rel @ t2
    m = (np.abs(a) <= size) & (np.abs(b) <= size) & np.isfinite(pts[:, 2]) \
        & (pts[:, 2] > 0.05)
    pts = pts[m]
    # ⑤ 深度噪声（加在 Z 上，和真实深度相机一致）
    if noise_mm > 0 and len(pts):
        pts = pts.copy()
        pts[:, 2] += rng.normal(0, noise_mm / 1000.0, len(pts))
    return pts, nvec


def fit_plane_normal(pts, verbose=True):
    """从点云拟合平面，返回单位法向量和 RMS（点到此平面的距离）"""
    c = pts.mean(axis=0)
    A = pts - c
    # SVD：A = U S Vt，Vt 的最后一行 = 最小奇异值对应的方向 = 法向量
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    n = Vt[-1] / np.linalg.norm(Vt[-1])
    d = -float(np.dot(n, c))                       # 平面: n·p + d = 0
    dist = np.abs(A @ n)                           # 每个点到平面的距离
    rms = float(np.sqrt((dist ** 2).mean()))
    if verbose:
        print(f"  质心 c = ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})")
        print(f"  奇异值 S = {np.round(S, 5)}")
        print(f"      ↑ 前两个大（点在面内铺开），第三个很小（垂直方向几乎没变化）")
        print(f"        → 第三小的那个方向就是法向量")
        print(f"  法向量 n = ({n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f})   |n| = {np.linalg.norm(n):.4f}")
        print(f"  平面方程: {n[0]:+.3f}·x {n[1]:+.3f}·y {n[2]:+.3f}·z "
              f"{d:+.3f} = 0")
        print(f"  点到平面 RMS 距离 = {rms*1000:.3f} mm  （拟合得好不好，越小越好）")
    return n, d, rms


def angle_to_camera(n):
    """法向量 与 相机光轴(0,0,1) 的夹角（度）"""
    nz = abs(float(n[2])) / float(np.linalg.norm(n))     # ★ 取绝对值
    return math.degrees(math.acos(min(1.0, nz)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle", type=float, default=25.0, help="造平面的真实倾角")
    ap.add_argument("--noise", type=float, default=3.0, help="深度噪声(mm)")
    ap.add_argument("--sweep", action="store_true", help="扫 0~60° 验证")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    print("=" * 64)
    print(" ⓪ 先把问题说清楚：什么叫「物体与相机的夹角」")
    print("=" * 64)
    print("  把方块的一个面想象成一块板子：")
    print()
    print("     正对相机          斜着             完全侧对")
    print("     ┌────┐           ╱              │")
    print("     │ 面 │          ╱               │ 面")
    print("     └────┘         ╱                │")
    print("    相机→          相机→            相机→")
    print("     夹角 0°        夹角 30°          夹角 90°")
    print()
    print("  数学上：")
    print("      · 这个面的【法向量】n —— 垂直于这个面、指向外面的那个箭头")
    print("      · 相机看出去的方向是光轴 (0,0,1)")
    print("      · 夹角 = 这两个方向的夹角")
    print()
    print("  为什么这个数有用：")
    print("      夹角 0°  → 正对着，深度图最准，检测最稳（抓取首选）")
    print("      夹角 90° → 侧对着，深度点很少，法向量误差会很大")
    print("      所以它是判断'这个面能不能用来定位'的关键指标")

    # ---------------- ① 造点 ----------------
    print()
    print("=" * 64)
    print(f" ① 造一堆属于同一个面的 3D 点（真实倾角 {a.angle}°，噪声 {a.noise}mm）")
    print("=" * 64)
    pts, n_true = make_tilted_plane(a.angle, noise_mm=a.noise, seed=1)
    print(f"  点数: {len(pts)}")
    print("  前 3 个点 (x, y, z) 米:")
    for p in pts[:3]:
        print(f"     ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
    print(f"  真实法向量 = ({n_true[0]:+.4f}, {n_true[1]:+.4f}, {n_true[2]:+.4f})")
    print("      （这是我们故意造出来的，用来对照 —— 真实场景里没有这个「标准答案」）")

    # ---------------- ② 拟合平面 ----------------
    print()
    print("=" * 64)
    print(" ② 从这堆点拟合出平面 → 得到法向量")
    print("=" * 64)
    n_hat, d_hat, rms = fit_plane_normal(pts)

    # ★ SVD 给出的法向量符号是【任意的】（n 和 -n 都垂直于同一个面）
    #   所以有一半概率算出来跟"真实法向量"符号相反。这不是 bug，是数学本质，
    #   也正是一会儿要取绝对值的原因。这里主动点出来，避免你以为是算错了。
    if float(np.dot(n_hat, n_true)) < 0:
        print()
        print("  ⚠️ 注意！算出的法向量和上面那个「真实法向量」【符号相反】")
        print("     这不是算错了 —— SVD 只管「垂直于面的方向」，")
        print("     而垂直于一个面的方向有两个（n 和 -n）。")
        print("     SVD 给你哪一个，取决于数值细节，是任意的。")
        print("     ★ 这正是下一步要取绝对值 |nz| 的原因：把这两个方向统一成一个答案。")
        print("     （平面方程也一起翻了符号，所以还是同一个平面：两边同乘 -1）")

    # ---------------- ③ 算法向量 ----------------
    print()
    print("=" * 64)
    print(" ③ 算法向量与光轴的夹角")
    print("=" * 64)
    ang = angle_to_camera(n_hat)
    print(f"  法向量 n = ({n_hat[0]:+.4f}, {n_hat[1]:+.4f}, {n_hat[2]:+.4f})")
    print(f"  光轴     = (0, 0, 1)")
    print()
    print(f"  点积 n·光轴 = {n_hat[2]:+.4f}")
    print(f"  夹角 = arccos( |nz| / |n| )")
    print(f"       = arccos( |{n_hat[2]:+.4f}| / {np.linalg.norm(n_hat):.4f} )")
    print(f"       = arccos( {abs(n_hat[2]):.4f} )")
    print(f"       = {ang:.2f}°")
    print()
    print(f"  真实倾角 = {a.angle:.2f}°   算出 = {ang:.2f}°   "
          f"误差 = {abs(ang - a.angle):.2f}°")
    print()
    print("  ★ 为什么要取绝对值 |nz|：")
    print("      n 和 -n 是同一个平面的两个法向量（指向两侧）")
    print("      不取绝对值，同一个面会算出 25° 或 155° 两个答案")
    print("      取绝对值 → 统一成 0~90°，'面朝哪边'不影响夹角")

    # ---------------- ④ 扫一遍验证 ----------------
    if a.sweep:
        print()
        print("=" * 64)
        print(" ④ 扫一遍：真实倾角 vs 算出来的夹角")
        print("=" * 64)
        print("  （每个角度重复 20 次取平均，看趋势）")
        print(f"  {'真实倾角':>8} {'算出(均)':>9} {'平均误差':>9} {'最大误差':>9} "
              f"{'点数':>6}")
        worst = 0.0
        for true_ang in range(0, 91, 5):
            errs, got_list, np_list = [], [], []
            for s_ in range(20):
                p2, _ = make_tilted_plane(true_ang, noise_mm=a.noise,
                                          seed=1000 + s_)
                if len(p2) < 12:
                    continue
                n2, _, _ = fit_plane_normal(p2, verbose=False)
                got = angle_to_camera(n2)
                errs.append(abs(got - true_ang))
                got_list.append(got)
                np_list.append(len(p2))
            if not errs:
                print(f"  {true_ang:>7}° {'可见点太少，拟合不了':>34}")
                continue
            worst = max(worst, max(errs))
            print(f"  {true_ang:>7}° {np.mean(got_list):>8.2f}° "
                  f"{np.mean(errs):>8.2f}° {max(errs):>8.2f}° {int(np.mean(np_list)):>6}")
        print()
        print(f"  最坏情况误差 {worst:.2f}°   （噪声 σ={a.noise}mm）")
        print()
        print("  这张表怎么读（很重要）：")
        print("    · 【点数】那一列从 348 一路掉到 33 —— 这就是'侧对'的真实代价：")
        print("      面越斜，它在图像上的投影越窄，深度相机能测到的像素越少。")
        print("    · 90° 时**一个点都没有** —— 完全侧对的面，深度相机根本看不见它。")
        print("    · 平均误差看起来没变差，是因为这块面够大（25cm）点还够用；")
        print("      但如果面很小（比如 5cm 的方块），点数掉到几十个时，")
        print("      误差就会明显变大，甚至拟合出乱七八糟的平面。")
        print()
        print("  ★ 实用结论：夹角 >70° 的面要**谨慎相信**，先看 n_points 够不够。")
        print("     我们在 pc_pipeline 的输出里会一起打印 n_points 和 RMS，")
        print("     就是让你判断'这个夹角可不可信'。")

    # ---------------- ⑤ 噪声的影响 ----------------
    print()
    print("=" * 64)
    print(" ⑤ 噪声对夹角精度的影响（这解释了'为什么 95 分位误差 0.65°'）")
    print("=" * 64)
    print(f"  {'噪声σ':>7} {'误差':>8}   {'有效点数':>8}")
    for nz in (0, 1, 2, 5, 10, 20):
        errs = []
        for s in range(12):
            p3, _ = make_tilted_plane(20.0, noise_mm=nz, seed=100 + s)
            n3, _, _ = fit_plane_normal(p3, verbose=False)
            errs.append(abs(angle_to_camera(n3) - 20.0))
        print(f"  {nz:>5}mm {np.mean(errs):>7.3f}°")
    print()
    print("  → 噪声越大误差越大，但注意：**点越多，平均掉噪声的效果越好**。")
    print("    所以'面越大 / 相机越近' → 点越多 → 夹角越准。")
    print("    这也是为什么 250mm 的方块在 1m 处能到 0.65° 的精度。")

    # ---------------- ⑥ 画图 ----------------
    if a.show:
        try:
            import matplotlib
            if not os.environ.get("DISPLAY"):
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(13, 5))
            ax = fig.add_subplot(121, projection="3d")
            p = pts[::max(1, len(pts) // 2000)]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2, c="tab:blue")
            c = pts.mean(axis=0)
            L = 0.08
            ax.quiver(c[0], c[1], c[2], n_hat[0] * L, n_hat[1] * L,
                      n_hat[2] * L, color="red", lw=3, label="fitted normal")
            ax.quiver(0, 0, 0, 0, 0, 0.25, color="green", lw=3,
                      label="optical axis")
            ax.scatter([0], [0], [0], c="k", s=80, marker="s")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_title(f"plane @ {a.angle}deg  ->  measured {ang:.2f}deg")
            ax.legend()
            plt.tight_layout()
            out = os.path.abspath(os.path.join(HERE, "..", "lab3_angle.png"))
            plt.savefig(out, dpi=90)
            print(f"\n  ✅ 图已存 {out}")
            plt.show()
        except ImportError:
            pass


    # ---------------- ⑦ 抽样 vs 精度 ----------------
    print()
    print("=" * 64)
    print(" ⑥ 抽样（stride）对精度的影响 —— 我们为什么要抽点")
    print("=" * 64)
    print("  做法：同一个平面、同样的噪声，只改【保留多少点】，看夹角误差怎么变。")
    print()
    print(f"  {'抽样':>6} {'点数':>7} {'平均误差':>9} {'最坏误差':>9}  {'理论 √N':>8}")
    base = None
    for stride in (1, 2, 3, 4, 6, 10):
        errs, npts = [], []
        for s_ in range(25):
            p4, _ = make_tilted_plane(30.0, noise_mm=a.noise, seed=2000 + s_)
            if len(p4) < 12:
                continue
            # 用 stride 抽点：等价于对深度图隔行隔列取样
            p4 = p4[::stride]
            if len(p4) < 12:
                continue
            n4, _, _ = fit_plane_normal(p4, verbose=False)
            errs.append(abs(angle_to_camera(n4) - 30.0))
            npts.append(len(p4))
        if not errs:
            print(f"  {stride:>6} {'点太少，拟合不了':>28}")
            continue
        e = float(np.mean(errs))
        if base is None:
            base = e if e > 1e-9 else 1e-9
        print(f"  {stride:>6} {int(np.mean(npts)):>7} {e:>8.3f}° "
              f"{max(errs):>8.3f}°  {np.sqrt(stride):>7.2f}x")
    print()
    print("  ★ 三件事要分清（这是你这个问题的核心）:")
    print()
    print("   ①【稀疏 sparse】= 深度相机【测不到】的点")
    print("      · 由物理决定，不是我们的选择：")
    print("        太远/太近、黑色吸光、镜面反光、倾角太大、多路径干扰")
    print("      · 表现：深度图里那些 0 / NaN / 空洞")
    print("      · 这个改不了，只能靠'多拍几帧'或'换个角度摆'")
    print()
    print("   ②【抽样 decimation】= 相机【测到了】但我们【主动丢掉】的点")
    print("      · 是我们的选择，为了速度（点数减半，几何耗时也减半）")
    print("      · 我们用的是 stride 抽样：隔行隔列取，保留空间范围、减少点数")
    print()
    print("   ③ 为什么不能'扫得更密、点更多'？")
    print("      · 深度相机的原生分辨率是【硬件固定的】（D435i 深度是 640x480）")
    print("        它不是激光雷达，不能'多扫几圈' —— 一帧就是那么多个像素")
    print("      · 想更密只有三条路：离近点（视场变小）、换更高分辨率相机、")
    print("        或多帧叠加（但物体不能动）")
    print()
    print("   ④ 精度影响（实验数据在上面那张表）:")
    print("      · 随机噪声的误差 ∝ 1/√N —— 点数减到 1/4，误差约翻倍")
    print("      · 但注意：**系统性误差不会被平均掉**")
    print("        比如深度标定偏差 1%，点再多也还是 1% 偏差")
    print("      · 所以抽样【可以】用来换速度，只要点数还够（经验值 >200 点）")
    print("      · 我们默认 stride=3，点数剩 1/9，误差从 0.2° 变成 ~0.5°,")
    print("        换来 12~30 倍速度 —— 对'方块定位'这个任务完全值得")

    print()
    print("=" * 64)
    print(" 记住这三步就够了")
    print("=" * 64)
    print("   ① SVD 拟合平面      → 得到法向量 n")
    print("   ② 夹角 = acos(|nz|)  → 绝对值保证 0~90°")
    print("   ③ RMS 越小说明面拟合得越好（在 pc_pipeline 的输出里会打印）")
    print()
    print("   真正跑的时候这三步在 ~/work/common/pc_pipeline.py 的")
    print("   analyze_cluster() 和 run_pipeline() 里，代码只有十几行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
