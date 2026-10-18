#!/usr/bin/env python3
"""
============================================================
 教学 ①：点云到底是怎么来的？
============================================================
把"深度图 → 点云"这一层彻底讲清楚，而且每一步都打印真实数字，
你可以拿计算器自己验一遍。

用法:
    python3 lab2_pointcloud.py --fake            # 用合成深度图（不接相机）
    python3 lab2_pointcloud.py --fake --show     # 顺便画出点云
    python3 lab2_pointcloud.py --npy depth.npy --intr '{"fx":918,"fy":919,"cx":637,"cy":351}'
    python3 lab2_pointcloud.py --rs              # 板子上直接抓一帧真实深度

输出:
    lab2_step1_depth.png    深度图（伪彩色）
    lab2_final.ply          点云文件（可以用 view_ply.py 看）

------------------------------------------------------------
核心就三行公式（针孔相机模型）：

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = 深度值

    u,v = 像素坐标（列, 行）
    fx,fy,cx,cy = 相机内参（出厂标定好的，fx/fy 约等于焦距的像素数）
    Z = 这个像素测到的深度（米）

就这么简单 —— **深度图 + 内参 = 点云**。
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from depth_to_cloud import depth_to_points, write_ply   # noqa: E402


def make_fake(shape=(480, 640)):
    """合成一帧深度：一个倾斜的桌面 + 上面一个正方体"""
    H, W = shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    z = 1.0 + 0.15 * (u - W / 2) / (W / 2)      # 带倾斜的桌面
    # 中间放一个 25cm 的立方体（正面比桌面近 0.25m）
    m = (np.abs(u - W / 2) < 90) & (np.abs(v - H / 2) < 90)
    z[m] -= 0.25
    return z.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--npy", default=None, help="深度图 .npy（米）")
    ap.add_argument("--rs", action="store_true", help="从 RealSense 抓一帧")
    ap.add_argument("--intr", default=None, help="内参 JSON")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    # ---------------- ① 拿到深度图 + 内参 ----------------
    print("=" * 62)
    print(" ① 先有【深度图】和【内参】")
    print("=" * 62)
    if a.rs:
        from realsense_capture import RealSenseSource
        src = RealSenseSource((640, 480), (1280, 720))
        depth, color, intr, _ = src.get()
        src.close()
        print(f"  深度图: {depth.shape[1]}x{depth.shape[0]}（RealSense D435i）")
    else:
        depth = make_fake() if not a.npy else np.load(a.npy).astype(np.float32)
        intr = {"fx": 918.4, "fy": 919.2, "cx": 637.6, "cy": 351.1}
        if a.intr:
            intr = json.loads(a.intr)
        print(f"  深度图: {depth.shape[1]}x{depth.shape[0]}"
              f"{'（合成）' if not a.npy else '（从文件读）'}")

    print(f"  内参  : fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
          f"cx={intr['cx']:.1f} cy={intr['cy']:.1f}")
    print()
    print("  深度图长什么样：它就是个二维数组，每个格子存一个距离（米）")
    print(f"    形状 {depth.shape}，最小值 {np.nanmin(depth):.3f}m，"
          f"最大值 {np.nanmax(depth):.3f}m")
    print()
    print("  深度图中间那一小块（数字是米）：")
    H, W = depth.shape
    for v in range(H // 2 - 2, H // 2 + 3):
        print("    " + "  ".join(f"{depth[v, u]:.3f}"
                                 for u in range(W // 2 - 3, W // 2 + 4)))

    # ---------------- ② 一个像素怎么变成一个 3D 点 ----------------
    print()
    print("=" * 62)
    print(" ② 一个像素 → 一个 3D 点（手算给你看）")
    print("=" * 62)
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    print("  公式（针孔相机模型）:")
    print("      X = (u - cx) * Z / fx")
    print("      Y = (v - cy) * Z / fy")
    print("      Z = 深度值")
    print()
    print("  直观理解：")
    print("      u - cx   = 这个像素离画面中心偏了多少像素")
    print("      除以 fx  = 把'像素'换算成'比例'（fx 就是焦距有多少像素）")
    print("      乘以 Z   = 越远的东西，同样的像素偏移对应越大的实际距离")
    print("      → 这就是'近大远小'的数学形式")
    print()
    for (u, v) in [(W // 2, H // 2), (W // 2 + 100, H // 2),
                   (W // 2, H // 2 + 80)]:
        Z = float(depth[v, u])
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        print(f"  像素 (u={u}, v={v})  深度 Z={Z:.3f}m")
        print(f"      X = ({u} - {cx:.1f}) × {Z:.3f} / {fx:.1f} "
              f"= {u - cx:.0f} × {Z:.3f} / {fx:.1f} = {X:+.4f} m")
        print(f"      Y = ({v} - {cy:.1f}) × {Z:.3f} / {fy:.1f} "
              f"= {v - cy:.0f} × {Z:.3f} / {fy:.1f} = {Y:+.4f} m")
        print(f"      → 3D 点 ({X:+.4f}, {Y:+.4f}, {Z:.4f}) 米")
        print()
    print("  ⚠️ 注意：点云是在【相机坐标系】里的 ——")
    print("     原点 = 相机的光心")
    print("     +Z   = 光轴方向（镜头正前方）")
    print("     +X   = 画面右方    +Y = 画面下方")

    # ---------------- ③ 整张图一起算 = 点云 ----------------
    print()
    print("=" * 62)
    print(" ③ 整张图所有像素一起算 = 点云")
    print("=" * 62)
    pts, valid, (X, Y, Z) = depth_to_points(depth, intr)
    print(f"  像素总数      : {depth.size:,}")
    print(f"  有效点（深度>0）: {int(valid.sum()):,}  ({100*valid.sum()/depth.size:.1f}%)")
    print(f"  点云形状      : {pts.shape}   （每行是一个 (X,Y,Z)）")
    print()
    print("  为什么会有无效点？")
    print("      深度相机测不到的地方（太近/太远/反光/纯黑）会输出 0 或 NaN，")
    print("      这些点必须丢掉，否则会得到 (0,0,0) 这种假点，把算法带偏。")

    # ---------------- ④ 存下来 ----------------
    print()
    print("=" * 62)
    print(" ④ 存成文件（这就是「点云文件」）")
    print("=" * 62)
    out_ply = os.path.join(HERE, "..", "lab2_final.ply")
    out_ply = os.path.abspath(out_ply)
    write_ply(out_ply, pts[valid])
    print(f"  ✅ 点云 → {out_ply}")
    print("     格式是 PLY（我们写的是二进制版，体积比文本版小 10 倍）。")
    print("     头部是纯文本，可以直接看：")
    with open(out_ply, "rb") as f:
        head = f.read(160).decode("ascii", errors="replace")
    for l in head.split("\n")[:8]:
        print("        " + l.rstrip())
    print("        （后面就是二进制数据了，记事本打不开，用查看器看）")
    print()
    print("  拿电脑上看（在电脑上敲）:")
    print(f"     python3 ~/kinect/view_ply.py {out_ply}")

    # ---------------- ⑤ 存深度图伪彩色 ----------------
    try:
        import matplotlib
        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        png = os.path.abspath(os.path.join(HERE, "..", "lab2_depth.png"))
        plt.figure(figsize=(8, 5))
        plt.imshow(depth, cmap="jet")
        plt.colorbar(label="depth (m)")
        plt.title("depth image  (each pixel = a distance)")
        plt.xlabel("u (pixel column)"); plt.ylabel("v (pixel row)")
        plt.tight_layout(); plt.savefig(png, dpi=90)
        print(f"  ✅ 深度图伪彩色 → {png}")
        if a.show:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(121, projection="3d")
            p = pts[valid][::max(1, int(valid.sum()) // 20000)]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c=p[:, 2], cmap="jet")
            ax.scatter([0], [0], [0], c="k", s=80, marker="s")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
            ax.set_title("point cloud (camera coords)")
            ax2 = fig.add_subplot(122)
            ax2.imshow(depth, cmap="jet")
            ax2.set_title("depth image")
            plt.tight_layout(); plt.show()
    except ImportError:
        pass

    print()
    print("=" * 62)
    print(" 记住这一句就够了")
    print("=" * 62)
    print("   ★ 点云 = 深度图 + 内参，逐像素套一个公式算出来的。")
    print("     之后的去桌面/聚类/拟合面/算法向量/算夹角，全都只跟点云打交道，")
    print("     跟'用的是哪个相机'再没关系 —— 这就是换相机不用改算法的原因。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
