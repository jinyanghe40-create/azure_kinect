#!/usr/bin/env python3
"""
在电脑上看点云文件（.ply）—— 不需要装 RViz / MeshLab / Open3D

用法:
    python3 view_ply.py 点云.ply                  # 看一个
    python3 view_ply.py a.ply b.ply c.ply         # 并排看多个（对比管线每一步）
    python3 view_ply.py --dir out/                # 看一个目录里所有 ply
    python3 view_ply.py x.ply --elev 90           # 换个视角

鼠标：左键拖动=旋转  右键拖动=平移  滚轮=缩放
（关掉窗口就退出）

为什么用 matplotlib 而不是 Open3D：
    Open3D 在 aarch64 上装起来很麻烦，而且我们要看的是"几千到十万个点"
    这种规模，matplotlib 完全够用，还不用装任何东西。
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg") if not os.environ.get("DISPLAY") else None
import matplotlib.pyplot as plt                     # noqa: E402


# PLY 属性类型 -> numpy 类型
PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply(path):
    """
    读 PLY —— ASCII 和【二进制】都要支持。

    ⚠️ 我们的 write_ply 写的是 binary_little_endian（体积小 10 倍），
       所以只支持 ASCII 的读法会直接读成乱码。
    """
    with open(path, "rb") as f:
        raw = f.read()

    # ---- 解析头部（头部一定是 ASCII）----
    end = raw.find(b"end_header")
    if end < 0:
        raise SystemExit(f"❌ {path} 不是 PLY（找不到 end_header）")
    nl = raw.find(b"\n", end)
    header = raw[:nl].decode("ascii", errors="replace")
    body = raw[nl + 1:]

    fmt = "ascii"
    n_vert = 0
    props = []
    in_vertex = False
    for line in header.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            in_vertex = (t[1] == "vertex")
            if in_vertex:
                n_vert = int(t[2])
        elif t[0] == "property" and in_vertex:
            props.append((t[2], t[1]))          # (名字, 类型)

    names = [p[0] for p in props]
    if "x" not in names:
        raise SystemExit(f"❌ {path} 里没有顶点坐标")

    if fmt == "ascii":
        rows = body.decode("ascii", errors="replace").split("\n")[:n_vert]
        arr = np.array([[float(x) for x in r.split()] for r in rows if r.strip()],
                       np.float32)
        pts = arr[:, :3]
        col = None
        if "red" in names:
            i = names.index("red")
            col = arr[:, i:i + 3].astype(np.float32)
            if col.max() > 1.0:
                col /= 255.0
        return pts, col

    if fmt != "binary_little_endian":
        raise SystemExit(f"❌ 暂不支持 {fmt} 格式")

    # ---- 二进制：按头部声明的属性构造 numpy dtype ----
    dt = np.dtype([(p[0], "<" + PLY_TYPES[p[1]]) for p in props])
    arr = np.frombuffer(body, dtype=dt, count=n_vert)
    pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    col = None
    if "red" in names:
        col = np.stack([arr["red"], arr["green"], arr["blue"]],
                       axis=1).astype(np.float32)
        if col.max() > 1.0:
            col /= 255.0
    return pts, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="ply 文件")
    ap.add_argument("--dir", default=None, help="看这个目录里所有 ply")
    ap.add_argument("--elev", type=float, default=25)
    ap.add_argument("--azim", type=float, default=-60)
    ap.add_argument("--size", type=float, default=2.0, help="点大小")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="view_ply.png", help="输出图片路径")
    ap.add_argument("--show", action="store_true",
                    help="弹出交互窗口（会阻塞，直到你关掉它）")
    ap.add_argument("--cjk-font", default=None,
                    help="想在图里显示中文就指定字体，例如 'Noto Sans CJK SC'；"
                         "不指定时图里请用英文（matplotlib 遇到中文会全系统搜字体，极慢）")
    a = ap.parse_args()

    if a.cjk_font:
        plt.rcParams["font.sans-serif"] = [a.cjk_font] + \
            list(plt.rcParams["font.sans-serif"])
        plt.rcParams["axes.unicode_minus"] = False

    files = list(a.files)
    if a.dir:
        files += sorted(glob.glob(os.path.join(a.dir, "*.ply")))
    if not files:
        print("用法: python3 view_ply.py 点云.ply [更多.ply] [--dir 目录]")
        return 1

    n = len(files)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    if a.title:
        fig.suptitle(a.title, fontsize=14)

    for k, fp in enumerate(files):
        pts, col = read_ply(fp)
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        # 只画一部分点，否则 matplotlib 会卡
        step = max(1, len(pts) // 8000)   # 最多画 8000 点，再多 matplotlib 会卡死
        p = pts[::step]
        c = col[::step] if col is not None else None
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, s=a.size, marker=".")
        # 相机在原点，光轴朝 +Z：画三条坐标轴方便理解
        m = max(0.3, float(np.abs(pts).max()) if len(pts) else 1.0)
        ax.plot([0, m], [0, 0], [0, 0], "r-", lw=2)
        ax.plot([0, 0], [0, m], [0, 0], "g-", lw=2)
        ax.plot([0, 0], [0, 0], [0, m], "b-", lw=2)
        ax.text(m, 0, 0, " X", color="r")
        ax.text(0, m, 0, " Y", color="g")
        ax.text(0, 0, m, " Z(axis)", color="b")
        ax.scatter([0], [0], [0], c="k", s=60, marker="s")   # 相机位置
        ax.set_title(f"{os.path.basename(fp)}\n{len(pts)} pts", fontsize=10)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.view_init(elev=a.elev, azim=a.azim)
        # 三轴等比例，否则会被拉变形
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    plt.tight_layout()
    out = a.out
    plt.savefig(out, dpi=90)
    print(f"✅ 已存图 {out}（{len(files)} 个点云）")
    print(f"   在 VS Code 里打开它就能看（点文件 → 右侧预览）")
    if a.show:
        # ⚠️ plt.show() 会【阻塞】直到你关掉窗口。
        #    而且 DISPLAY 存在不代表你能看到那个窗口（比如从 SSH/脚本里跑），
        #    所以默认不开窗口，只存图。
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
