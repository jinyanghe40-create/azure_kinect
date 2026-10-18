#!/usr/bin/env python3
"""距离扫描：一次跑完 40/50/60/70/80cm，自动存文件、自动出表。

为什么要有它：
    手动一轮轮敲 test_precision.py，很容易像上次那样 ——
    文件名和实际距离对不上、被覆盖、或者忘了填 --tape。
    这个脚本会：
      ① 逐个距离提示你摆好方块（你只管量卷尺、按回车）
      ② 每轮存到独立的文件（不会互相覆盖）
      ③ 全部跑完自动调用 distance_report.py 出总表

用法（电脑上，先 source ROS）:
    source /opt/ros/humble/setup.bash

    # 默认扫 40/50/60/70/80 cm
    /usr/bin/python3 tools/sweep_distance.py --true-edge 0.25

    # 自定义距离
    /usr/bin/python3 tools/sweep_distance.py --true-edge 0.25 \
        --distances 0.4 0.5 0.6 0.8

要点:
    · 相机固定死，全程不要碰
    · 每轮**方块摆正（一个面朝相机）、尽量放在画面正中**
    · 卷尺量「相机镜头前表面 → 方块离相机最近的那个点」
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=float, nargs="+",
                    default=[0.4, 0.5, 0.6, 0.7, 0.8],
                    help="要扫的距离(m)，默认 0.4 0.5 0.6 0.7 0.8")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--true-edge", type=float, default=0.25)
    ap.add_argument("--outdir", default="/tmp/sweep")
    ap.add_argument("--timeout", type=float, default=240)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    jsons = []
    print("=" * 68)
    print(" 距离扫描")
    print("=" * 68)
    print(f" 距离: {' '.join(f'{d*100:.0f}cm' for d in a.distances)}")
    print(f" 每轮 {a.frames} 帧，方块真实边长 {a.true_edge*100:.1f}cm")
    print(f" 结果存到 {a.outdir}/")
    print()
    print(" ★ 相机全程不要碰")
    print(" ★ 每轮方块摆正（一个面朝相机），尽量放在画面正中")
    print(" ★ 卷尺量「相机镜头 → 方块离相机最近的那个点」")
    print("=" * 68)

    for d in a.distances:
        mm = int(round(d * 1000))
        out = os.path.join(a.outdir, f"d{mm:04d}.json")
        print()
        print(f"─── 第 {a.distances.index(d)+1}/{len(a.distances)} 轮："
              f"{d*100:.0f} cm ───")
        try:
            input(f"  把方块放到 {d*100:.0f} cm 处，用卷尺量好，按回车开始采样…")
        except (EOFError, KeyboardInterrupt):
            print("\n（中止）")
            break
        cmd = [sys.executable, os.path.join(HERE, "test_precision.py"),
               "--frames", str(a.frames),
               "--true-edge", str(a.true_edge),
               "--tape", str(d),
               "--timeout", str(a.timeout),
               "--json", out]
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode == 0 and os.path.exists(out):
            jsons.append(out)
        else:
            print(f"  ⚠️ 这一轮没成功（返回码 {r.returncode}），跳过")

    if len(jsons) < 2:
        print("\n❌ 成功的数据不足 2 组，没法汇总。")
        return 1

    print("\n\n" + "=" * 68)
    print(" 汇总")
    print("=" * 68)
    return subprocess.run([sys.executable,
                           os.path.join(HERE, "distance_report.py")] + jsons,
                          cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
