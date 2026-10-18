#!/usr/bin/env python3
"""鲁棒性扫描：换个姿态/距离，看哪些字段还准、哪些掉了。

为什么需要它：
    "任意摆放都能出类别/位置/尺寸吗" 这个问题，靠猜没用 ——
    不同字段的可靠性差很多：
        · 类别 / 面心 / 法向     → 基本任意摆放都能出
        · 尺寸（边长）           → 要求【那个面完整可见】，被裁就量小
        · 体心                   → 要求【同时看到 3 个面】才是精确解
    这个脚本让你把方块摆成各种姿态，逐轮采样，最后出一张表，
    直接告诉你"哪种姿态下哪个字段不能用"。

用法（电脑上，先 source ROS）:
    source /opt/ros/humble/setup.bash
    cd ~/jetson-deploy

    # 默认提示 6 种姿态（你自己转方块）
    /usr/bin/python3 tools/sweep_pose.py --true-edge 0.25

    # 自定义提示语
    /usr/bin/python3 tools/sweep_pose.py --true-edge 0.25 \
        --poses "正对相机" "左转30度" "右转30度" "上翘30度" "斜45度" "立起来"

要点:
    · 每轮之间**只动方块**，相机不要碰
    · 每轮尽量保持距离不变（或者记下距离，看它随距离怎么变）
    · 方块保持在画面内、别被裁
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

DEFAULT_POSES = ["正对相机（一个面朝相机）", "左转约30°", "右转约30°",
                 "上翘约30°", "斜着45°（看到两个面）", "立起来（看到一个侧面）"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", nargs="+", default=DEFAULT_POSES)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--true-edge", type=float, default=0.25)
    ap.add_argument("--outdir", default="/tmp/sweep_pose")
    ap.add_argument("--timeout", type=float, default=150)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    jsons = []
    print("=" * 72)
    print(" 姿态鲁棒性扫描")
    print("=" * 72)
    print(f" 共 {len(a.poses)} 种姿态，每轮 {a.frames} 帧")
    print(f" 方块真实边长 {a.true_edge*100:.1f} cm")
    print(" ★ 每轮只动方块，相机不要碰；方块保持在画面内")
    print("=" * 72)

    for i, pose in enumerate(a.poses, 1):
        out = os.path.join(a.outdir, f"pose{i:02d}.json")
        print(f"\n─── 第 {i}/{len(a.poses)} 轮：{pose} ───")
        try:
            input("  摆好之后按回车开始采样…")
        except (EOFError, KeyboardInterrupt):
            print("\n（中止）")
            break
        cmd = [sys.executable, os.path.join(HERE, "test_precision.py"),
               "--frames", str(a.frames), "--true-edge", str(a.true_edge),
               "--timeout", str(a.timeout), "--json", out]
        r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL)
        if r.returncode == 0 and os.path.exists(out):
            jsons.append((pose, out))
        else:
            print("  ⚠️ 这轮没采到数据")

    if not jsons:
        print("\n❌ 一轮都没成功")
        return 1

    # ---- 汇总 ----
    import json
    import numpy as np
    print("\n\n" + "=" * 96)
    print(" 姿态鲁棒性汇总")
    print("=" * 96)
    print(f"{'姿态':<22}{'面数':>6}{'边长(cm)':>10}{'边长误差':>10}"
          f"{'面心抖动':>10}{'角度抖动':>10}{'质量':>8}{'体心可信':>10}")
    print("-" * 96)
    ok_cnt = 0
    for pose, path in jsons:
        r = json.load(open(path))
        hist = r.get("n_faces_hist") or {}
        # 最常见的那种面数
        nf = max(hist.items(), key=lambda kv: kv[1])[0] if hist else "?"
        edge = (r.get("edge_mean_m") or 0) * 100
        eerr = r.get("edge_error_pct")
        jit = r.get("面心_std_norm_mm") or r.get("face_center_std_norm_mm")
        ajit = r.get("angle_std_deg")
        # quality 不在 json 里（那是串口字段），这里用实测抖动自己判
        q = "GOOD" if (jit is not None and jit < 5) else (
            "FAIR" if (jit is not None and jit < 20) else "COARSE")
        cube_ok = "✅" if str(nf) == "3" else "❌"
        if q == "GOOD" and cube_ok == "✅":
            ok_cnt += 1
        print(f"{pose:<22}{str(nf):>6}{edge:>10.2f}"
              f"{(f'{eerr:+.2f}%' if eerr is not None else '?'):>10}"
              f"{(f'±{jit:.2f}' if jit is not None else '?'):>10}"
              f"{(f'±{ajit:.3f}°' if ajit is not None else '?'):>10}"
              f"{q:>8}{cube_ok:>10}")
    print("-" * 96)
    print(f"  {ok_cnt}/{len(jsons)} 种姿态下「位置准 + 体心可信」")
    print()
    print(" 怎么读：")
    print("   · 边长误差 —— 超过 ±3% 基本是【那个面被裁了】或只露一部分")
    print("   · 面心抖动 —— 大了说明面上点太少（太远）或面不完整")
    print("   · 体心可信 —— 只有看到 3 个面才是精确解；1~2 个面是估的")
    print("   · 电控侧对应字段：quality / cube_ok / edge_src")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
