#!/usr/bin/env python3
"""把多轮"距离测试"的 json 汇总成一张表，并把误差拆成【尺度】和【偏移】。

为什么必须拆开：
    直接拿 |P| 和卷尺比，会混进一个**固定偏移** ——
    深度相机的光心并不在相机外壳的前沿（在里面几个 mm 到十几 mm），
    所以你会看到一个"所有距离都偏同一个量"的系统误差。
    这个偏移**不影响相对测量**，也不是标定问题。

    把 |P| 对卷尺读数拟合一条直线：  |P| = k·d + b
        k（斜率）= 尺度因子 → **这个才是精度**，k=1 就是完美
        b（截距）= 光心到测量基准的偏移 → 常数，无害

用法:
    # 先在不同距离各跑一轮（每轮都要 --json 和 --tape）
    python3 tools/test_precision.py --frames 120 --true-edge 0.25 \\
        --tape 0.30 --json /tmp/d30.json
    ... 移动方块 ...
    python3 tools/test_precision.py --frames 120 --true-edge 0.25 \\
        --tape 0.40 --json /tmp/d40.json
    ...

    # 再汇总
    python3 tools/distance_report.py /tmp/d30.json /tmp/d40.json /tmp/d50.json
"""
import json
import sys

import numpy as np


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    rows = []
    for path in sys.argv[1:]:
        try:
            r = json.load(open(path))
        except Exception as e:
            print(f"⚠️ 读不了 {path}: {e}")
            continue
        d = r.get("tape_m")
        # ★ 优先用"最近点距离" —— 卷尺量的就是它
        p = r.get("nearest_mean_m") or r.get("face_center_mean_norm_m")
        if d is None or p is None:
            print(f"⚠️ {path} 缺 tape_m 或 face_center_mean_norm_m，跳过")
            continue
        rows.append({
            "file": path, "tape": float(d), "z": float(p),
            "by": "最近点" if r.get("nearest_mean_m") else "面心",
            # 报告里这两个键是中文名（历史原因），两种都认
            "jitter": (r.get("face_center_std_norm_mm")
                       or r.get("面心_std_norm_mm")),
            "cube_jit": (r.get("cube_center_std_norm_mm")
                         or r.get("体心_std_norm_mm")),
            "ang": r.get("angle_mean_deg"),
            "ang_std": r.get("angle_std_deg"),
            "edge": r.get("edge_mean_m"),
            "edge_err": r.get("edge_error_pct"),
            "n": r.get("n_frames"),
            "drift": r.get("drift_face_mm"),
        })
    if not rows:
        print("❌ 没有可用数据")
        return 1
    rows.sort(key=lambda x: x["tape"])

    print("═" * 78)
    print(" 距离准确度汇总")
    print("═" * 78)
    print(f"{'卷尺(m)':>9} {'管线|P|(m)':>11} {'用的点':>7} {'差(mm)':>9} {'差(%)':>8} "
          f"{'抖动(mm)':>9} {'角度(°)':>9} {'边长(‰)':>9} {'帧数':>6}")
    print("-" * 78)
    for r in rows:
        e = r["z"] - r["tape"]
        print(f"{r['tape']:9.4f} {r['z']:11.4f} {r['by']:>7} {e*1000:+9.1f} "
              f"{e/r['tape']*100:+8.2f} "
              f"{(r['jitter'] if r['jitter'] is not None else float('nan')):9.1f} "
              f"{(r['ang'] if r['ang'] is not None else float('nan')):9.2f} "
              f"{(r['edge_err'] if r['edge_err'] is not None else float('nan')):9.2f} "
              f"{(r['n'] or 0):6d}")

    d = np.array([r["tape"] for r in rows])
    z = np.array([r["z"] for r in rows])

    print("\n" + "═" * 78)
    if len(rows) >= 3 and float(np.ptp(d)) > 0.05:
        k, b = np.polyfit(d, z, 1)
        pred = k * d + b
        resid = z - pred
        rms = float(np.sqrt(np.mean(resid ** 2)))
        print(" 把误差拆成【尺度】和【偏移】：  |P| = k·d + b")
        print(f"    k（尺度因子） = {k:.5f}      → 尺度误差 {(k-1)*100:+.2f}%")
        print(f"    b（固定偏移） = {b*1000:+.1f} mm   ← 深度光心到相机外壳前沿的距离")
        print(f"    拟合残差 RMS  = {rms*1000:.1f} mm  ← 扣掉尺度和偏移后，剩下的就是真噪声")
        print()
        print("  怎么读：")
        if abs(k - 1) < 0.01:
            print(f"    ✅ 尺度误差 {abs(k-1)*100:.2f}% < 1% —— 测距很准")
        elif abs(k - 1) < 0.03:
            print(f"    ✅ 尺度误差 {abs(k-1)*100:.2f}% < 3% —— 可用")
        else:
            print(f"    ⚠️ 尺度误差 {abs(k-1)*100:.2f}% 偏大 —— 查相机内参标定")
        print(f"    · b = {b*1000:+.1f}mm 是**常数偏移**，不影响相对测量，"
              f"要用绝对距离时减掉它即可")
        if rms * 1000 > 10:
            print(f"    · 残差 {rms*1000:.1f}mm 偏大 —— 每一轮的抖动也不小，"
                  f"先确认相机固定住了")
    else:
        print(" 数据点不够（需要 ≥3 个不同距离，跨度 >5cm）才拟合直线。")
        print(" 但可以看**增量**：方块移动 10cm，|P| 应该也变 10cm。")
        for i in range(1, len(rows)):
            dz = z[i] - z[i - 1]
            dd = d[i] - d[i - 1]
            print(f"    卷尺 +{dd*100:6.1f}cm  →  |P| +{dz*100:6.1f}cm   "
                  f"误差 {(dz-dd)*1000:+6.1f}mm  尺度 {(dz/dd-1)*100:+.2f}%")

    # 抖动随距离的变化
    js = [(r["tape"], r["jitter"]) for r in rows if r["jitter"]]
    if len(js) >= 2:
        print("\n 抖动随距离的变化（检查是否符合 z² 规律）：")
        for t, j in js:
            pred = j  # 只展示
            print(f"     {t:.2f}m  →  面心抖动 ±{j:5.1f}mm")
        print("     参考：深度噪声 σz ∝ z²，所以抖动应该涨得比距离快")
    print("═" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
