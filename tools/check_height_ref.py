#!/usr/bin/env python3
"""量【桌面平面】在相机系里的高度 —— 判断 Y 到底该不该等于"离地高度"。

为什么需要：
    报告里的 face_center 是 (X, Y, Z)，其中
        ★ Y 是【相对深度相机光轴】的，不是【相对桌面】的。
    所以"卷尺量 16.5cm、相机报 8cm"这件事，本身不构成"不准" ——
    要先知道光轴离桌面多高，两个数才可比：
        面心离桌面高度 = (桌面平面的 Y) − (面心的 Y)
                        └ 光轴到桌面的距离      └ 面心到光轴的距离（负数=在上方）

    这个量**不用卷尺也能测**：桌面是画面里最大的一块平面，
    点云里 Y 的直方图会有一个又高又窄的尖峰，峰位就是桌面所在的高度。

判读：
    桌面Y ≈ +0.082  → 光轴在桌面上方 8.2cm
                      → 面心离桌面 = 0.082+0.083 = 16.5cm  ✅ 和卷尺对上，Y 没坏
    桌面Y ≈ +0.025  → 光轴只离桌面 2.5cm
                      → 面心离桌面 = 0.025+0.083 = 10.8cm  ❌ 真少了 ~6cm，要查

    另外：如果桌面平面的法向明显偏离 ±Y 轴，说明相机是【俯仰】的。
    俯仰时 Y 轴不再竖直，"Y 的差"就不再等于"高度差"，
    这时候先加 --rotate 之外还得按重力方向重新定义竖直。

用法（电脑上，和 test_precision.py 一样先 source ROS）:
    source /opt/ros/humble/setup.bash
    /usr/bin/python3 tools/check_height_ref.py --frames 15
"""
import argparse
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=15,
                    help="采集多少帧点云（默认 15）")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--bin", type=float, default=0.003,
                    help="Y 直方图的分箱宽度(m)，默认 3mm")
    ap.add_argument("--band", type=float, default=0.020,
                    help="拟合平面时取的带宽(m)，默认 20mm")
    a = ap.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                               DurabilityPolicy)
        from sensor_msgs.msg import PointCloud2
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError as e:
        print(f"❌ 缺 ROS 环境：{e}\n   先 source /opt/ros/humble/setup.bash")
        return 1

    rclpy.init()
    node = rclpy.create_node("check_height_ref")
    qos = QoSProfile(depth=10)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.VOLATILE

    clouds, marks = [], []
    node.create_subscription(PointCloud2, "/camera/points",
                             lambda m: clouds.append(m), qos)
    node.create_subscription(MarkerArray, "/detections",
                             lambda m: marks.append(m), qos)

    print(f"采集 {a.frames} 帧点云（最多等 {a.timeout:.0f}s）…", flush=True)
    t0 = time.time()
    while len(clouds) < a.frames and time.time() - t0 < a.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)

    if not clouds:
        print("\n❌ 一帧 /camera/points 都没收到。")
        print("   ① 板子上 run_azure.py 在跑吗（它才是数据源）")
        print("   ② 电脑上 ros_bridge/cloud_bridge.py 在跑吗")
        print("   ③ 话题名对不对：ros2 topic list | grep camera")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    try:
        from sensor_msgs_py.point_cloud2 import read_points_numpy
        def to_xyz(m):
            p = read_points_numpy(m, field_names=("x", "y", "z"),
                                  skip_nans=True)
            return np.asarray(p, dtype=np.float64).reshape(-1, 3)
    except ImportError:
        from sensor_msgs_py.point_cloud2 import read_points
        def to_xyz(m):
            g = read_points(m, field_names=("x", "y", "z"), skip_nans=True)
            return np.asarray(list(g), dtype=np.float64).reshape(-1, 3)

    print(f"✅ 收到 {len(clouds)} 帧，开始量…\n")

    table_ys, tilts, ns, npts = [], [], [], []
    for m in clouds:
        pts = to_xyz(m)
        if len(pts) < 500:
            continue
        npts.append(len(pts))
        y = pts[:, 1]

        # ---- 桌面 = Y 直方图的主峰（又高又窄那个）----
        lo, hi = np.percentile(y, [0.2, 99.8])
        edges = np.arange(lo, hi + a.bin, a.bin)
        if len(edges) < 4:
            continue
        hist, edges = np.histogram(y, bins=edges)
        i = int(np.argmax(hist))
        peak = 0.5 * (edges[i] + edges[i + 1])

        # ---- 在主峰附近取一条带，拟合平面，看法向 ----
        band = pts[np.abs(y - peak) < a.band]
        if len(band) < 100:
            continue
        c = band.mean(axis=0)
        u, s, vt = np.linalg.svd(band - c, full_matrices=False)
        n = vt[2]
        if n[1] < 0:                       # 统一朝上（+Y 是"下"，朝上即 n_y<0）
            n = -n
        table_ys.append(float(peak))
        ns.append(n)
        # 法向与 ±Y 轴的夹角 → 相机的俯仰/横滚
        tilts.append(float(np.degrees(np.arccos(
            min(1.0, abs(float(n[1])))))))

    node.destroy_node()
    rclpy.shutdown()

    if not table_ys:
        print("❌ 没能拟合出桌面平面（点太少？）")
        return 1

    ty = float(np.median(table_ys))
    tlt = float(np.median(tilts))
    n_mean = np.mean(np.array(ns), axis=0)
    n_mean /= np.linalg.norm(n_mean)

    print("=" * 70)
    print(f" 点云规模：{int(np.median(npts))} 点/帧，共 {len(table_ys)} 帧有效")
    print(f" 桌面平面 Y = {ty:+.4f} m   （抖动 ±{np.std(table_ys)*1000:.1f} mm）")
    print(f" 桌面法向   = ({n_mean[0]:+.3f}, {n_mean[1]:+.3f}, {n_mean[2]:+.3f})")
    print(f" 法向偏离 Y 轴 {tlt:.2f}°   ← 这就是相机的俯仰/横滚角")
    print("=" * 70)

    if tlt < 2.0:
        print(f" ✅ 相机基本是【水平】的（偏离 {tlt:.1f}°），Y 轴 ≈ 竖直方向")
    else:
        print(f" ⚠️ 相机【不是水平的】，偏离 Y 轴 {tlt:.1f}°")
        print(f"    → Y 轴不再竖直，Y 的差值 ≠ 高度差（差个 cos{tlt:.0f}° ≈ "
              f"{np.cos(np.radians(tlt)):.2f} 倍，" +
              (f"还混进去 X/Z 的分量）" if tlt > 5 else "）"))

    print()
    print(f" ★★ 光轴到桌面的距离 = {ty*100:+.1f} cm")
    print("    （Y 正向朝下 → 桌面 Y 为正，说明桌面在光轴下方这么远）")
    print()

    # ---- 用 marker 里的面心做换算 ----
    face = None
    if marks:
        for mk in marks[-1].markers:
            if mk.type == Marker.TEXT_VIEW_FACING:
                face = (mk.pose.position.x, mk.pose.position.y,
                        mk.pose.position.z)
                break
    if face is None:
        face = (-0.0294, -0.0827, 0.5923)          # 上次报告的均值兜底
        print(" （没拿到 /detections 的 marker，用上次报告的面心均值）")

    h_above_axis = ty - face[1]        # 面心在光轴上方多少（Y 向下为正）
    print(" " + "-" * 66)
    print(f" 面心 Y = {face[1]:+.4f} m  （在光轴{'上' if face[1] < 0 else '下'}方 "
          f"{abs(face[1])*100:.1f} cm）")
    print(f" 光轴高于桌面 {ty*100:+.1f} cm")
    print(f" ⇒ 面心高于桌面 = {h_above_axis*100:.1f} cm")
    print(" " + "-" * 66)
    print()
    print(" 把你卷尺量到的数字和上面这个【面心高于桌面】对比：")
    print("   · 对得上 → Y 没问题，之前只是拿 Y 直接当'离地高度'看了")
    print("   · 对不上 → 记下差多少，再查（先跑 check_gravity.py 确认相机水平）")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
