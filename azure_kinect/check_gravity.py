#!/usr/bin/env python3
"""用 Azure Kinect 的【加速度计】测出"下"在相机坐标系的哪个方向 —— 一锤定音。

为什么需要：
    "Y 是朝上还是朝下"这种问题，看照片容易看错（场景里有桌椅、箱子、
    还有人为摆歪的东西，很容易当作参考）。但**重力骗不了人**。
    Azure Kinect 自带 IMU，加速度计静止时读到的就是重力向量，
    它的符号直接告诉你"下"在相机系的哪一边。

判读：
    ay > 0  → 相机的 +Y 指向地面  →  就是标准的"光学系（+Y 向下）"约定 ✅
    ay < 0  →  相机的 +Y 指向上方  →  相机装反了（或约定被翻过）

用法（板子上，需要有 X）:
    DISPLAY=:1 python3 ~/work/azure_kinect/check_gravity.py
"""
import ctypes
import os
import sys
import time
from ctypes import POINTER, byref, c_int, c_size_t, c_uint32, c_void_p

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from azure_capture import (AzureSource, K4A_CALIBRATION_TYPE_DEPTH,  # noqa
                           K4A_CALIBRATION_TYPE_COLOR)
# IMU 的坐标系和深度相机系不是同一个！加速度计的读数要先转过去。
K4A_CALIBRATION_TYPE_GYRO = 2
K4A_CALIBRATION_TYPE_ACCEL = 3


class K4AFloat3(ctypes.Union):
    """k4a_float3_t：3 个 float（x,y,z 和 xyz[3] 是同一块内存）"""
    class _XYZ(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float),
                    ("z", ctypes.c_float)]
    _fields_ = [("xyz", ctypes.c_float * 3), ("v", _XYZ)]


class K4AImuSample(ctypes.Structure):
    """k4a_imu_sample_t

    注意 C 里的字段顺序和对齐：
        float    temperature            @0   (4)
        float3   acc_sample             @4   (12)
        uint64   acc_timestamp_usec     @16  (8)
        float3   gyro_sample            @24  (12)
        uint64   gyro_timestamp_usec    @40  (8)   ← 8 字节对齐要补 4 字节
    总共 48 字节。ctypes 会自动按 C 规则对齐，所以直接照抄顺序即可。
    """
    _fields_ = [("temperature", ctypes.c_float),
                ("acc_sample", K4AFloat3),
                ("acc_timestamp_usec", ctypes.c_uint64),
                ("gyro_sample", K4AFloat3),
                ("gyro_timestamp_usec", ctypes.c_uint64)]


def main():
    print(f"k4a_imu_sample_t 大小 = {ctypes.sizeof(K4AImuSample)} 字节（C 里是 48）")

    # ★ read_gravity=False：这个脚本要的是【原始】读数，
    #   自动对齐会多读一次 IMU，也会把点云提前转正，就没法看原始状态了。
    src = AzureSource(verbose=True, read_gravity=False)
    k = src.k
    k.k4a_device_start_imu.restype = c_int
    k.k4a_device_start_imu.argtypes = [c_void_p]
    k.k4a_device_get_imu_sample.restype = c_int
    k.k4a_device_get_imu_sample.argtypes = [c_void_p,
                                            POINTER(K4AImuSample), c_int]
    k.k4a_device_stop_imu.restype = None
    k.k4a_device_stop_imu.argtypes = [c_void_p]

    rc = k.k4a_device_start_imu(src.h)
    if rc != 0:
        print(f"❌ 启动 IMU 失败 rc={rc}（这台设备的 IMU 可能不可用）")
        src.close()
        return 1

    accs, gyros = [], []
    try:
        print("采样 3 秒（相机保持不动）…")
        t0 = time.time()
        while time.time() - t0 < 3.0:
            s = K4AImuSample()
            if k.k4a_device_get_imu_sample(src.h, byref(s), 200) == 0:
                accs.append([s.acc_sample.v.x, s.acc_sample.v.y,
                             s.acc_sample.v.z])
                gyros.append([s.gyro_sample.v.x, s.gyro_sample.v.y,
                              s.gyro_sample.v.z])
    finally:
        k.k4a_device_stop_imu(src.h)
        src.close()

    if not accs:
        print("❌ 一个 IMU 样本都没拿到")
        return 1

    a = np.array(accs)
    g = np.array(gyros)
    m = a.mean(axis=0)
    n = np.linalg.norm(m)
    print()
    print("=" * 70)
    print(f" 加速度计样本 {len(a)} 个，抖动 ±{a.std(axis=0)}")
    print("=" * 70)
    print(f" 平均加速度 (ax, ay, az) = ({m[0]:+.3f}, {m[1]:+.3f}, {m[2]:+.3f}) m/s²")
    print(f" 模长 = {n:.2f} m/s²   （重力约 9.81；接近说明读数是重力）")
    if n > 1e-6:
        u = m / n
        print(f" 单位化 = ({u[0]:+.3f}, {u[1]:+.3f}, {u[2]:+.3f})")
    print(" ↑ 这是在【加速度计自己的坐标系】里 —— IMU 和深度相机不是同一个系，")
    print("   必须用标定里的外参转过去，否则会得出错误结论。")

    # ★★ 关键：把重力向量从【加速度计系】转到【深度相机系】
    if src.calib is not None:
        e = src.calib.extrinsics[K4A_CALIBRATION_TYPE_ACCEL][
            K4A_CALIBRATION_TYPE_DEPTH]
        R = np.array(e.rotation, np.float64).reshape(3, 3)
        t = np.array(e.translation, np.float64) / 1000.0
        print()
        print(f" 加速度计→深度 的外参 |t| = {np.linalg.norm(t)*1000:.1f} mm")
        print(f" R = \n{np.round(R, 4)}")
        g_depth = R @ m
        gd = g_depth / max(1e-9, np.linalg.norm(g_depth))
        print()
        print(f" ★★ 转到【深度相机系】后的重力向量 = "
              f"({g_depth[0]:+.3f}, {g_depth[1]:+.3f}, {g_depth[2]:+.3f}) m/s²")
        print(f"    单位化 = ({gd[0]:+.3f}, {gd[1]:+.3f}, {gd[2]:+.3f})")
    print()
    print(" ★ 相机【静止】时，加速度计读到的主要就是重力（指向地面）")
    print("   所以这个向量的方向 = 相机坐标系里【向下】的方向")
    print()
    # 用【转到深度系之后】的向量来判断
    gv = (R @ m) if src.calib is not None else m
    print()
    print(" ══ 判读（以深度相机系为准；加速度计静止时读的是支撑力，")
    print("     也就是指向【上】的方向）══")
    for name, val in (("X", gv[0]), ("Y", gv[1]), ("Z", gv[2])):
        tag = "  ← 竖直方向" if abs(val) > 0.8 * np.linalg.norm(gv) else ""
        print(f"    {name} = {val:+8.3f} m/s²{tag}")
    big = int(np.argmax(np.abs(gv)))
    axis = "XYZ"[big]
    sign = "+" if gv[big] > 0 else "−"
    print()
    print(f" ★ 重力主要落在 {sign}{axis} 轴上")
    if big == 1:
        if gv[1] > 0:
            print("   → +Y 指向【上】→ 深度系的 +Y 是【向上】的")
            print("     这【不是】标准光学系（标准是 +Y 向下），")
            print("     说明相机是【倒装】的：加 --rotate 180 把它转正。")
        else:
            print("   → −Y 指向【上】，也就是 +Y 指向【下】✅")
            print("     这就是标准光学系约定（+X 右 / +Y 下 / +Z 前），")
            print("     我们的代码是对的，Y 为正 = 目标在相机下方。")
    elif big == 2:
        print("   → 光轴（±Z）是【竖直】的！相机在俯视或仰视。")
        print("     这时 +Y 是【水平】方向，报告里的 Y 不代表'高度差'。")
        if gv[2] > 0:
            print("     +Z 向上 → 相机【朝上】看")
        else:
            print("     −Z 向下 → 相机【朝下】看（俯视桌面）")
    else:
        print(f"   → {axis} 轴是竖直的 → 相机是【侧装】的，")
        print("     +Y 不是竖直方向，报告里的 Y 不代表'高度差'。")
    print(f"\n 陀螺仪均值 = {np.round(g.mean(axis=0), 3)}（静止时应接近 0）")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
