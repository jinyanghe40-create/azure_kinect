#!/usr/bin/env python3
"""
RealSense 分层测试 —— 精确定位是哪一路流打不开

用法:  python3 test_streams.py
"""
import sys

import pyrealsense2 as rs

print("librealsense 版本:", rs.__version__)
print()

ctx = rs.context()
devs = ctx.query_devices()
if len(devs) == 0:
    print("❌ 没找到设备")
    sys.exit(1)
d = devs[0]
print("设备:", d.get_info(rs.camera_info.name),
      "固件:", d.get_info(rs.camera_info.firmware_version),
      "USB:", d.get_info(rs.camera_info.usb_type_descriptor))
print()

# ---- ① 列出所有传感器 ----
print("=== ① 传感器列表 ===")
for s in d.sensors:
    print(f"  {s.get_info(rs.camera_info.name)}")

# ---- ② 逐个测：深度+彩色 ----
print()
print("=== ② 深度 + 彩色 ===")
try:
    p = rs.pipeline()
    c = rs.config()
    c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    c.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    p.start(c)
    for _ in range(5):
        f = p.wait_for_frames()
    df, cf = f.get_depth_frame(), f.get_color_frame()
    print(f"  ✅ 成功！深度 {df.get_width()}x{df.get_height()}  "
          f"彩色 {cf.get_width()}x{cf.get_height()}")
    dm = rs.align(rs.stream.color)
    fa = dm.process(f)
    di = fa.get_depth_frame().profile.as_video_stream_profile().get_intrinsics()
    print(f"  对齐后内参: fx={di.fx:.1f} fy={di.fy:.1f} "
          f"cx={di.ppx:.1f} cy={di.ppy:.1f}")
    p.stop()
except Exception as e:
    print(f"  ❌ 失败: {type(e).__name__}: {e}")

# ---- ③ 单独测 IMU（Motion Module）----
print()
print("=== ③ IMU（Motion Module）===")
for name, st in (("加速度计 Accel", rs.stream.accel),
                 ("陀螺仪 Gyro", rs.stream.gyro)):
    try:
        p = rs.pipeline()
        c = rs.config()
        c.enable_stream(st)
        p.start(c)
        for _ in range(5):
            f = p.wait_for_frames()
        m = f.first_motion_frame()
        print(f"  ✅ {name}: {m.get_motion_data()}")
        p.stop()
    except Exception as e:
        print(f"  ❌ {name} 失败: {type(e).__name__}: {e}")
