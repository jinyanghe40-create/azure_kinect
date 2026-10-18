#!/usr/bin/env python3
"""Azure Kinect 体检：不依赖 pyk4a，直接用 ctypes 调 libk4a。

为什么做成 ctypes：
    板子上只装了 libk4a（C 库），没有 pyk4a / k4a 的 Python 绑定，
    也没有 k4aviewer。用 ctypes 直接调最省事 —— 不用再装东西。

用法（板子上）:
    python3 ~/work/azure_kinect/k4a_probe.py

它会依次检查：
    ① 认到几个设备
    ② 能不能 open（权限够不够）
    ③ 序列号
    ④ 能不能真的开起深度流（这一步才证明相机是好的）
"""
import ctypes
import sys
import time

LIB_CANDIDATES = ["libk4a.so.1.4", "libk4a.so.1", "libk4a.so"]


def load():
    for name in LIB_CANDIDATES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    print("❌ 找不到 libk4a。装了 SDK 吗？")
    sys.exit(1)


class DepthMode:
    OFF = 0
    NFOV_2X2BINNED = 1
    NFOV_UNBINNED = 2
    WFOV_2X2BINNED = 3
    WFOV_UNBINNED = 4
    PASSIVE_IR = 5


class ColorRes:
    OFF = 0
    RES_720P = 1
    RES_1080P = 2
    RES_1440P = 3
    RES_1536P = 4
    RES_2160P = 5
    RES_3072P = 6


class Fps:
    FPS_5 = 0
    FPS_15 = 1
    FPS_30 = 2


def main():
    k = load()
    print("=" * 60)
    print(" Azure Kinect 体检")
    print("=" * 60)

    n = k.k4a_device_get_installed_count()
    print(f"① 认到的设备数: {n}")
    if n == 0:
        print("   ❌ 一个都没有 —— 检查 USB 线（要 USB3.0 线）和供电")
        print("      板子上看: lsusb | grep 045e")
        return 1

    h = ctypes.c_void_p()
    r = k.k4a_device_open(0, ctypes.byref(h))
    print(f"② k4a_device_open -> {r}", "✅ 成功" if r == 0 else "❌ 失败")
    if r != 0:
        print()
        print("   失败的常见原因：")
        print("     a) 缺 udev 规则 → 先跑 sudo bash ~/work/azure_kinect/k4a_udev_setup.sh")
        print("     b) 被别的进程占用 → sudo lsof /dev/bus/usb/002/* | head")
        print("     c) 供电不足 → Kinect 需要 12V 独立电源；")
        print("        光靠 Jetson 的 USB 口带不动（尤其板子还是 micro-USB 供电时）")
        return 1

    buf = ctypes.create_string_buffer(256)
    sz = ctypes.c_size_t(256)
    r = k.k4a_device_get_serialnum(h, buf, ctypes.byref(sz))
    print(f"③ 序列号: {buf.value.decode() if r == 0 else f'读取失败({r})'}")

    # ④ 试着真开一路深度流 —— 这一步才证明相机是好的
    cfg = (ctypes.c_uint32 * 16)()      # k4a_device_configuration_t 是 16 个 uint32
    # 布局: color_format, color_resolution, depth_mode, camera_fps,
    #       synchronized_images_only, depth_delay_off_color_usec,
    #       wired_sync_mode, subordinate_delay_off_master_usec,
    #       disable_streaming_indicator, (padding)
    cfg[0] = 0                          # color_format = MJPG
    cfg[1] = ColorRes.RES_720P
    cfg[2] = DepthMode.NFOV_UNBINNED    # 640x576 深度
    cfg[3] = Fps.FPS_15
    cfg[4] = 0                          # synchronized_images_only
    cfg[5] = 0
    cfg[6] = 0
    cfg[7] = 0
    cfg[8] = 0

    r = k.k4a_device_start_cameras(h, ctypes.byref(cfg))
    print(f"④ k4a_device_start_cameras -> {r}",
          "✅ 深度流开起来了！" if r == 0 else "❌ 开不了")
    if r == 0:
        cap = ctypes.c_void_p()
        got = None
        for _ in range(50):             # 最多等 5 秒
            r2 = k.k4a_device_get_capture(h, ctypes.byref(cap), 100)
            if r2 == 0:
                got = cap
                break
        if got:
            print("   ✅ 拿到了第一帧 capture —— 相机完全正常")
            print("      注: 3.x 的 SDK 里 k4a_capture_t 是句柄，不用手动释放")
        else:
            print("   ⚠️ 流开了但拿不到帧 —— 多半是供电/带宽问题")
        k.k4a_device_stop_cameras(h)
    else:
        print()
        print("   开不了流的原因：")
        print("     · 供电不足（最常见）—— Kinect 深度引擎要 12V 独立电源")
        print("     · USB 不是 3.0 / 线材不行")

    k.k4a_device_close(h)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
