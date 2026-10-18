#!/usr/bin/env python3
"""
k4a 冒烟测试 —— 验证 Azure Kinect 的彩色 + 深度是否真的能出数据。

只依赖 python3 标准库（ctypes），不需要编译，不需要 k4a headers。

用法:
    python3 k4a_smoke.py [帧数]        默认 30 帧

退出码:
    0 = 彩色和深度都出数据了
    1 = 找不到设备
    2 = 打开了设备但深度全空（深度引擎没工作）
    3 = 彩色全空
    4 = 库加载失败
"""
import ctypes
import sys
import time
from ctypes import c_char_p, c_int, c_size_t, c_uint32, c_uint64, c_void_p, POINTER, byref

# ---------- k4a 常量 ----------
K4A_WAIT_INFINITE = -1
K4A_WAIT_TIMEOUT = 0

K4A_IMAGE_FORMAT_COLOR_BGRA32 = 3

K4A_COLOR_RESOLUTION_720P = 1          # 1280x720
K4A_DEPTH_MODE_NFOV_UNBINNED = 2       # 640x576，深度质量最好

K4A_FRAMES_PER_SECOND_15 = 1
K4A_FRAMES_PER_SECOND_30 = 0

DEVICE_DEFAULT = 0


class K4ADeviceConfiguration(ctypes.Structure):
    _fields_ = [
        ("color_format", c_int),
        ("color_resolution", c_int),
        ("depth_mode", c_int),
        ("camera_fps", c_int),
        ("synchronized_images_only", c_int),
        ("depth_delay_off_color_usec", ctypes.c_int32),
        ("wired_sync_mode", c_int),
        ("subordinate_delay_off_master_usec", c_uint32),
        ("disable_streaming_indicator", c_int),
    ]


def load_lib():
    for name in ("libk4a.so.1.4", "libk4a.so.1.4.2", "libk4a.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError(
        "找不到 libk4a。请确认已把 libk4a.so.1.4.2 放到 /usr/lib/x86_64-linux-gnu/ 并执行了 sudo ldconfig"
    )


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    try:
        k4a = load_lib()
    except OSError as e:
        print(f"[FAIL] {e}")
        return 4

    k4a.k4a_device_get_installed_count.restype = c_uint32
    k4a.k4a_device_open.restype = c_int
    k4a.k4a_device_open.argtypes = [c_uint32, POINTER(c_void_p)]
    k4a.k4a_device_get_serialnum.restype = c_int
    k4a.k4a_device_get_serialnum.argtypes = [c_void_p, c_char_p, POINTER(c_size_t)]
    k4a.k4a_device_start_cameras.restype = c_int
    k4a.k4a_device_start_cameras.argtypes = [c_void_p, POINTER(K4ADeviceConfiguration)]
    k4a.k4a_device_get_capture.restype = c_int
    k4a.k4a_device_get_capture.argtypes = [c_void_p, POINTER(c_void_p), c_int]
    k4a.k4a_capture_release.restype = None
    k4a.k4a_capture_release.argtypes = [c_void_p]
    k4a.k4a_device_stop_cameras.restype = None
    k4a.k4a_device_stop_cameras.argtypes = [c_void_p]
    k4a.k4a_device_close.restype = None
    k4a.k4a_device_close.argtypes = [c_void_p]
    k4a.k4a_image_release.restype = None
    k4a.k4a_image_release.argtypes = [c_void_p]

    # ⚠️ 这几个返回指针的函数必须显式声明 restype = c_void_p，
    # 否则 ctypes 默认按 c_int 解释，64 位指针会被截断成负数 → 段错误
    for fn in (k4a.k4a_capture_get_color_image, k4a.k4a_capture_get_depth_image):
        fn.restype = c_void_p
        fn.argtypes = [c_void_p]
    for fn in (k4a.k4a_image_get_width_pixels, k4a.k4a_image_get_height_pixels):
        fn.restype = c_int
        fn.argtypes = [c_void_p]
    k4a.k4a_image_get_buffer.restype = POINTER(ctypes.c_uint8)
    k4a.k4a_image_get_buffer.argtypes = [c_void_p]

    n = k4a.k4a_device_get_installed_count()
    print(f"[INFO] 检测到 {n} 个 Azure Kinect 设备")
    if n == 0:
        print("[FAIL] 没找到设备。检查 lsusb | grep 045e")
        return 1

    dev = c_void_p()
    rc = k4a.k4a_device_open(DEVICE_DEFAULT, byref(dev))
    if rc != 0 or not dev:
        print(f"[FAIL] k4a_device_open 失败 rc={rc}（试试用 sudo 运行，或加 plugdev 组）")
        return 1

    try:
        buf = ctypes.create_string_buffer(256)
        need = c_size_t(len(buf))
        if k4a.k4a_device_get_serialnum(dev, buf, byref(need)) == 0:
            print(f"[INFO] 序列号: {buf.value.decode(errors='replace')}")

        cfg = K4ADeviceConfiguration(
            color_format=K4A_IMAGE_FORMAT_COLOR_BGRA32,
            color_resolution=K4A_COLOR_RESOLUTION_720P,
            depth_mode=K4A_DEPTH_MODE_NFOV_UNBINNED,
            camera_fps=K4A_FRAMES_PER_SECOND_15,
            synchronized_images_only=1,
            depth_delay_off_color_usec=0,
            wired_sync_mode=0,
            subordinate_delay_off_master_usec=0,
            disable_streaming_indicator=0,
        )
        rc = k4a.k4a_device_start_cameras(dev, byref(cfg))
        if rc != 0:
            print(f"[FAIL] start_cameras 失败 rc={rc}（USB 速率不够时会这样）")
            return 1
        print("[INFO] 相机已启动，采集中...")

        depth_ok = 0
        color_ok = 0
        depth_wh = color_wh = None
        t0 = time.time()

        for i in range(n_frames):
            cap = c_void_p()
            rc = k4a.k4a_device_get_capture(dev, byref(cap), 5000)
            if rc != 0 or not cap:
                print(f"[WARN] 第 {i} 帧取帧失败 rc={rc}")
                continue
            try:
                dimg = k4a.k4a_capture_get_depth_image(cap)
                cimg = k4a.k4a_capture_get_color_image(cap)
                try:
                    if dimg:
                        w = k4a.k4a_image_get_width_pixels(dimg)
                        h = k4a.k4a_image_get_height_pixels(dimg)
                        depth_wh = (w, h)
                        p = k4a.k4a_image_get_buffer(dimg)
                        npx = w * h
                        nz = 0
                        if p:
                            for k in range(npx):
                                if p[k * 2] or p[k * 2 + 1]:
                                    nz += 1
                                    if nz > 500:
                                        break
                        if nz > 500:
                            depth_ok += 1
                    if cimg:
                        color_wh = (k4a.k4a_image_get_width_pixels(cimg),
                                    k4a.k4a_image_get_height_pixels(cimg))
                        color_ok += 1
                finally:
                    if dimg:
                        k4a.k4a_image_release(dimg)
                    if cimg:
                        k4a.k4a_image_release(cimg)
            finally:
                k4a.k4a_capture_release(cap)

        dt = time.time() - t0
        print(f"[INFO] 深度分辨率: {depth_wh}   彩色分辨率: {color_wh}")
        print(f"[INFO] 耗时 {dt:.1f}s -> 约 {n_frames / dt:.1f} FPS（含首次启动开销）")
        print(f"[INFO] 深度有效帧 {depth_ok}/{n_frames}   彩色有效帧 {color_ok}/{n_frames}")

        if color_ok == 0:
            print("[FAIL] 彩色全空")
            return 3
        if depth_ok == 0:
            print("[FAIL] 深度全空 —— 深度引擎没工作。用 LD_DEBUG=libs 排查，或确认 libdepthengine.so.2.0 已就位")
            return 2

        print("[PASS] 彩色 + 深度都正常，可以进入下一步（采数据 / 点云）")
        return 0

    finally:
        try:
            k4a.k4a_device_stop_cameras(dev)
        except Exception:
            pass
        k4a.k4a_device_close(dev)


if __name__ == "__main__":
    sys.exit(main())
