#!/usr/bin/env python3
"""
相机简易查看器 —— 同时看彩色 + 深度

现在（深度不可用）: 只显示彩色（走 V4L2，不需要 k4a）
深度可用后:        自动加一行深度彩色映射

用法:
    python3 view_camera.py                 # 彩色 1280x720
    python3 view_camera.py --color-only    # 明确只要彩色
    python3 view_camera.py --depth         # 尝试彩色+深度（需要 k4a）
    python3 view_camera.py --device 0      # 换成 /dev/video0

按键（在窗口上按）:
    q / ESC   退出
    s         保存当前帧到 /tmp/camera_shots/
    d         切换深度彩色映射
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

SHOT_DIR = "/tmp/camera_shots"


def parse_args():
    p = argparse.ArgumentParser(description="看相机画面")
    p.add_argument("--device", type=int, default=2,
                   help="V4L2 设备号（Kinect 彩色=2，内置摄像头=0）")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--color-only", action="store_true", help="不尝试深度")
    p.add_argument("--no-resize", action="store_true",
                   help="不缩小显示（默认缩到 960 宽）")
    return p.parse_args()


def open_color(args):
    path = f"/dev/video{args.device}"
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, f"打不开 {path}"
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, f = cap.read()
    if not ok or f is None:
        cap.release()
        return None, f"{path} 能开但读不到帧"
    return cap, f"OK  {f.shape[1]}x{f.shape[0]}"


def depth_colormap(depth):
    """深度图 -> 彩色映射（近红远蓝），无效点涂黑。"""
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.percentile(depth[valid], 2), np.percentile(depth[valid], 98)
    t = np.clip((depth.astype(np.float32) - lo) / max(1.0, hi - lo), 0, 1)
    t8 = (t * 255).astype(np.uint8)
    col = cv2.applyColorMap(t8, cv2.COLORMAP_JET)
    col[~valid] = 0
    return col


def main():
    args = parse_args()
    os.makedirs(SHOT_DIR, exist_ok=True)

    cap, msg = open_color(args)
    if cap is None:
        print(f"[ERROR] {msg}")
        print("        检查: v4l2-ctl --list-devices")
        return 1
    print(f"[彩色] {msg}")

    use_depth = False
    if not args.color_only:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from pc_pipeline import read_ply   # noqa: F401  只是探一下模块在不在
            import ctypes
            for nm in ("libk4a.so.1.4", "libk4a.so.1.4.2", "libk4a.so"):
                try:
                    ctypes.CDLL(nm)
                    break
                except OSError:
                    continue
            else:
                raise OSError("libk4a 未安装")
            # 真取深度需要完整 k4a 流程；先用 k4a_smoke 的探测结果判断
            print("[深度] 检测到 libk4a，但取深度流需要相机可打开")
            print("       若深度不可用，本查看器自动只显示彩色")
            use_depth = _try_start_depth()
        except Exception as e:
            print(f"[深度] 不可用: {e}")

    print(f"\n按键:  q/ESC=退出   s=截图   d=切换深度映射")
    print(f"截图保存到: {SHOT_DIR}\n")

    show_depth = False
    frame_count = 0
    t0 = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] 空帧")
            time.sleep(0.05)
            continue

        frame_count += 1
        if frame_count % 15 == 0:
            fps = 15.0 / max(1e-6, time.time() - t0)
            t0 = time.time()

        H, W = frame.shape[:2]
        if use_depth and show_depth and _depth_frame is not None:
            dcol = depth_colormap(_depth_frame)
            dcol = cv2.resize(dcol, (W, int(W * dcol.shape[0] / dcol.shape[1])))
            canvas = np.vstack([frame, dcol])
        else:
            canvas = frame

        hud = canvas.copy()
        cv2.putText(hud, f"{fps:4.1f} FPS", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        tag = "COLOR" + ("+DEPTH" if (use_depth and show_depth) else "")
        cv2.putText(hud, tag, (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(hud, "q quit | s shot | d depth", (12, hud.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        if not args.no_resize and hud.shape[1] > 960:
            sc = 960.0 / hud.shape[1]
            hud = cv2.resize(hud, None, fx=sc, fy=sc)

        cv2.imshow("camera", hud)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            p = os.path.join(SHOT_DIR, f"shot_{ts}.png")
            cv2.imwrite(p, frame)
            print(f"[SHOT] {p}")
        elif key == ord("d"):
            show_depth = not show_depth
            print(f"[INFO] 深度显示: {'开' if show_depth else '关'}")

    cap.release()
    cv2.destroyAllWindows()
    print("退出")
    return 0


_depth_frame = None


def _try_start_depth():
    """尝试启动 k4a 深度流。失败返回 False（只显示彩色）。"""
    global _depth_frame
    try:
        import threading
        import ctypes
        from ctypes import (c_int, c_uint32, c_void_p, POINTER, byref)

        for nm in ("libk4a.so.1.4", "libk4a.so.1.4.2", "libk4a.so"):
            try:
                k = ctypes.CDLL(nm)
                break
            except OSError:
                continue
        else:
            return False

        class Cfg(ctypes.Structure):
            _fields_ = [("color_format", c_int), ("color_resolution", c_int),
                        ("depth_mode", c_int), ("camera_fps", c_int),
                        ("synchronized_images_only", c_int),
                        ("depth_delay_off_color_usec", ctypes.c_int32),
                        ("wired_sync_mode", c_int),
                        ("subordinate_delay_off_master_usec", c_uint32),
                        ("disable_streaming_indicator", c_int)]

        k.k4a_device_get_installed_count.restype = c_uint32
        k.k4a_device_open.restype = c_int
        k.k4a_device_open.argtypes = [c_uint32, POINTER(c_void_p)]
        k.k4a_device_start_cameras.restype = c_int
        k.k4a_device_start_cameras.argtypes = [c_void_p, POINTER(Cfg)]
        k.k4a_device_get_capture.restype = c_int
        k.k4a_device_get_capture.argtypes = [c_void_p, POINTER(c_void_p), c_int]
        for fn in (k.k4a_capture_get_depth_image,):
            fn.restype = c_void_p
            fn.argtypes = [c_void_p]
        for fn in (k.k4a_image_get_width_pixels, k.k4a_image_get_height_pixels):
            fn.restype = c_int
            fn.argtypes = [c_void_p]
        k.k4a_image_get_buffer.restype = POINTER(ctypes.c_uint16)
        k.k4a_image_get_buffer.argtypes = [c_void_p]
        k.k4a_image_release.restype = None
        k.k4a_image_release.argtypes = [c_void_p]
        k.k4a_capture_release.restype = None
        k.k4a_capture_release.argtypes = [c_void_p]

        dev = c_void_p()
        if k.k4a_device_open(0, byref(dev)) != 0 or not dev:
            print("[深度] 打不开设备（供电问题），只显示彩色")
            return False
        cfg = Cfg(color_format=3, color_resolution=1, depth_mode=2,
                  camera_fps=1, synchronized_images_only=0,
                  depth_delay_off_color_usec=0, wired_sync_mode=0,
                  subordinate_delay_off_master_usec=0,
                  disable_streaming_indicator=0)
        if k.k4a_device_start_cameras(dev, byref(cfg)) != 0:
            print("[深度] 启动失败，只显示彩色")
            return False
        print("[深度] 已启动，按 d 切换显示")

        def loop():
            global _depth_frame
            while True:
                cap = c_void_p()
                if k.k4a_device_get_capture(dev, byref(cap), 2000) != 0 or not cap:
                    continue
                try:
                    img = k.k4a_capture_get_depth_image(cap)
                    if img:
                        try:
                            w = k.k4a_image_get_width_pixels(img)
                            h = k.k4a_image_get_height_pixels(img)
                            buf = k.k4a_image_get_buffer(img)
                            arr = np.ctypeslib.as_array(buf, shape=(h * w,))
                            _depth_frame = arr.reshape(h, w).astype(np.uint16)
                        finally:
                            k.k4a_image_release(img)
                finally:
                    k.k4a_capture_release(cap)

        threading.Thread(target=loop, daemon=True).start()
        return True
    except Exception as e:
        print(f"[深度] 异常: {e}")
        return False


if __name__ == "__main__":
    raise SystemExit(main())
