#!/usr/bin/env python3
"""UVC 相机快速查看 —— 打开、抓帧、存图（也能实时窗口看）。

为什么用这个而不是直接 cv2.imshow：
    USB 相机的格式支持五花八门（MJPG / YUYV，各种分辨率）。
    直接 open 经常黑屏或者报错，得先试一圈才知道这台支持什么。
    这个脚本会：
      ① 枚举所有 /dev/video*
      ② 逐个尝试常用格式和分辨率
      ③ 抓到第一帧就告诉你"这个组合行"
      ④ 存成 jpg，再决定要不要开实时窗口

用法:
    python3 tools/cam_view.py --list                 # 列出所有相机
    python3 tools/cam_view.py --dev 2                # 打开 /dev/video2，存一帧
    python3 tools/cam_view.py --dev 2 --show         # 开窗口实时看（需 DISPLAY）
    python3 tools/cam_view.py --dev 2 --probe        # 只探测支持哪些格式/分辨率
    python3 tools/cam_view.py --dev 2 --snap a.jpg   # 存到指定文件
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    print("❌ 没装 opencv：pip3 install opencv-python")
    sys.exit(1)


# 常用组合：先试 MJPG（带宽小、帧率高），再试 YUYV；从高分辨率往低试
COMBOS = [
    ("MJPG", 1920, 1080), ("MJPG", 1280, 720), ("MJPG", 640, 480),
    ("YUYV", 1280, 720), ("YUYV", 640, 480),
    ("", 1920, 1080), ("", 1280, 720), ("", 640, 480),
]


def list_cams():
    nodes = sorted(glob.glob("/dev/video*"),
                   key=lambda s: int(re.sub(r"\D", "", s) or 0))
    if not nodes:
        print("没有 /dev/video* 设备")
        return
    print(f"找到 {len(nodes)} 个视频节点：")
    # 尽量用 v4l2-ctl 拿到名字
    names = {}
    try:
        import subprocess
        out = subprocess.run(["v4l2-ctl", "--list-devices"],
                             capture_output=True, text=True).stdout
        cur = None
        for line in out.splitlines():
            if line and not line.startswith("\t"):
                cur = line.strip()
            elif line.startswith("\t") and cur:
                names[line.strip()] = cur
    except Exception:
        pass
    for n in nodes:
        print(f"  {n}   {names.get(n, '')}")


def probe(dev):
    """试一圈，打印哪些组合能出图"""
    print(f"探测 {dev} …")
    ok = []
    for fmt, w, h in COMBOS:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            print(f"  ❌ 打不开 {dev}")
            return []
        if fmt == "MJPG":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        elif fmt == "YUYV":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        got = None
        for _ in range(8):
            r, f = cap.read()
            if r and f is not None:
                got = f
                break
            time.sleep(0.05)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        tag = f"{fmt or 'default':7s} {w}x{h}"
        if got is not None:
            print(f"  ✅ {tag}  → 实际 {got.shape[1]}x{got.shape[0]}  "
                  f"{fps:.0f}fps")
            ok.append((fmt, w, h, got))
        else:
            print(f"  —  {tag}  → 读不到帧（请求 {w}x{h}，实际 {aw}x{ah}）")
    return ok


def open_best(dev, want_w=0, want_h=0):
    """找到第一个能出图的组合"""
    for fmt, w, h in COMBOS:
        if want_w and (w != want_w or h != want_h):
            continue
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        if fmt == "MJPG":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        elif fmt == "YUYV":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(10):
            r, f = cap.read()
            if r and f is not None:
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"✅ 用 {fmt or 'default'} {aw}x{ah} 打开成功")
                return cap, f
            time.sleep(0.05)
        cap.release()
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="2", help="设备号（/dev/video2 → 2）")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--probe", action="store_true", help="探测支持的格式/分辨率")
    ap.add_argument("--snap", default=None, help="存 jpg 的路径")
    ap.add_argument("--show", action="store_true", help="开实时窗口")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    a = ap.parse_args()

    if a.list:
        list_cams()
        return 0

    dev = a.dev if a.dev.startswith("/dev/") else f"/dev/video{a.dev}"
    if not os.path.exists(dev):
        print(f"❌ {dev} 不存在")
        list_cams()
        return 1

    if a.probe:
        probe(dev)
        return 0

    cap, frame = open_best(dev, a.width, a.height)
    if cap is None:
        print(f"❌ {dev} 打不开/读不到帧")
        print("   试试 --probe 看支持哪些组合，或者换设备号（--list）")
        return 1

    out = a.snap or "/tmp/cam_snap.jpg"
    cv2.imwrite(out, frame)
    print(f"✅ 第一帧已存 {out}   {frame.shape}")

    if a.show:
        print("实时窗口已开（按 q 或 ESC 退出）")
        n = 0
        t0 = time.time()
        try:
            while True:
                r, f = cap.read()
                if not r:
                    break
                n += 1
                if n % 15 == 0:
                    cv2.putText(f, f"{n/(time.time()-t0):.1f} FPS",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)
                cv2.imshow(f"Camera {dev}", f)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()
        print(f"共 {n} 帧")
    else:
        print("   （加 --show 可以开实时窗口看）")

    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
