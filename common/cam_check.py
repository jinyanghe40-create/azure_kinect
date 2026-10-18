#!/usr/bin/env python3
"""
摄像头体检工具 —— 打不开 /dev/videoN 时先跑这个
（电脑和 Jetson 板子都能跑，只用 opencv + numpy，无其它依赖）

用法:
    python3 cam_check.py            # 体检所有 /dev/video*
    python3 cam_check.py 0          # 只体检 /dev/video0
    python3 cam_check.py --grab 0   # 体检并保存一帧到 cam0.jpg

它按"从底层到上层"的顺序查 5 件事：
    1. 设备节点在不在            (/dev/video*)
    2. 权限够不够                (你在不在 video 组)
    3. 内核认成了什么            (/sys/class/video4linux/*/name)
    4. USB 链路速度对不对        (5000M=USB3.0  480M=USB2.0 ← Kinect 深度必需 5000M)
    5. OpenCV 能不能真读到一帧   (有些节点是"元数据节点"，打不开也不出图)
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] 没装 opencv-python：pip3 install opencv-python")
    sys.exit(1)


# ---------------------------------------------------------------------
def usb_speed_of(node):
    """从 /sys/class/video4linux/videoN 往上找 USB 设备的 speed 文件"""
    p = os.path.realpath(f"/sys/class/video4linux/{os.path.basename(node)}/device")
    for _ in range(8):                       # 往上爬最多 8 层
        f = os.path.join(p, "speed")
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    return fh.read().strip()
            except OSError:
                return "?"
        p = os.path.dirname(p)
        if p in ("/", ""):
            break
    return "-"                               # 不是 USB 设备（如 CSI/MIPI 摄像头）


def sysfs_name(node):
    f = f"/sys/class/video4linux/{os.path.basename(node)}/name"
    try:
        with open(f) as fh:
            return fh.read().strip()
    except OSError:
        return "?"


def probe_capture(node):
    """用 OpenCV 真开一次，返回 (能否打开, 能否读帧, 尺寸)"""
    cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return False, False, None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, fr = False, None
    for _ in range(10):                      # 摄像头起流慢，多读几次
        ok, fr = cap.read()
        if ok and fr is not None:
            break
    cap.release()
    return True, bool(ok), (None if fr is None else fr.shape)


# ---------------------------------------------------------------------
def main():
    only = None
    grab = None
    argv = sys.argv[1:]
    if "--grab" in argv:
        i = argv.index("--grab")
        grab = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if argv:
        only = int(argv[0])

    print("=" * 62)
    print("① 用户 / 权限")
    print("=" * 62)
    who = subprocess.run(["id"], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
    print("  ", who)
    groups = subprocess.run(["id", "-Gn"], stdout=subprocess.PIPE,
                            universal_newlines=True).stdout.split()
    in_video = "video" in groups
    print(f"   所属组: {' '.join(groups)}")
    print(f"   在 video 组里? {'✅ 是' if in_video else '❌ 否  ← 摄像头默认只给 video 组用'}")
    if not in_video:
        print("      修: sudo usermod -aG video $USER   然后注销重登（或 reboot）")
        print("      （桌面登录时 systemd-logind 会给当前会话发临时 ACL，")
        print("        所以有时不报错；但 SSH / 开机自启 里就没这个优待了）")

    print()
    print("=" * 62)
    print("② 设备节点 /dev/video*")
    print("=" * 62)
    nodes = sorted(glob.glob("/dev/video*"),
                   key=lambda s: int(re.sub(r"\D", "", s) or 0))
    if only is not None:
        nodes = [n for n in nodes if n == f"/dev/video{only}"]
    if not nodes:
        print("   ❌ 一个都没有 → 不是软件问题，是 USB 层面没认到设备")
        print("      下一步看 ③ lsusb 和 ④ dmesg，并检查摄像头指示灯")
    for n in nodes:
        st = os.stat(n)
        r = os.access(n, os.R_OK)
        w = os.access(n, os.W_OK)
        print(f"   {n}  mode={oct(st.st_mode)[-3:]}  uid={st.st_uid} gid={st.st_gid}"
              f"  可读={'✅' if r else '❌'} 可写={'✅' if w else '❌'}")
        print(f"        内核名字: {sysfs_name(n)}")
        print(f"        USB 速度: {usb_speed_of(n)}")

    print()
    print("=" * 62)
    print("③ lsusb —— 设备到底枚举出来没有")
    print("=" * 62)
    try:
        out = subprocess.run(["lsusb"], stdout=subprocess.PIPE, universal_newlines=True).stdout
        print("".join("   " + l + "\n" for l in out.strip().splitlines()))
        if "045e" in out:
            print("   ✅ 看到微软(045e)设备 → USB 认到了")
        else:
            print("   ❌ 没有 045e 微软设备 → Kinect 根本没枚举")
            print("      查供电(灯色)/换 USB3.0 蓝口/换线")
    except FileNotFoundError:
        print("   跳过（没装 usbutils: sudo apt-get install -y usbutils）")

    print()
    print("=" * 62)
    print("④ lsusb -t —— 插在 USB3.0 还是 USB2.0 上")
    print("=" * 62)
    try:
        out = subprocess.run(["lsusb", "-t"], stdout=subprocess.PIPE, universal_newlines=True).stdout
        print("".join("   " + l + "\n" for l in out.strip().splitlines()))
        print("   提示: 5000M=USB3.0(蓝口)  480M=USB2.0(黑口，Kinect 深度用不了)")
    except FileNotFoundError:
        print("   跳过")

    print()
    print("=" * 62)
    print("⑤ OpenCV 实际能不能出图")
    print("=" * 62)
    good = []
    for n in nodes:
        can_open, can_read, shape = probe_capture(n)
        if can_open and can_read:
            print(f"   {n}  ✅ 打开成功，读到帧 {shape}   ← 可以用这个")
            good.append(n)
            if grab is not None and n == f"/dev/video{grab}":
                cap = cv2.VideoCapture(n, cv2.CAP_V4L2)
                ok, fr = cap.read()
                cap.release()
                if ok:
                    fn = f"cam{grab}.jpg"
                    cv2.imwrite(fn, fr)
                    print(f"        已存一帧 → {fn}")
        elif can_open:
            print(f"   {n}  ⚠️  能打开但读不到帧 → 多半是'元数据节点'或被别的程序占用")
        else:
            print(f"   {n}  ❌ 打不开 → 权限不足 / 被占用 / 不支持 V4L2 采集")

    print()
    print("=" * 62)
    print("结论")
    print("=" * 62)
    if good:
        print(f"   可用摄像头: {', '.join(good)}")
        print(f"   跑推理:  python3 trt_infer.py --device {good[0].replace('/dev/video', '')}")
    else:
        print("   没有可用摄像头。按上面 ①→⑤ 第一个 ❌ 的地方往下查。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
