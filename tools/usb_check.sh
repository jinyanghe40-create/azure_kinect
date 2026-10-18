#!/usr/bin/env bash
# RealSense USB 连接自检 —— 在【板子】上跑
#
# 为什么需要它：
#   D435i 必须跑在 USB 3.0 (5000M) 上。如果它挂在 USB 2.0 (480M) 总线上，
#   带宽不够，librealsense 会报：
#       RuntimeError ... Couldn't resolve requests
#   这个错看起来像"驱动/权限问题"，其实九成是**插错口了**（插到 2.0 口、
#   或者经过一个 USB 2.0 hub）。查一次 USB 树就知道了。
#
# 用法:
#   bash tools/usb_check.sh          # 在板子上
set -u

echo "════════════════════════════════════════════════════"
echo " RealSense USB 连接自检"
echo "════════════════════════════════════════════════════"

if ! command -v lsusb >/dev/null 2>&1; then
    echo "❌ 没装 lsusb（sudo apt install usbutils）"; exit 1
fi

# 1. 认到设备没有
if lsusb | grep -qi "8086:0b3a\|Intel.*RealSense"; then
    echo "✅ 认到 RealSense 设备"
else
    echo "❌ 没认到 RealSense（8086:0b3a）"
    echo "   → 拔插一次；换根 USB 3.0 线（很多线只有 2.0 的针）"
    exit 1
fi

# 2. 挂在哪条总线上、什么速率
echo
echo "--- USB 树 ---"
lsusb -t 2>/dev/null

# 找出 RealSense 所在的行，向上追溯是 480M 还是 5000M
SPEED=$(lsusb -t 2>/dev/null | grep -i "uvcvideo" | head -1 | grep -o "[0-9]*M" | head -1)
if [ -z "$SPEED" ]; then
    echo "⚠️ 解析不出速率，请人工看上面的树"
    exit 0
fi

echo
echo "════════════════════════════════════════════════════"
case "$SPEED" in
    5000M|10000M)
        echo "✅ 相机跑在 USB 3.0 ($SPEED) —— 正常"
        echo "   对带宽要求高的设置（640x480 深度 + 1280x720 彩色）都能开"
        ;;
    480M)
        echo "❌ 相机跑在 USB 2.0 ($SPEED) —— 这就是问题！"
        echo
        echo "   典型症状："
        echo "     RuntimeError: 深度 (640, 480) 和几个备用分辨率都打不开"
        echo "     最后一次错误: Couldn't resolve requests"
        echo
        echo "   怎么修（按顺序试）："
        echo "     ① 拔下来，直接插到 Jetson 的【蓝色 USB 3.0 口】"
        echo "        —— 别经过任何 hub"
        echo "     ② 换一根【真正的 USB 3.0 线】（很多线只有 2.0 的针，"
        echo "        外观一样但插上去就是 480M）"
        echo "     ③ 如果必须用 hub：要【带独立电源的 USB 3.0 hub】"
        echo "        （无源 hub 既缺电又只能跑 2.0）"
        echo "     ④ 改完拔插一次相机，再跑这个脚本确认变成 5000M"
        echo
        echo "   另外：USB 2.0 上 D435i 的最小深度会变差、丢点也更多，"
        echo "   所以就算能开起来，测出来的精度也不代表真实水平。"
        exit 1
        ;;
    *)
        echo "⚠️ 速率 $SPEED —— 不确定是否正常，请人工看上面的树"
        ;;
esac
echo "════════════════════════════════════════════════════"
