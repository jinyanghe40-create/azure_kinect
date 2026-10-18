#!/usr/bin/env bash
# Azure Kinect DK 一次性配置（在【板子】上跑，需要 sudo）
#
# 为什么需要：
#   Azure Kinect 的 USB 设备节点默认是 root:root 0664，
#   普通用户只有读权限，libusb 抢不到设备，SDK 会报：
#       libusb device(s) are all unavailable.
#       Is the device being used by another application?
#   这个报错很坑 —— 它会让你以为是"被别的程序占用了"或"供电不足"，
#   其实九成是缺 udev 规则。
#
# 用法（板子上）:
#   sudo bash ~/work/azure_kinect/k4a_udev_setup.sh
set -e

if [ "$(id -u)" != "0" ]; then
    echo "❌ 需要 root：sudo bash $0"
    exit 1
fi

RULES=/etc/udev/rules.d/99-k4a.rules
echo "写入 $RULES"
cat > "$RULES" <<'EOF'
# Azure Kinect DK —— 让普通用户也能打开设备
# 官方 99-k4a.rules，五个 USB 功能各一条
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097a", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097b", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097c", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097d", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097e", MODE="0666", GROUP="plugdev"
# 深度引擎的 misc 设备
KERNEL=="k4a*", MODE="0666"
EOF

echo "重新加载 udev 规则"
udevadm control --reload-rules
udevadm trigger --subsystem-match=usb

echo
echo "✅ 装好了。请【拔插一次 Kinect 的 USB】，然后跑："
echo "     python3 ~/work/azure_kinect/k4a_probe.py"
echo
echo "如果还报 unavailable，那就是真的被别的进程占了，或者供电不足："
echo "     sudo lsof /dev/bus/usb/002/* 2>/dev/null | head"
