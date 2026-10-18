#!/bin/bash
# =====================================================================
#  显示器不亮时依次尝试救回来 —— 在【板子上】用 sudo 跑
#
#  用法:  sudo bash ~/work/fix_display.sh
#
#  为什么会有这个问题：
#     拔掉 HDMI 时，内核会执行
#         tegradc.0: blank - powerdown      ← 关掉显示输出
#         Extcon AUX1(HDMI) disable         ← 停用热插拔检测
#     之后插回去，如果 HPD（热插拔检测）中断没触发，驱动就不会重新点亮屏幕。
#     Jetson 上这个挺常见。
#
#  ★ 顺序是从"最温和"到"最彻底"，能救回来就停。
# =====================================================================
set -u
if [ "$(id -u)" != "0" ]; then
    echo "❌ 要用 sudo 跑:  sudo bash $0"
    exit 1
fi

hr() { echo "────────────────────────────────────────"; }

hr; echo " 当前状态"; hr
echo "  帧缓冲:"
for f in /sys/class/graphics/fb0 /sys/class/graphics/fb1; do
    [ -d "$f" ] && echo "    $(basename $f): name=$(cat $f/name 2>/dev/null) blank=$(cat $f/blank 2>/dev/null)  (0=亮 4=灭)"
done
echo "  最近的显示相关内核消息:"
dmesg 2>/dev/null | grep -i -e hdmi -e tegradc | tail -4 | sed 's/^/    /'

hr; echo " ① 强制点亮帧缓冲（最温和，先试这个）"; hr
for f in /sys/class/graphics/fb0 /sys/class/graphics/fb1; do
    if [ -w "$f/blank" ]; then
        echo 0 > "$f/blank" 2>/dev/null && echo "  ✅ $(basename $f) 已 unblank"
    fi
done
sleep 3
echo "  现在 tail 一下内核看有没有反应:"
dmesg 2>/dev/null | grep -i -e hdmi -e tegradc | tail -2 | sed 's/^/    /'
echo
echo -n "  屏幕亮了吗？(亮了就按 Ctrl+C 停下) "; read -t 8 -r ans || ans=""
if [ "$ans" = "亮" ] || [ "$ans" = "y" ] || [ "$ans" = "yes" ]; then
    echo " ✅ 搞定！"; exit 0
fi

hr; echo " ② 重启显示管理器（X 重新初始化，会重新探测显示器）"; hr
systemctl restart gdm3
echo "  已重启 gdm3，等 8 秒…"
sleep 8
dmesg 2>/dev/null | grep -i -e hdmi -e tegradc | tail -3 | sed 's/^/    /'
echo
echo -n "  屏幕亮了吗？(亮了就按 Ctrl+C 停下) "; read -t 8 -r ans || ans=""
if [ "$ans" = "亮" ] || [ "$ans" = "y" ]; then
    echo " ✅ 搞定！"; exit 0
fi

hr; echo " ③ 还是不行"; hr
cat <<'EOF'
  接下来只能重启了。重启前务必做这两件事（减少启动电流）：
     ① 拔掉 RealSense（3.5W）和键盘鼠标
     ② 确认网线和电源插紧
     ③ 确认【显示器是开着的、HDMI 插紧的】← 重启时插好，开机就会识别
  然后:
     sudo reboot
EOF
