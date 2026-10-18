#!/usr/bin/env bash
# 一次性安装：让裸 X 服务器【开机自启】（Azure Kinect 深度引擎需要）
#
# 为什么需要开机自启：
#   Azure Kinect 的 libdepthengine.so.2.0 硬依赖 libX11 + libGLX，
#   没有 X 就报 "Depth engine ... error code: 204"。
#   但板子为了省电关掉了桌面（multi-user.target），开机没有 X。
#   手工每次起太麻烦，而且 Xorg 要 root —— 做成 systemd 服务最省事。
#
# ★ 这也是【最省电】的方案：
#   裸 X 服务器不带窗口管理器、不带 GNOME，功耗远低于桌面，
#   但 GLX 仍然是硬件加速的（用 nvidia_drv.so + libGLX_nvidia.so）。
#
# 用法（板子上，一次性）:
#   sudo bash ~/work/azure_kinect/install_x_service.sh
#
# 装完：
#   systemctl status k4a-xorg      看状态
#   systemctl restart k4a-xorg     重启 X
#   journalctl -u k4a-xorg -n 30   看日志
set -e

if [ "$(id -u)" != "0" ]; then
    echo "❌ 需要 root：sudo bash $0"
    exit 1
fi

DISP="${K4A_DISPLAY:-:1}"
VT="${K4A_VT:-7}"

# 1) 有没有接显示器 / 欺骗器？（没接的话 Xorg 可能起不来）
echo "=== 检查显示输出 ==="
# ★ 注意 set -e 的坑：如果 /sys/class/drm/*/status 匹配不到文件，
#   bash 会把通配符原样保留，[ -f "...*status" ] 返回假，
#   for 循环退出码就是 1 → 命令替换失败 → set -e 直接杀掉脚本。
#   所以这里显式 || true。
CONN=""
for _f in /sys/class/drm/*/status; do
    [ -f "$_f" ] || continue
    CONN="$CONN$(basename "$(dirname "$_f")")=$(cat "$_f") "
done || true
if [ -n "$CONN" ]; then
    echo "  $CONN"
else
    echo "  ℹ️  /sys/class/drm 里没有连接器（tegradrm 没加载，属正常）"
fi
echo "  ℹ️  HDMI 有没有插，看: dmesg | grep 'hdmi: plugged' | tail -1"
echo "  ℹ️  如果 Xorg 起不来，先确认 HDMI 上接了【显示器或欺骗器】"

# 2) 写服务
SVC=/etc/systemd/system/k4a-xorg.service
echo "=== 写 $SVC ==="
cat > "$SVC" <<EOF
[Unit]
Description=Bare Xorg for Azure Kinect depth engine (headless)
# 深度引擎要 GLX，所以 X 必须在相机程序之前就绪。
# ★★ 千万别写 After=multi-user.target：本单元自己 WantedBy=multi-user.target，
#   再 After 它就成了环。systemd 破环时会删掉 k4a-vision 的 start job
#   → 服务 enabled 却开机不启动（2026-10-18 重启实测踩到）。
After=systemd-udev-settle.service
Wants=systemd-udev-settle.service

[Service]
Type=simple
# 裸 X：不带窗口管理器、不带桌面 —— 省电，但 GLX 硬件加速
ExecStart=/usr/bin/Xorg $DISP -noreset -nolisten tcp vt$VT
Restart=always
RestartSec=3
# Xorg 需要 root
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable k4a-xorg.service || true
# ★ 这里用 || true：Xorg 起不来时 restart 会返回非零，
#   但我们要继续往下走，好在后面打印日志帮用户定位
systemctl restart k4a-xorg.service || true

echo
echo "=== 等它就绪 ==="
for _ in $(seq 1 20); do
    sleep 1
    if [ -S "/tmp/.X11-unix/X${DISP#:}" ]; then
        echo "✅ X 起来了：DISPLAY=$DISP"
        break
    fi
done

if [ -S "/tmp/.X11-unix/X${DISP#:}" ]; then
    echo
    echo "以后开机自动就有 X。跑主程序不用 sudo："
    echo "    bash ~/work/azure_kinect/start_azure.sh"
    echo
    echo "★ 顺便建议：修掉 /tmp 被整个清空的坑（否则日志还会神秘消失）"
    echo "    sudo bash ~/work/azure_kinect/fix_tmp_cleanup.sh"
else
    echo "❌ X 还是没起来。看日志："
    echo "    journalctl -u k4a-xorg -n 40 --no-pager"
    echo
    echo "常见原因："
    echo "  · HDMI 上没接显示器/欺骗器"
    echo "  · vt$VT 被别的占了 → K4A_VT=8 sudo -E bash \$0"
    echo "  · 想直接用桌面：sudo systemctl start gdm3 && export DISPLAY=:0"
    exit 1
fi
