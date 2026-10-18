#!/usr/bin/env bash
# 一次性安装：把 /dev/ttyGS0（发给电控的串口）交给普通用户 nano，
# 并且把占着这个口的【登录程序 agetty】永久赶走。
#
# 为什么需要这一步（两个独立的问题）：
#
#   ① 权限：/dev/ttyGS0 默认是 crw--w---- root tty（0620），
#      而 nano 【不在 tty 组里】→ 一打开就 PermissionError。
#      ★ 这里不改 nano 的组成员：usermod 要重新登录才生效，
#        在"开机自启"的场景下等于没改（服务跑的时候组还没刷新）。
#      → 改成把【设备】划给 plugdev 组：nano 已经在 plugdev 里（实测），
#        而且本仓库的相机规则 k4a_udev_setup.sh 用的也是 plugdev，保持一致。
#      （不用 dialout：那是 /dev/ttyTHS2 真 UART 的组，nano 不在里面，
#        选它就得额外 usermod，没必要。）
#
#   ② 干扰：serial-getty@ttyGS0 是 L4T 默认开的【串口登录台】。
#      它抢不走我们发出去的字节，但会【往电控收到的数据流里掺登录提示符】，
#      而且回显（ECHO）开着时还会把电控发来的字节回显回去 ——
#      对端解析器会看到一堆 "nano login:" 垃圾。
#      ★ 只 stop/disable 不够：reboot 后 udev 会重新把它拉起来。
#        必须 mask（= /etc/systemd/system/serial-getty@ttyGS0.service → /dev/null），
#        这样连 generator/udev 主动要求启动它也起不来，且重启后依然有效。
#
# ★ 内核日志【不用管】：/proc/cmdline 里是 console=ttyS0（物理调试串口），
#   不是 console=ttyGS0。所以内核不会往这个口灌日志，不用改 extlinux.conf。
#   本脚本会把这个事实打出来给你确认。
#
# 用法（板子上，一次性）:
#   sudo bash ~/work/azure_kinect/ttygs_udev_setup.sh
#
# 装完自查:
#   ls -l /dev/ttyGS0                         期望 crw-rw---- root plugdev
#   systemctl is-enabled serial-getty@ttyGS0  期望 masked
#
# 撤销（想恢复成默认的串口登录台）:
#   sudo rm -f /etc/udev/rules.d/60-serial-ttygs0.rules
#   sudo systemctl unmask serial-getty@ttyGS0.service
#   sudo systemctl enable --now serial-getty@ttyGS0.service
#   sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=tty
set -e

if [ "$(id -u)" != "0" ]; then
    echo "❌ 需要 root：sudo bash $0"
    exit 1
fi

RULE=/etc/udev/rules.d/60-serial-ttygs0.rules
DEV=/dev/ttyGS0
GETTY=serial-getty@ttyGS0.service

echo "════════════════════════════════════════════════════"
echo " ttyGS0 串口放行 + 赶走 getty"
echo "════════════════════════════════════════════════════"

# ---------- 0. 先打印现状（出事时好对照）----------
echo "=== 现状 ==="
if [ -e "$DEV" ]; then
    echo "· $DEV        $(ls -l "$DEV")"
else
    echo "· $DEV        ❌ 不存在（gadget 没起来？先跑 k4a_udev_setup.sh 或查 lsusb）"
    exit 1
fi
echo "· nano 的组     $(id -nG nano)"
echo "· getty 启用?   $(systemctl is-enabled "$GETTY" 2>&1)"
echo "· getty 在跑?   $(systemctl is-active  "$GETTY" 2>&1)"
# ★ 方括号 [a]getty 是防"匹配到自己"——见 wait_ready.sh 里的详细说明
echo "· 谁占着这个口  $(pgrep -af '[a]getty.*ttyGS0' || echo '（没有）')"
echo "· 内核控制台    $(grep -o 'console=[^ ]*' /proc/cmdline | sort -u | tr '\n' ' ')"
if grep -q 'console=ttyGS0' /proc/cmdline; then
    echo "  ⚠️  /proc/cmdline 里有 console=ttyGS0 —— 内核日志会灌进这个口！"
    echo "      要去 /boot/extlinux/extlinux.conf 把 console=ttyGS0 删掉再重启。"
else
    echo "  ✅ 内核控制台不是 ttyGS0 → 内核日志不会掺进来，不用改 extlinux.conf"
fi

# ---------- 1. 权限：写自己的规则文件 ----------
# ★ 不能追加到 99-k4a.rules —— k4a_udev_setup.sh 用的是截断式 `cat >`，
#   以后重跑一次相机脚本，追加的内容就被静默吃掉了。
echo
echo "=== 写 $RULE ==="
cat > "$RULE" <<'EOF'
# USB gadget 串口（发给电控的 /dev/ttyGS0）划给 plugdev 组。
# nano 在 plugdev 里 → 不用改组成员、不用重新登录。
# 独立成文件：k4a_udev_setup.sh 写自己的 99-k4a.rules 时是截断式写入，
# 追加到那里会被覆盖掉。
KERNEL=="ttyGS0", SUBSYSTEM=="tty", MODE="0660", GROUP="plugdev"
EOF
echo "  ✅ 已写"

# systemd 237 的 udevadm 【没有 --sysname-match】，只能按子系统 trigger
udevadm control --reload-rules
udevadm trigger --subsystem-match=tty

# ---------- 2. 赶走 getty（mask，不只是 disable）----------
echo
echo "=== stop / disable / mask $GETTY ==="
# || true：getty 可能本来就没在跑，不要因此中断
systemctl stop    "$GETTY" || true
systemctl disable "$GETTY" || true
systemctl mask    "$GETTY" || true
echo "  ✅ 已 mask（reboot 后依然有效）"

# ---------- 3. 验证 ----------
echo
echo "=== 验证 ==="
for _ in $(seq 1 10); do
    [ "$(stat -c '%G' "$DEV" 2>/dev/null)" = "plugdev" ] && break
    sleep 0.5
done

FAIL=0
G=$(stat -c '%G' "$DEV")
M=$(stat -c '%a' "$DEV")
if [ "$G" = "plugdev" ]; then
    echo "  ✅ 组 = plugdev"
else
    echo "  ❌ 组 = $G（期望 plugdev）—— udev 规则没生效？"
    echo "     试: sudo udevadm trigger --subsystem-match=tty  然后再看 ls -l $DEV"
    FAIL=1
fi
case "$M" in 660|664|666) echo "  ✅ 权限 = $M";; *) echo "  ❌ 权限 = $M（期望 660）"; FAIL=1;; esac

# 以 nano 的身份真的开一次（这才是服务会遇到的真实情况）
if runuser -u nano -- env HOME=/home/nano python3 -c \
        "import serial;s=serial.Serial('$DEV',115200,timeout=0.1);s.close()" \
        >/dev/null 2>&1; then
    echo "  ✅ nano 能真的打开 $DEV（pyserial 实测）"
else
    echo "  ⚠️  nano 打不开 $DEV —— 手动确认一下："
    echo "      runuser -u nano -- env HOME=/home/nano python3 -c \\"
    echo "        \"import serial;print(serial.Serial('$DEV',115200))\""
    FAIL=1
fi

getty_state=$(systemctl is-enabled "$GETTY" 2>&1 || true)
if [ "$getty_state" = "masked" ]; then
    echo "  ✅ getty = masked"
else
    echo "  ❌ getty = $getty_state（期望 masked）"
    FAIL=1
fi

if pgrep -f '[a]getty.*ttyGS0' >/dev/null 2>&1; then
    echo "  ❌ 还有 agetty 占着这个口："
    pgrep -af '[a]getty.*ttyGS0' | sed 's/^/       /'
    echo "     （可能是 mask 之前就已经在跑的实例，reboot 后就没了）"
else
    echo "  ✅ 没有 agetty 占口"
fi

echo
echo "════════════════════════════════════════════════════"
if [ "$FAIL" = "0" ]; then
    echo "✅ 搞定。以后开机自动就是 crw-rw---- root plugdev，且没有登录台干扰。"
    echo
    echo "下一步：装自启服务"
    echo "    sudo bash ~/work/azure_kinect/install_vision_service.sh"
else
    echo "⚠️  有没通过的项，见上面的 ❌。"
    echo "    注意：mask 之前就在跑的 agetty 要等你 reboot 才会消失，"
    echo "    如果只有这一条没过，重启后复查即可。"
fi
echo
echo "撤销（恢复成默认的串口登录台）:"
echo "    sudo rm -f $RULE"
echo "    sudo systemctl unmask $GETTY && sudo systemctl enable --now $GETTY"
echo "    sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=tty"
echo "════════════════════════════════════════════════════"
