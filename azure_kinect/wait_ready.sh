#!/usr/bin/env bash
# k4a-vision 服务的【开机门禁】（systemd 的 ExecStartPre）。
#
# 干什么（按顺序）：
#   0. 挡住重复启动 —— 已经有一个 run_azure.py 在跑就别再起第二个
#   1. 等 X 服务器 —— 深度引擎硬依赖 GLX，没 X 会报 "error code: 204"
#   2. 等深度相机 —— 特别是 045e:097c 这个【深度】接口
#   3. 稳定 2 秒 —— 刚枚举出来的设备立刻开会失败
#
# 为什么要等，而不是让 python 自己去报错：
#   开机时 X 和相机都是【几秒后才就绪】的，服务如果抢跑就会 crash。
#   而本服务是 Restart=no（崩了不重启），抢跑一次就等于这次开机白开。
#   所以宁可在这里有界地等，也不要让主程序去撞。
#
# 用法（给 systemd 用的，也可以手动跑看看会不会过）:
#   bash ~/work/azure_kinect/wait_ready.sh 120
#
# ★ 故意【不用 set -e】：本仓库文档里记过这个坑 ——
#   set -e 配上 $( ) + 通配符会【静默退出、什么都不打印】，
#   看起来像卡死。这里只要 set -u（查未定义变量）就够。
set -u

TIMEOUT="${1:-120}"
case "$TIMEOUT" in
    ''|*[!0-9]*)
        echo "[门禁] 用法: $0 [超时秒]，默认 120"
        exit 2
        ;;
esac
if [ "$TIMEOUT" -lt 1 ]; then
    echo "[门禁] 超时秒必须 ≥1"
    exit 2
fi

DISP="${DISPLAY:-:1}"
DEPTH_ID="045e:097c"     # ★ 深度相机 —— 决定了流水线能不能跑
COLOR_ID="045e:097d"     # 彩色相机（--color-only 时够用，但我们要深度）

DEADLINE=$(( $(date +%s) + TIMEOUT ))
remaining() { echo $(( DEADLINE - $(date +%s) )); }

echo "[门禁] 开始（超时 ${TIMEOUT}s，DISPLAY=$DISP，要深度相机 $DEPTH_ID）"

# =====================================================================
# 0. 挡住重复启动
# =====================================================================
# ★★ 这里的方括号 [r]un_azure 是【必须的】，不是风格问题：
#   如果写成 pgrep -f "run_azure.py"，而这个字符串又出现在
#   `sh -c '...'` 或 exec 的命令行里，那么 pgrep 会【匹配到它自己所在的
#   那条命令行】→ 永远返回 0 → 服务永远起不来，且报"已经在跑"。
#   实测踩过：ssh nano 'pgrep -af "agetty.*ttyGS"'
#   把【ssh 远端那个 bash -c 自己】(pid 10922) 也打了出来。
#   （方括号让模式本身不再字面出现在命令行里 → 匹配不到自己。）
if pgrep -f '[r]un_azure\.py' >/dev/null 2>&1; then
    echo "[门禁] ❌ 已经有 run_azure.py 在跑，拒绝起第二个："
    pgrep -af '[r]un_azure\.py' | sed 's/^/         /'
    echo "[门禁]    先停掉它再启服务：  pkill -f run_azure.py"
    echo "[门禁]    （服务是 Restart=no，所以两个实例会抢相机，必须避免）"
    exit 1
fi

# =====================================================================
# 1. 等 X 服务器
# =====================================================================
x_socket_ok() { [ -S "/tmp/.X11-unix/X${DISP#:}" ]; }

if x_socket_ok; then
    echo "[门禁] ✅ X 已经在（$DISP）"
else
    echo "[门禁] 等 $DISP 的 X socket …"
fi

_n=0
while ! x_socket_ok; do
    if [ "$(remaining)" -le 0 ]; then
        echo "[门禁] ❌ 等 X 超时（${TIMEOUT}s）。$DISP 一直没起来。"
        echo "       查： systemctl status k4a-xorg"
        echo "           journalctl -u k4a-xorg -n 40 --no-pager"
        echo "       常见原因：HDMI 上没接显示器/欺骗器；vt 被别人占了。"
        exit 1
    fi
    _n=$((_n + 1))
    [ $((_n % 10)) -eq 0 ] && echo "[门禁]   …还在等 X（已 ${_n}s，剩 $(remaining)s）"
    sleep 1
done
[ "$_n" -gt 0 ] && echo "[门禁] ✅ X 就绪（等了 ${_n}s）"

# =====================================================================
# 2. 等深度相机
# =====================================================================
cam_ok() { lsusb -d "$DEPTH_ID" >/dev/null 2>&1; }

if cam_ok; then
    echo "[门禁] ✅ 深度相机已在（$DEPTH_ID）"
else
    echo "[门禁] 等深度相机 $DEPTH_ID …"
fi

_n=0
while ! cam_ok; do
    if [ "$(remaining)" -le 0 ]; then
        echo "[门禁] ❌ 等深度相机超时（${TIMEOUT}s）：没见过 $DEPTH_ID"
        echo
        echo "       现在 USB 上跟微软有关的设备："
        if lsusb | grep -qi '045e'; then
            lsusb | grep -i '045e' | sed 's/^/         /'
        else
            echo "         （一个都没有 —— 线没插？相机没上电？）"
        fi
        echo
        # ★ 这条是【实测记录过的典型症状】，直接点出来省得来回查：
        if lsusb -d "$COLOR_ID" >/dev/null 2>&1; then
            echo "       ★ 彩色相机（$COLOR_ID）在，但深度相机不在 ——"
            echo "         这正是【供电不足】的典型样子（说明书里记过）："
            echo "           · 深度处理器要的电流大，电不够时它根本不枚举，"
            echo "             而彩色相机照样能用。"
            echo "           · 指示灯会【琥珀色闪】。"
            echo "        怎么修："
            echo "           ① 相机电源适配器插上了吗（要单独供电，不是只插 USB）"
            echo "           ② 板子供电走 DC 桶形口，别走 USB host 口（带不动）"
        else
            echo "       彩色相机（$COLOR_ID）也不在 → 更像是【线/口】的问题："
            echo "           ① micro-USB 线插紧了吗，换根线试"
            echo "           ② 相机电源插了吗"
        fi
        echo "       跑一下体检： bash ~/work/azure_kinect/k4a_probe.py"
        exit 1
    fi
    _n=$((_n + 1))
    [ $((_n % 5)) -eq 0 ] && echo "[门禁]   …还在等相机（已 ${_n}s，剩 $(remaining)s）"
    sleep 1
done
[ "$_n" -gt 0 ] && echo "[门禁] ✅ 深度相机就绪（等了 ${_n}s）"

# =====================================================================
# 3. 稳定一下
# =====================================================================
# 刚枚举出来的 USB 设备 / 刚起来的 X，立刻开会偶发失败。
S=2
[ "$(remaining)" -lt "$S" ] && S=0
if [ "$S" -gt 0 ]; then
    echo "[门禁] 稳定 ${S}s 后放行…"
    sleep "$S"
fi

echo "[门禁] ✅ 全部就绪，交给 run_azure.py"
exit 0
