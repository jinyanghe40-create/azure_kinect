#!/usr/bin/env bash
# 在【无桌面】的板子上跑 Azure Kinect —— 先保证有 X 显示，再跑命令
#
# 为什么需要：
#   Azure Kinect 的深度引擎库 libdepthengine.so.2.0 硬依赖：
#       libGL.so.1 / libX11.so.6 / libGLX.so.0
#   也就是说它**必须有一个 X11 显示 + GLX 上下文**。
#   板子为了省电/防死机关掉了桌面（multi-user.target），
#   于是深度引擎建不出 GL 上下文，报：
#       Depth engine create and initialize failed with error code: 204
#   （204 = GPU API failure）
#
#   ★ 不需要整个 GNOME 桌面 —— 一个裸 X 服务器就够了（功耗低得多）。
#   ★ 但 Xorg 要 root 才能起，所以这个脚本请用 sudo 跑：
#         sudo bash ~/work/azure_kinect/run_with_x.sh
#     或者不带 sudo，脚本会自己调 sudo 起 X。
#
# 用法（板子上）:
#   sudo bash ~/work/azure_kinect/run_with_x.sh
#   sudo bash ~/work/azure_kinect/run_with_x.sh python3 run_azure.py
set -u

DISP="${K4A_DISPLAY:-:1}"
XVT="${K4A_VT:-7}"

# 判断某个 DISPLAY 能不能用 —— 直接看 X socket，不依赖 xdpyinfo
display_ok() {
    local d="${1:-}"
    [ -n "$d" ] || return 1
    local n="${d#:}"          # :0 -> 0
    n="${n%%.*}"              # :0.0 -> 0
    [ -S "/tmp/.X11-unix/X$n" ]
}

echo "════════════════════════════════════════════"
echo " 准备 X 显示（Azure Kinect 深度引擎需要）"
echo "════════════════════════════════════════════"

# 1) 已经有能用的显示就用它
if display_ok "${DISPLAY:-}"; then
    echo "✅ 已有显示 DISPLAY=$DISPLAY"
elif display_ok "$DISP"; then
    echo "✅ $DISP 已经在跑"
    export DISPLAY="$DISP"
else
    # 2) 没有就起一个裸 X
    echo "  当前没有可用 X（桌面被关掉了，这是为省电特意做的）"
    echo "  尝试起裸 X 服务器：Xorg $DISP vt$XVT"
    echo
    if [ "$(id -u)" = "0" ]; then
        setsid Xorg "$DISP" -noreset -nolisten tcp "vt$XVT" \
            >/tmp/xorg_k4a.log 2>&1 < /dev/null &
    else
        echo "  （需要 root，会提示输密码）"
        sudo setsid Xorg "$DISP" -noreset -nolisten tcp "vt$XVT" \
            >/tmp/xorg_k4a.log 2>&1 < /dev/null &
    fi

    for _ in $(seq 1 15); do
        sleep 1
        if display_ok "$DISP"; then break; fi
    done

    if display_ok "$DISP"; then
        echo "✅ X 起来了：DISPLAY=$DISP   日志 /tmp/xorg_k4a.log"
        export DISPLAY="$DISP"
    else
        echo "❌ X 起不来。下面是诊断："
        echo "──── /tmp/xorg_k4a.log ────"
        tail -15 /tmp/xorg_k4a.log 2>/dev/null || echo "  (没有日志)"
        echo "───────────────────────────"
        echo
        echo "  常见原因和备选："
        echo "   a) 没接显示器/欺骗器 → Xorg 找不到输出。"
        echo "      现在是: $(for f in /sys/class/drm/*/status; do \
            printf '%s=%s ' "$(basename "$(dirname "$f")")" "$(cat "$f")"; done 2>/dev/null)"
        echo "   b) 换个 VT：K4A_VT=8 sudo -E bash $0"
        echo "   c) 起完整桌面（功耗高，但肯定能用）："
        echo "        sudo systemctl start gdm3 && export DISPLAY=:0"
        echo "   d) 用 Xvfb（软件 GL，可能慢）："
        echo "        sudo apt install -y xvfb && xvfb-run -a python3 ~/work/azure_kinect/k4a_probe.py"
        exit 1
    fi
fi

echo
echo "════════════════════════════════════════════"
if [ $# -eq 0 ]; then
    set -- python3 "$(dirname "$0")/k4a_probe.py"
fi
echo " DISPLAY=$DISPLAY  执行: $*"
echo "════════════════════════════════════════════"
exec "$@"
