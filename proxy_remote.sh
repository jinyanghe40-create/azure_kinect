#!/bin/bash
# =====================================================================
#  在【板子上】执行：设置/撤销 代理
#  由电脑上的 board.sh proxy on/off 通过 ssh 调用，不要手动跑
#
#  要配【两套】，因为不同的程序读不同的地方：
#    ① 命令行程序（curl/wget/pip/apt/git/python）→ 读 http_proxy 环境变量
#    ② 图形程序（Chromium/Firefox）             → 读桌面设置 gsettings
#       ★ 浏览器【完全不理】http_proxy 环境变量！
#         所以会出现"命令行能上网，浏览器完全打不开"这种现象
#
#  还有一个 bash 坑：
#    Ubuntu 的 ~/.bashrc 开头有
#        case $- in *i*) ;; *) return;; esac
#    —— 非交互式 shell 直接 return，写在文件【末尾】的 export 永远执行不到！
#    所以必须插到 .bashrc 的【最前面】。~/.profile 管登录 shell，两个都写才保险。
# =====================================================================
set -u

ACTION=${1:-on}
PROXY=${2:-10.42.0.1:7890}
PHOST=${PROXY%:*}
PPORT=${PROXY#*:}

BLOCK="# === board-proxy ===
export http_proxy=http://${PROXY}
export https_proxy=http://${PROXY}
export all_proxy=socks5://${PROXY}
export no_proxy=localhost,127.0.0.1,10.42.0.0/24
export HTTP_PROXY=http://${PROXY}
export HTTPS_PROXY=http://${PROXY}
# === end ==="

# ---------------- 撤销 ----------------
if [ "$ACTION" = "off" ]; then
    sed -i '/# === board-proxy ===/,/# === end ===/d' "$HOME/.bashrc" 2>/dev/null || true
    sed -i '/# === board-proxy ===/,/# === end ===/d' "$HOME/.profile" 2>/dev/null || true
    echo "✅ 已清掉命令行代理（~/.bashrc 和 ~/.profile）"
    if command -v gsettings >/dev/null 2>&1 && [ -S "/run/user/$(id -u)/bus" ]; then
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus" \
            gsettings set org.gnome.system.proxy mode 'none' 2>/dev/null \
          && echo "✅ 已关掉桌面代理"
    fi
    exit 0
fi

# ---------------- ① 命令行程序：环境变量 ----------------
sed -i '/# === board-proxy ===/,/# === end ===/d' "$HOME/.bashrc" 2>/dev/null || true
sed -i '/# === board-proxy ===/,/# === end ===/d' "$HOME/.profile" 2>/dev/null || true

# 插到 .bashrc 最前面 —— 必须在"非交互就 return"之前
{
    printf '%s\n' "$BLOCK"
    cat "$HOME/.bashrc"
} > /tmp/.bashrc.new && mv /tmp/.bashrc.new "$HOME/.bashrc"

# 也写到 .profile（登录 shell 用）
printf '%s\n' "$BLOCK" >> "$HOME/.profile"

echo "✅ ① 命令行程序（curl/wget/pip/apt/git/python）"
echo "      已写入 ~/.bashrc（顶部）和 ~/.profile"

# ---------------- ② 图形程序：gsettings ----------------
if command -v gsettings >/dev/null 2>&1 && [ -S "/run/user/$(id -u)/bus" ]; then
    export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"
    gsettings set org.gnome.system.proxy mode 'manual'
    gsettings set org.gnome.system.proxy.http host "$PHOST"
    gsettings set org.gnome.system.proxy.http port "$PPORT"
    gsettings set org.gnome.system.proxy.https host "$PHOST"
    gsettings set org.gnome.system.proxy.https port "$PPORT"
    gsettings set org.gnome.system.proxy ignore-hosts \
        "['localhost', '127.0.0.0/8', '::1', '10.42.0.0/24']"
    echo "✅ ② 图形程序（Chromium / Firefox）"
    echo "      桌面代理已设为 $PHOST:$PPORT（浏览器读这个，不读环境变量）"
else
    echo "ℹ️  ② 没检测到桌面会话（dbus），跳过 gsettings"
    echo "      板子登录桌面后重跑一次 $0 on 即可"
fi

echo
echo "代理地址: $PROXY"
echo
echo "验证（在板子上另开一个终端）："
echo "   命令行: python3 -c \"import urllib.request;print(urllib.request.urlopen('https://www.google.com',timeout=15).status)\""
echo "   浏览器: 打开 Chromium 访问 google.com"
