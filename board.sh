#!/bin/bash
# =====================================================================
#  电脑 ↔ Jetson 板子 一键工具
# =====================================================================
#  用法:
#     ./board.sh            看当前状态（板子在不在、IP 多少、挂没挂）
#     ./board.sh up         一键：发现板子 + 挂载 + 打印 VS Code 该打开哪个目录
#     ./board.sh down       一键收工：备份 + 卸载
#     ./board.sh ssh        登录板子（远程终端）
#     ./board.sh run "命令"  在板子上跑一条命令
#     ./board.sh push       电脑 code/ → 板子 ~/work/code/
#     ./board.sh pull       板子 ~/work → 电脑 ~/board_backup/
#     ./board.sh mount      只挂载（VS Code 直接编辑板子文件）
#     ./board.sh umount     只卸载
#     ./board.sh ap         板子怎么上网 / 网络诊断
#     ./board.sh net         分层网络体检（DNS/ping/TCP/代理/HTTP）
#     ./board.sh ros         在电脑上启动 ROS 桥接（RViz 可视化）
#     ./board.sh proxy test  查板子能不能走电脑的梯子
#     ./board.sh proxy on    在板子上配好代理环境变量
#     ./board.sh proxy off   撤销
#
#  前置（只需做一次）:
#     sudo apt install -y sshfs
#     ssh-copy-id nano@10.42.0.26
# =====================================================================
set -u

IFACE=${IFACE:-enp5s0}                       # 电脑上直连板子的网口
BOARD_MAC=${BOARD_MAC:-00:04:4b:e5:45:3b}    # 板子 eth0 的 MAC（认卡用，比 IP 可靠）
BOARD_USER=${BOARD_USER:-nano}
IP_CACHE=$HOME/.board_ip
LOCAL_WORK=${LOCAL_WORK:-$HOME/jetson-deploy}
MOUNT=${MOUNT:-$HOME/board}
SSHOPT="-o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new"

# ---------------------------------------------------------------------
reachable() { ping -c 1 -W 1 "$1" >/dev/null 2>&1; }

# ★ 挂载点是否"卡死"：远端失联后 sshfs 会变成僵尸挂载，
#   任何访问它的程序都会永久卡住（文件管理器打不开主文件夹就是这个原因）。
#   用 timeout 探测，探不动就是卡死。
mount_hung() {
    [ -d "$MOUNT" ] || return 1
    mountpoint -q "$MOUNT" || return 1
    timeout 3 ls "$MOUNT" >/dev/null 2>&1 && return 1
    return 0
}

# 强杀僵尸挂载（-z = 懒惰卸载，立即摘掉，不等它响应）
force_umount() {
    mountpoint -q "$MOUNT" || return 0
    timeout 8 fusermount -u "$MOUNT" 2>/dev/null && return 0
    echo "  （普通卸载卡住了，改用强制卸载 -z）"
    timeout 8 fusermount -uz "$MOUNT" 2>/dev/null && return 0
    return 1
}

arp_lookup() {
    ip neigh show dev "$IFACE" 2>/dev/null \
      | awk -v m="$BOARD_MAC" 'tolower($0) ~ tolower(m) {print $1; exit}'
}

sweep() {
    # 广播 ping 一遍，逼所有在线设备回 ARP（比逐个 ping 快很多）
    echo "  …扫描 10.42.0.0/24（约 3 秒）"
    for i in $(seq 1 254); do (ping -c 1 -W 1 "10.42.0.$i" >/dev/null 2>&1 &) ; done
    sleep 3
}

discover() {
    local ip
    # ① 上次记下来的 IP
    if [ -f "$IP_CACHE" ]; then
        ip=$(cat "$IP_CACHE")
        reachable "$ip" && { echo "$ip"; return 0; }
    fi
    # ② ARP 表里按 MAC 找
    ip=$(arp_lookup)
    if [ -n "$ip" ] && reachable "$ip"; then echo "$ip" > "$IP_CACHE"; echo "$ip"; return 0; fi
    # ③ 扫一遍再找
    sweep
    ip=$(arp_lookup)
    if [ -n "$ip" ]; then echo "$ip" > "$IP_CACHE"; echo "$ip"; return 0; fi
    return 1
}

need_ip() {   # 拿到 IP 或退出
    BOARD_IP=$(discover) || {
        cat <<EOF
❌ 找不到板子。按顺序查：
   ① 网线插好了吗（电脑 $IFACE  ←→  板子 eth0，网口灯亮不亮）
   ② 电脑网口是不是"共享给其他计算机"？
      nmcli connection show | grep $IFACE
   ③ 板子开机了吗？板子上敲: ip a   看 eth0 有没有 10.42.0.x
   ④ 板子 IP 变了？
      BOARD_IP=10.42.0.99 $0 $*
EOF
        exit 1
    }
    BOARD=${BOARD_USER}@${BOARD_IP}
}

ssh_ok() { ssh $SSHOPT -o BatchMode=yes -o PasswordAuthentication=no "$BOARD" true 2>/dev/null; }
# ---------------------------------------------------------------------

case "${1:-status}" in

  status|ip|"")
    echo "── 电脑这边 ──────────────────────────────"
    ip -br addr show "$IFACE" 2>/dev/null || { echo "❌ 找不到网口 $IFACE"; ip -br addr; }
    nmcli -t -f NAME,DEVICE connection show --active 2>/dev/null | grep "$IFACE" \
      | sed 's/^/   连接: /'
    echo
    echo "── 板子 ──────────────────────────────────"
    if BOARD_IP=$(discover); then
        echo "   IP: $BOARD_IP"
        ping -c 1 -W 1 "$BOARD_IP" >/dev/null 2>&1 && echo "   ping: ✅"
        if ssh $SSHOPT -o BatchMode=yes -o PasswordAuthentication=no \
               "${BOARD_USER}@${BOARD_IP}" 'echo "   SSH : ✅ $(hostname) / $(uptime -p)"' 2>/dev/null; then
            :
        else
            echo "   SSH : ❌ 没通 —— 跑一次: ssh-copy-id ${BOARD_USER}@${BOARD_IP}"
        fi
    fi
    echo
    echo "── 挂载 ──────────────────────────────────"
    if mount_hung; then
        echo "   $MOUNT ⚠️⚠️ 卡死了！"
        echo "      板子失联导致 sshfs 变僵尸挂载 —— 任何访问它的程序都会永久卡住"
        echo "      （文件管理器打不开主文件夹、VS Code 卡死…… 都是这个）"
        echo "      修:  $0 umount"
    elif mountpoint -q "$MOUNT"; then
        echo "   $MOUNT ✅ 已挂载（VS Code 打开这个目录）"
    else
        echo "   $MOUNT 未挂载 —— 跑: $0 up"
    fi
    ;;

  up)
    need_ip "$@"
    echo "✅ 板子: $BOARD"
    ssh_ok || { echo "❌ 免密 SSH 没通 —— 先跑: ssh-copy-id $BOARD"; exit 1; }
    mkdir -p "$MOUNT"
    if mountpoint -q "$MOUNT"; then
        echo "✅ 已挂载: $MOUNT"
    else
        sshfs "$BOARD:/home/${BOARD_USER}/work" "$MOUNT" \
            -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,idmap=user \
            && echo "✅ 已挂载: $MOUNT"
    fi
    echo
    echo "接下来："
    echo "   code $MOUNT            # VS Code 打开板子上的代码"
    echo "   $0 run 'python3 code/trt_infer.py --print'   # 让板子跑"
    echo "   收工: $0 down"
    ;;

  down)
    # ★ 先判板子在不在：不通就别去 rsync（会卡死），直接强杀挂载
    if discover >/dev/null 2>&1 && ssh_ok; then
        mkdir -p "$HOME/board_backup"
        timeout 120 rsync -a --delete "$BOARD:~/work/" "$HOME/board_backup/" 2>/dev/null \
          && echo "✅ 板子 ~/work 已备份到 ~/board_backup/"
    else
        echo "⚠️ 板子联系不上，跳过备份（避免 rsync 卡死）"
    fi
    if mountpoint -q "$MOUNT"; then
        force_umount && echo "✅ 已卸载 $MOUNT" || echo "❌ 卸载失败: sudo umount -l $MOUNT"
    else
        echo "（没挂载）"
    fi
    echo "可以安全给板子断电了。"
    ;;

  ssh)     need_ip "$@"; exec ssh $SSHOPT "$BOARD" ;;

  run)     need_ip "$@"; shift; ssh $SSHOPT "$BOARD" "$@" ;;

  mount)
    need_ip "$@"
    mkdir -p "$MOUNT"
    if mount_hung; then
        echo "⚠️ 发现僵尸挂载，先强杀…"
        force_umount
    fi
    mountpoint -q "$MOUNT" && { echo "已经挂着: $MOUNT"; exit 0; }
    sshfs "$BOARD:/home/${BOARD_USER}/work" "$MOUNT" \
        -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,idmap=user \
        && echo "✅ 板子的 ~/work 已挂到 $MOUNT —— VS Code 打开这个文件夹"
    ;;

  umount)
    if mountpoint -q "$MOUNT"; then
        force_umount && echo "✅ 已卸载 $MOUNT" || echo "❌ 卸载失败，试试: sudo umount -l $MOUNT"
    else
        echo "没挂着"
    fi
    ;;

  push)
    need_ip "$@"
    # ★ 这里是【电脑 → 板子】，会把板子上同名的文件覆盖掉。
    #   板子上有、电脑上没有的文件不会被删（没加 --delete），
    #   但只要两边都有，就换成电脑这份 —— 所以推之前先想清楚哪边新。
    #   （这个仓库踩过：板子比电脑新一天，差点被覆盖。）
    #   不确定就先 `b pull` 备份，或者直接用 git。
    # ★ 排除体积大又不是源码的东西：libs/include 是下载的 SDK，
    #   out*/logs 是产物，_history 是本地的历史备份。
    rsync -av --exclude '__pycache__' --exclude '*.pyc' \
          --exclude 'libs/' --exclude 'include/' --exclude 'pkgs/' \
          --exclude 'out/' --exclude 'out[0-9]*/' --exclude 'logs/' \
          --exclude '_history/' --exclude '*.ply' --exclude '*.avi' \
          --exclude '*.png' --exclude '*.jpg' \
          "$LOCAL_WORK/" "$BOARD:~/work/" \
      && echo "✅ 已同步到板子 ~/work/" \
      && echo "   （注意：两边都有的文件已被【电脑这份】覆盖）"
    ;;

  pull)
    need_ip "$@"
    mkdir -p "$HOME/board_backup"
    rsync -av "$BOARD:~/work/" "$HOME/board_backup/" \
      && echo "✅ 板子 ~/work 已备份到 ~/board_backup/"
    ;;

  ap)
    need_ip "$@"
    echo "── 板子的网 ──────────────────────────────"
    ssh $SSHOPT "$BOARD" 'ip -br addr show eth0; echo; echo -n "默认路由: "; ip route | grep default; echo -n "DNS: "; cat /etc/resolv.conf | grep nameserver | head -2'
    echo
    echo "── 能不能上外网 ──────────────────────────"
    ssh $SSHOPT "$BOARD" 'ping -c 1 -W 3 223.5.5.5 >/dev/null 2>&1 && echo "  IP 层: ✅" || echo "  IP 层: ❌"; wget -4 -q -O /dev/null -T 8 https://mirrors.aliyun.com/ubuntu/dists/bionic/Release 2>/dev/null && echo "  国内 HTTPS: ✅" || echo "  国内 HTTPS: ❌"'
    ;;

  ros)
    # 在电脑上启动 ROS 2 桥接（RViz 可视化）
    if [ ! -f /opt/ros/humble/setup.bash ]; then
        echo "❌ 没装 ROS 2 Humble"; exit 1
    fi
    echo "启动 ROS 桥接（Ctrl+C 退出）"
    echo "另一个终端跑:  source /opt/ros/humble/setup.bash && rviz2 -d $(dirname "$0")/ros_bridge/cloud.rviz"
    echo "板子那边跑:    b run 'cd ~/work/realsense && python3 run_realsense.py --print --ros 10.42.0.1:9100'"
    echo
    exec bash -c "source /opt/ros/humble/setup.bash && /usr/bin/python3 $(dirname "$0")/ros_bridge/cloud_bridge.py"
    ;;

  net)
    # 分层网络体检: board.sh net [主机:端口]
    need_ip "$@"
    ssh $SSHOPT "$BOARD" "bash -s ${2:-10.42.0.1:7890}" < "$(dirname "$0")/net_remote.sh"
    ;;

  proxy)
    # 用法: board.sh proxy on|off|test [主机:端口]
    need_ip "$@"
    ACTION=${2:-test}
    PROXY=${3:-10.42.0.1:7890}
    PHOST=${PROXY%:*}; PPORT=${PROXY#*:}
    case "$ACTION" in
      on)
        ssh $SSHOPT "$BOARD" "bash -s on $PROXY" < "$(dirname "$0")/proxy_remote.sh"
        ;;
      off)
        ssh $SSHOPT "$BOARD" "bash -s off" < "$(dirname "$0")/proxy_remote.sh"
        ;;
      test)
        echo "── ① 板子能不能连上电脑的代理端口 $PROXY ──"
        ssh $SSHOPT "$BOARD" "timeout 4 bash -c 'cat < /dev/null > /dev/tcp/$PHOST/$PPORT' 2>/dev/null && echo '   连通 ✅' || echo '   连不上 ❌ → 电脑上开「允许局域网连接」+ 放行 ufw'"
        echo
        echo "── ② 板子直连谷歌（不走代理，预期失败）──"
        ssh $SSHOPT "$BOARD" 'python3 -c "
import urllib.request
try:
    print(\"   竟然通了?\", urllib.request.urlopen(\"https://www.google.com\", timeout=6).status)
except Exception as e:
    print(\"   失败（符合预期）:\", type(e).__name__)
"'
        echo
        echo "── ③ 板子走代理抓谷歌 ──"
        ssh $SSHOPT "$BOARD" "python3 -c \"
import urllib.request
o=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://$PROXY','https':'http://$PROXY'}))
try:
    print('   ✅ 成功！HTTP', o.open('https://www.google.com', timeout=15).status)
except Exception as e:
    print('   ❌ 失败:', type(e).__name__, e)
\""
        ;;
      *) echo "用法: $0 proxy on|off|test [主机:端口]" ;;
    esac
    ;;

  *)
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
