#!/usr/bin/env bash
# 一键把 Azure Kinect 全流程跑起来（含 X、日志、ROS）
#
# 为什么要有这个脚本：
#   板子上 /usr/lib/tmpfiles.d/tmp.conf 是
#       D /tmp 1777 root root -
#   ★ D = 每次清理【清空整个 /tmp】，而且没有年龄限制
#     （标准 Ubuntu 是 q /tmp 1777 root root 10d，只删 10 天前的）
#   所以日志/脚本放 /tmp 会被定时清掉，看起来像"进程莫名消失"。
#   → 本脚本把日志放到 ~/work/logs/，绕开这个坑。
#
# 用法（板子上）:
#   bash ~/work/azure_kinect/start_azure.sh              # 默认参数
#   bash ~/work/azure_kinect/start_azure.sh --frames 100 # 透传给 run_azure.py
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HOME/work/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/azure_$(date +%m%d_%H%M).log"

# ---------- 1. 确保有 X（深度引擎要 GLX）----------
DISP="${K4A_DISPLAY:-:1}"
x_ok() { [ -S "/tmp/.X11-unix/X${1#:}" ]; }
if ! x_ok "$DISP"; then
    echo "[X] $DISP 不在，尝试起一个裸 X …"
    if [ "$(id -u)" = "0" ]; then
        setsid Xorg "$DISP" -noreset -nolisten tcp vt7 \
            >"$LOGDIR/xorg.log" 2>&1 </dev/null &
    else
        sudo setsid Xorg "$DISP" -noreset -nolisten tcp vt7 \
            >"$LOGDIR/xorg.log" 2>&1 </dev/null &
    fi
    for _ in $(seq 1 15); do sleep 1; x_ok "$DISP" && break; done
fi
if x_ok "$DISP"; then
    export DISPLAY="$DISP"
    echo "[X] ✅ DISPLAY=$DISP"
else
    echo "[X] ❌ 起不来。看 $LOGDIR/xorg.log"
    echo "    备选：sudo systemctl start gdm3 && export DISPLAY=:0"
    exit 1
fi

# ---------- 2. 别重复起 ----------
if pgrep -f "run_azure.py" >/dev/null 2>&1; then
    echo "[警告] 已经有 run_azure.py 在跑。先停它："
    echo "       pkill -f run_azure.py"
    exit 1
fi

# ---------- 3. 起主程序 ----------
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
    ARGS=(--print --ros 10.42.0.1:9100 --stride 2 --min-dist 0.3 \
          --max-dist 1.5 --min-cluster-points 200 --only-classified)
fi
echo "[跑] python3 run_azure.py ${ARGS[*]}"
echo "[日志] $LOG   （放家目录，不会被 systemd-tmpfiles 清掉）"
cd "$HERE"
setsid nohup python3 run_azure.py "${ARGS[@]}" >"$LOG" 2>&1 </dev/null &
sleep 3
if pgrep -f "run_azure.py" >/dev/null 2>&1; then
    echo "[✅] 已启动。看日志： tail -f $LOG"
else
    echo "[❌] 没起来，看日志： $LOG"
    tail -15 "$LOG"
    exit 1
fi
