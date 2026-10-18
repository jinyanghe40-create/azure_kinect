#!/usr/bin/env bash
# 一次性安装：让 Azure Kinect 视觉流水线【开机自启】，并把结果从
# /dev/ttyGS0 发给电控。
#
# 为什么要有这个服务：
#   现在靠手工 `bash start_azure.sh`。机器人上电后没人去敲那行命令，
#   所以必须做成 systemd 服务。用户已明确要求：
#     ★【只开机自启，崩了不重启】—— 机器人上宁可停下来被人发现，
#       也不要静默重启循环、让电控一直收到过期数据。
#
# ★ 为什么不直接 ExecStart=start_azure.sh：
#   start_azure.sh 最后是 `setsid nohup python3 ... &` —— 它【后台化后自己退出】。
#   systemd 会判定"主进程已退出"→ 服务立刻变 inactive/failed。
#   Type=forking 也救不了：start_azure.sh 【从不写 PID file】，没地方拿 $!。
#   → 服务里直接跑 python；start_azure.sh 原样保留做手动/调试入口，互不干扰。
#
# 用法（板子上）:
#   sudo bash ~/work/azure_kinect/install_vision_service.sh
#        只装 + enable（不启动，不擅自动相机、不驱动电控）
#   sudo bash ~/work/azure_kinect/install_vision_service.sh --start
#        装完顺便启动，并做实测验证
#
# 依赖：请先跑过这两个
#   sudo bash ~/work/azure_kinect/k4a_udev_setup.sh     # 相机权限
#   sudo bash ~/work/azure_kinect/ttygs_udev_setup.sh   # 串口权限 + 赶走 getty
#
# 装完:
#   systemctl status k4a-vision
#   journalctl -u k4a-vision -f
#
# 卸载:
#   sudo systemctl disable --now k4a-vision && sudo rm -f /etc/systemd/system/k4a-vision.service
set -e

if [ "$(id -u)" != "0" ]; then
    echo "❌ 需要 root：sudo bash $0"
    exit 1
fi

# ---------- 参数 ----------
DO_START=0
case "${1:-}" in
    --start) DO_START=1 ;;
    "")      ;;
    *)       echo "用法: $0 [--start]"; exit 2 ;;
esac

# ---------- 路径（从脚本自己的位置推，别写死）----------
HERE="$(cd "$(dirname "$0")" && pwd)"          # …/work/azure_kinect
WORK="$(cd "$HERE/.." && pwd)"                 # …/work
SVC_USER="${K4A_USER:-nano}"
SVC_HOME="$(getent passwd "$SVC_USER" | cut -d: -f6)"
SVC=/etc/systemd/system/k4a-vision.service
SERIAL="${K4A_SERIAL:-/dev/ttyGS0}"
ENGINE="$WORK/models/best_trt.engine"

echo "════════════════════════════════════════════════════"
echo " 安装 k4a-vision 开机自启服务"
echo "════════════════════════════════════════════════════"
echo "  运行用户 : $SVC_USER  (HOME=$SVC_HOME)"
echo "  工作目录 : $WORK"
echo "  串口     : $SERIAL"
echo "  引擎     : $ENGINE"
echo

# =====================================================================
# 1. 前置检查（每条给不同的、能直接照做的报错）
# =====================================================================
echo "=== 1. 文件在不在 ==="
BAD=0
for f in "$HERE/run_azure.py" "$HERE/wait_ready.sh" "$WORK/common/serial_out.py" \
         "$WORK/realsense/run_realsense.py"; do
    if [ -f "$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ 缺文件：$f"
        BAD=1
    fi
done
if [ -f "$ENGINE" ]; then
    echo "  ✅ $ENGINE  ($(du -h "$ENGINE" | cut -f1))"
else
    echo "  ❌ 缺 TensorRT 引擎：$ENGINE"
    echo "     这是训好/编译好的东西，必须存在。看看 $WORK/models/ 里有什么。"
    BAD=1
fi
[ "$BAD" = "0" ] || { echo; echo "先把上面缺的补上再装。"; exit 1; }

chmod +x "$HERE/wait_ready.sh" "$HERE/run_azure.py" 2>/dev/null || true

# =====================================================================
# 2. 【最容易翻车的一步】用"服务的环境"试一遍
# =====================================================================
# 手动跑能行、做成服务就挂 —— 九成是【环境变量差异】。
# 这里用 env -i 清空环境，只给服务会有的那几个变量，提前把问题炸出来。
echo
echo "=== 2. 用服务环境试 import + 开串口 ==="
#   ★ PATH 用 systemd 的默认值
#   ★ runuser 会带上 nano 的补充组（含 plugdev）→ 串口权限能一起验
#   ★ 也带上 HOME（保险）；但实测【没有 HOME 也能 import 到用户级的 pyserial】，
#     因为 expanduser 会回退查 pwd。所以别把这个失败当成 HOME 的问题。
PROBE='
import sys
bad = []
for m in ("numpy", "cv2", "tensorrt", "serial"):
    try:
        __import__(m)
    except Exception as e:
        bad.append("%s: %s: %s" % (m, type(e).__name__, e))
if bad:
    print("IMPORTFAIL " + " | ".join(bad))
    sys.exit(1)
print("IMPORTS ok")
try:
    import serial
    s = serial.Serial(sys.argv[1], 115200, timeout=0.1)
    s.close()
    print("SERIAL ok")
except Exception as e:
    print("SERIALFAIL %s: %s" % (type(e).__name__, e))
'
OUT="$(runuser -u "$SVC_USER" -- env -i \
        HOME="$SVC_HOME" \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        python3 -c "$PROBE" "$SERIAL" 2>&1)" || true
echo "$OUT" | sed 's/^/    /'

case "$OUT" in
    *IMPORTFAIL*)
        echo
        echo "  ❌ 服务环境里 import 不到库 —— 这样装上去必然起不来。"
        echo "     上面 IMPORTFAIL 那行写明了【哪个库、什么错】，照它修。"
        echo "     如果缺的是用户级装的（pyserial 在 $SVC_HOME/.local 下），"
        echo "     装成系统级最省事："
        echo "         sudo -H pip3 install pyserial"
        exit 1
        ;;
esac
echo "  ✅ import 没问题"

case "$OUT" in
    *SERIALFAIL*)
        echo
        echo "  ⚠️  服务环境里打不开 $SERIAL —— 别忘了先跑："
        echo "        sudo bash $HERE/ttygs_udev_setup.sh"
        echo "      （只影响“能不能发给电控”，不影响检测本身）"
        ;;
    *)
        echo "  ✅ $SERIAL 打得开"
        ;;
esac

# =====================================================================
# 3. 提醒项（警告，不阻止安装）
# =====================================================================
echo
echo "=== 3. 提醒 ==="

if systemctl is-active --quiet k4a-xorg 2>/dev/null; then
    echo "  ✅ k4a-xorg 在跑（深度引擎要的 GLX 有了）"
else
    echo "  ⚠️  k4a-xorg 没在跑 —— 服务会卡在门禁等 X。先装："
    echo "        sudo bash $HERE/install_x_service.sh"
fi

if [ -d /var/log/journal ]; then
    echo "  ✅ /var/log/journal 存在（journal 是持久的，重启后日志还在）"
else
    echo "  ⚠️  /var/log/journal 不存在 → journald 是 Storage=auto，"
    echo "      会退化成【纯内存】，reboot 后日志全丢，出问题没法查。修："
    echo "        sudo mkdir -p /var/log/journal"
    echo "        sudo systemd-tmpfiles --create --prefix /var/log/journal"
    echo "        sudo systemctl restart systemd-journald"
fi

if [ -f /etc/tmpfiles.d/tmp.conf ]; then
    echo "  ✅ /etc/tmpfiles.d/tmp.conf 在（/tmp 被清空的坑修过了）"
else
    echo "  ℹ️  没修过 /tmp 清理的坑（本服务不写 /tmp，但其他脚本会）："
    echo "        sudo bash $HERE/fix_tmp_cleanup.sh"
fi

# 串口权限现状
if [ -e "$SERIAL" ]; then
    echo "  ℹ️  $SERIAL 现在：$(ls -l "$SERIAL" | awk '{print $1, $3, $4}')"
    if [ "$(stat -c '%G' "$SERIAL")" != "plugdev" ]; then
        echo "      ⚠️  组不是 plugdev → $SVC_USER 打不开。跑："
        echo "        sudo bash $HERE/ttygs_udev_setup.sh"
    fi
fi

# =====================================================================
# 4. 写服务单元
# =====================================================================
# ★ ExecStart 写成【一行】：下面 heredoc 是【不带引号】的（要展开 $HERE 等），
#   而 bash 会把 `\`+换行 当成【行继续】吃掉 —— 那样 systemd 就看不到
#   多行参数了。写成一行最省事，也不依赖 systemd 的行继续语法。
# ★★ `--rotate` 必须与 body_calib.json 里的 image_rotate_deg【完全一致】。
#   run_azure 会硬断言这一点，改一个不改另一个就起不来。
#   2026-10-18：板上实测确认相机【倒装】—— rotate 0 时画面倒置、2D 检测 0 个框；
#   rotate 180 时画面正立、同一场景检出 2 个箱子；check_gravity 也测出 +Y 朝上。
#   所以生产用 --rotate 180，标定文件也用 roll=180 / image_rotate_deg=180。
#   （2026-10-17 曾按用户当时命令行写成 0，当时无画面可验证，已纠正。）
# ★★ `--calib`：缺文件就【拒绝启动】（exit 1）。这是故意的 ——
#   没有标定文件时静默退回相机系，电控分不清坐标是哪个系。
#   现在的 body_calib.json 是零偏置声明（只含 180° 倒装），不是卷尺量出来的。
# ★★ `--no-gravity`：车体系输出不读 IMU（标定存在时本来也会忽略重力）。
#   显式写出来，免得以后有人以为服务还在做重力对齐。
EXEC_ARGS="--rotate 180 --calib $HERE/body_calib.json --no-gravity --serial $SERIAL --baud 115200 --hz 10 --mm --print --stride 2 --min-dist 0.3 --max-dist 1.5 --min-cluster-points 200 --only-classified"

echo
echo "=== 4. 写 $SVC ==="
cat > "$SVC" <<EOF
[Unit]
Description=Azure Kinect vision pipeline (detect + 3D pose -> serial to MCU)

# X 必须先起来：深度引擎硬依赖 GLib/GLX，否则报 "Depth engine ... error code: 204"
Wants=k4a-xorg.service
After=k4a-xorg.service

# ★ 故意【不加】network-online.target：
#   这次的 ExecStart 里没有 --ros（不再往电脑发 RViz 数据），加它只会在
#   启动路径上白等 30~90 秒网络就绪。
#
# ⚠ 绝对不要加下面两个 —— 会把服务需要的东西藏起来：
#   PrivateTmp=yes   → 看不见 /tmp/.X11-unix/X1 → "X 不存在"
#   ProtectHome=yes  → 看不见 $WORK

[Service]
Type=simple
User=$SVC_USER
WorkingDirectory=$HERE

# 深度引擎要 GLX，必须有 DISPLAY
Environment=DISPLAY=:1

# HOME：pyserial 是【用户级】安装（$SVC_HOME/.local/lib/python3.6/site-packages）。
# ★ 实测（2026-10-12）：就算【不给】HOME，Python 也会从 pwd 库里查到
#   家目录（os.path.expanduser("~") 会回退到 getpwuid），
#   ENABLE_USER_SITE 依然为 True，serial 照样 import 得到。
#   所以这行是【保险】，不是必需 —— 有些库会直接读 \$HOME 而不走 expanduser。
Environment=HOME=$SVC_HOME

# TRT 引擎要加载几十秒，stdout 默认是块缓冲的；
# 不加这个，journalctl 里会"看起来没启动"（其实正在加载引擎）。
Environment=PYTHONUNBUFFERED=1

# 门禁：等 X + 等深度相机(045e:097c) + 挡重复启动，最多等 120 秒。
# ★ 这个守卫必须放在脚本【里面】，不能内联成
#   ExecStartPre=/bin/sh -c '... pgrep -f run_azure.py ...'
#   —— 那样 pgrep 会匹配到 sh 自己所在的命令行，永远返回 0，
#      服务就再也起不来了。
ExecStartPre=$HERE/wait_ready.sh 120

ExecStart=/usr/bin/python3 $HERE/run_azure.py $EXEC_ARGS

# ★ 崩了不自动重启：机器人上宁可停下来被人发现，
#   也不要静默重启循环、让电控一直收到过期数据。
Restart=no

# ★ run_azure.py 捕获 KeyboardInterrupt 走 finally 关相机/关串口。
#   默认的 SIGTERM 【不会】触发 finally → 相机会被占着不放。
KillSignal=SIGINT
TimeoutStopSec=15

# 默认 TimeoutStartSec=90s，比门禁的 120s 短，
# 会在门禁还没等到时就报 "start operation timed out"。
TimeoutStartSec=180

# ★ 不能用 StandardOutput=append:/... ：板上是 systemd 237，
#   append: 要 240 才有，写了会【解析失败】，整个服务都装不上。
StandardOutput=journal
StandardError=journal

LimitNOFILE=4096

[Install]
WantedBy=multi-user.target
EOF
echo "  ✅ 已写"

systemctl daemon-reload
systemctl enable k4a-vision.service >/dev/null 2>&1 || true
echo "  ✅ 已 enable（开机自启）"

# =====================================================================
# 5. 收尾
# =====================================================================
echo
echo "════════════════════════════════════════════════════"
if [ "$DO_START" = "0" ]; then
    echo "✅ 装好了，但【没有启动】（安装器不该擅自动相机、驱动电控）。"
    echo
    echo "现在手动启动、看着它跑："
    echo "    sudo systemctl start k4a-vision"
    echo "    journalctl -u k4a-vision -f"
    echo
    echo "下次重启会自动起来。"
    exit 0
fi

echo "启动并验证…"
echo "════════════════════════════════════════════════════"
systemctl start k4a-vision

# 有界等待：引擎加载要几十秒，别急着下结论
OK=0
for i in $(seq 1 40); do
    sleep 2
    if ! systemctl is-active --quiet k4a-vision; then
        echo "  ❌ 服务退出了（第 $((i*2))s）"
        break
    fi
    # 看到第一帧检测输出就算起来了
    if journalctl -u k4a-vision --no-pager -n 50 2>/dev/null | grep -q '\[0*1\]'; then
        echo "  ✅ 已经开始出检测结果（第 $((i*2))s）"
        OK=1
        break
    fi
    [ $((i % 5)) -eq 0 ] && echo "  …还在等（第 $((i*2))s，引擎加载中）"
done

echo
if [ "$OK" = "1" ]; then
    # ★ 验一下服务进程真的有 plugdev 组 —— 不然串口打不开，
    #   而这只会在"要发数据的那一刻"才暴露，很容易漏掉。
    PID="$(systemctl show -p MainPID --value k4a-vision 2>/dev/null || echo 0)"
    if [ -n "$PID" ] && [ "$PID" != "0" ] && [ -r "/proc/$PID/status" ]; then
        if grep -q '^Groups:' "/proc/$PID/status"; then
            GRPS="$(grep '^Groups:' "/proc/$PID/status")"
            PLUGDEV_GID="$(getent group plugdev | cut -d: -f3)"
            echo "  $GRPS"
            if echo " $GRPS " | grep -q " $PLUGDEV_GID "; then
                echo "  ✅ 服务进程有 plugdev 组（GID $PLUGDEV_GID）→ 能写 $SERIAL"
            else
                echo "  ⚠️  服务进程【没有】plugdev 组（GID $PLUGDEV_GID）→ 写不了 $SERIAL"
                echo "      修：在 $SVC 的 [Service] 里加一行"
                echo "          SupplementaryGroups=plugdev"
                echo "      然后 systemctl daemon-reload && systemctl restart k4a-vision"
            fi
        fi
    fi
    echo
    echo "  最近几行日志："
    journalctl -u k4a-vision --no-pager -n 6 2>/dev/null | sed 's/^/    /'
    echo
    echo "✅ 服务在跑。看实时日志： journalctl -u k4a-vision -f"
    echo "   停止：             sudo systemctl stop k4a-vision"
else
    echo "❌ 没起来。看日志："
    echo "    journalctl -u k4a-vision -n 60 --no-pager"
    echo
    echo "常见原因："
    echo "  · 卡在门禁 → 看 wait_ready.sh 打的那几行（它会把原因说清楚）"
    echo "  · 相机没插 / 供电不足（琥珀色闪灯）"
    echo "  · 引擎加载失败（Cublas / 反序列化）→ 见仓库根目录 bug.txt"
    exit 1
fi
