#!/bin/bash
# =====================================================================
#  RealSense SDK 编译依赖 —— 在【板子上】用 sudo 跑
#
#  用法（在板子上）:
#      sudo bash ~/work/realsense/install_deps.sh
#
#  ⚠️ 说明：脚本里【故意不用 set -e】。
#     apt/dpkg 经常会打一堆吓人的 warning，但那是警告不是错误；
#     用了 set -e 会因为一条警告就静默退出，让人以为装好了。
#     所以这里改成：一步一步显式检查，最后逐个验证并明确报 PASS/FAIL。
#
#  为什么要装这些：
#      libusb-1.0-0-dev   ★ 必需。Jetson 上必须用 RSUSB 后端，靠 libusb
#                           直接跟相机说话（绕开内核 uvcvideo 驱动）
#      libssl-dev         编译需要
#      libgtk-3-dev       ★ realsense-viewer 的图形界面
#      libglfw3-dev       ★ 例子程序 / 3D 点云窗口
#      libgl1-mesa-dev    3D 渲染
#      python3-dev        ★ pyrealsense2 的 Python 绑定
# =====================================================================

PKGS="git cmake build-essential pkg-config libssl-dev libusb-1.0-0-dev libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev python3-dev python3-pip"

if [ "$(id -u)" != "0" ]; then
    echo "❌ 要用 sudo 跑:  sudo bash $0"
    exit 1
fi

hr() { echo "=============================================="; }

hr; echo " ⓪ 环境自检"; hr
MEM=$(free -m | awk '/Mem/{print $7}')
echo "  可用内存: ${MEM} MB"
echo "  可用磁盘: $(df -h / | tail -1 | awk '{print $4}')"
if [ "$MEM" -lt 400 ]; then
    echo "  ⚠️ 可用内存不足 400MB，apt 可能被杀。建议先关掉占内存的程序"
fi

# ---------- ① 修掉 JetPack 镜像自带的小瑕疵 ----------
hr; echo " ① 修 dpkg 文件列表的小瑕疵（fio.list 缺换行）"; hr
if [ -f /var/lib/dpkg/info/fio.list ]; then
    LAST=$(tail -c 1 /var/lib/dpkg/info/fio.list | od -An -c | tr -d ' \n')
    if [ "$LAST" != "\n" ]; then
        printf '\n' >> /var/lib/dpkg/info/fio.list
        echo "  ✅ 已给 fio.list 补上换行，那行 warning 不会再出现"
        echo "     这个文件只是「fio 包装了哪些文件」的清单，缺个换行纯粹是格式瑕疵，无害"
    else
        echo "  ✅ fio.list 正常"
    fi
else
    echo "  （没有 fio.list，跳过）"
fi

# ---------- ② 清理上次中断留下的烂摊子 ----------
hr; echo " ② 清理上次中断的状态"; hr
if pgrep -x apt-get >/dev/null || pgrep -x dpkg >/dev/null; then
    echo "  ⚠️ 还有 apt/dpkg 在跑，等它结束…"
    while pgrep -x apt-get >/dev/null || pgrep -x dpkg >/dev/null; do sleep 2; done
    echo "  ✅ 已结束"
fi
echo "  执行 dpkg --configure -a（把上次没配完的包配完）"
dpkg --configure -a 2>&1 | tail -3
echo "  ✅ 完成"

# ---------- ③ 给 apt 配代理 ----------
hr; echo " ③ 给 apt 配代理"; hr
echo 'Acquire::http::Proxy "http://10.42.0.1:7890";'  > /etc/apt/apt.conf.d/99proxy
echo 'Acquire::https::Proxy "http://10.42.0.1:7890";' >> /etc/apt/apt.conf.d/99proxy
echo "  ✅ 已写入 /etc/apt/apt.conf.d/99proxy"
echo "     （sudo 会清空环境变量，所以 apt 必须用它自己的配置文件）"

# ---------- ④ apt update ----------
hr; echo " ④ apt-get update"; hr
apt-get update 2>&1 | tail -6
echo "  （warning 不用慌，只有 E: 开头的才是错误）"

# ---------- ⑤ 装依赖 ----------
hr; echo " ⑤ 装编译依赖（约 100MB，5~15 分钟，别中断）"; hr
echo "  包列表: $PKGS"
echo
apt-get install -y $PKGS 2>&1 | tail -25
RC=${PIPESTATUS[0]}

# ---------- ⑥ 逐个验证 ----------
hr; echo " ⑥ 验证结果"; hr
MISS=0
for p in git cmake build-essential pkg-config libssl-dev libusb-1.0-0-dev libgtk-3-dev libglfw3-dev libgl1-mesa-dev python3-dev; do
    v=$(dpkg -l "$p" 2>/dev/null | awk '/^ii/{print $3}')
    if [ -n "$v" ]; then printf "  ✅ %-20s %s\n" "$p" "$v"
    else printf "  ❌ %-20s 没装上\n" "$p"; MISS=$((MISS+1)); fi
done

hr
if [ "$MISS" -eq 0 ]; then
    echo " ✅ 全部就绪！（apt 退出码=$RC，0 正常）"
    echo
    echo " 下一步（在电脑上敲，不用 sudo）—— 开始编译 SDK："
    echo "     b run 'bash ~/work/realsense/build.sh'"
    echo " 编译要 40~90 分钟，中途别断电。"
else
    echo " ❌ 还缺 $MISS 个包。把上面 ⑤ 那段的完整输出发我"
fi
hr
