#!/bin/bash
# =====================================================================
#  把板子改成【真·无显示器】模式 —— 在【板子上】用 sudo 跑
#
#  用法（在板子上）:
#      sudo bash ~/work/headless_setup.sh
#
#  ★★ 跑完【不用重启】！桌面立刻停掉，GPU 和内存立刻释放。
#     只有"下次开机也不启动桌面"这一条等下次重启生效 —— 但你现在需要的
#     效果（不占 GPU、不占内存）是【立刻】就有的。
#
#  ⚠️ 为什么要做这个（今天踩的坑）：
#     板子一直跑着完整的 GNOME 桌面（gdm3 + Xorg + compiz），
#     而 compiz 占着 /dev/nvmap 和 /dev/nvhost-ctrl —— 那是 GPU 的设备。
#     它和 TensorRT/CUDA 抢 GPU，症状是：
#         · CUDA initialization failure with error 999
#         · Could not initialize cublas
#         · 拔显示器后整个系统卡死 / 卡在 NVIDIA logo（只能拔电源）
#     而且桌面白吃 500MB 内存 —— 你这是要靠它跑推理的板子。
#
#  改完之后：
#     · 桌面立刻停掉，多出 400~500MB 内存，GPU 不再被抢
#     · SSH 照常用（SSH 不需要图形界面）
#     · ⚠️ 板子上的 rs-pointcloud / realsense-viewer 会打不开
#          → 改用电脑上的 RViz（b ros + rviz2）
#
#  想改回来:
#     sudo systemctl set-default graphical.target
#     sudo systemctl enable --now gdm3
# =====================================================================
set -u
if [ "$(id -u)" != "0" ]; then
    echo "❌ 要用 sudo 跑:  sudo bash $0"
    exit 1
fi

hr() { echo "=============================================="; }
gpu_holders() {
    for x in /dev/nvmap /dev/nvhost-ctrl /dev/nvhost-gpu; do
        printf "      %-20s " "$x"
        fuser "$x" 2>/dev/null | head -1 || echo "空闲"
    done
}

hr; echo " 改之前的状态"; hr
echo "  默认启动目标 : $(systemctl get-default)"
echo "  gdm3         : $(systemctl is-active gdm3 2>/dev/null || echo 未运行)"
echo "  内存占用     : $(free -m | awk '/Mem/{print $3" / "$2" MB"}')"
echo "  GPU 被谁占用 :"
gpu_holders

hr; echo " ① 立刻停掉桌面（这一步就有效果，不用重启）"; hr
systemctl stop gdm3 2>/dev/null && echo "  ✅ gdm3 已停止" || echo "  （gdm3 本来就没在跑）"
pkill -f "/usr/bin/compiz" 2>/dev/null && echo "  ✅ compiz 已结束" || true
sleep 2

hr; echo " ② 让它开机也不启动（下次重启生效）"; hr
systemctl disable gdm3 2>/dev/null && echo "  ✅ gdm3 已禁用" || echo "  （gdm3 本来就没启用）"
systemctl set-default multi-user.target
echo "  ✅ 默认启动目标 = $(systemctl get-default)"

hr; echo " ③ 改完之后的状态"; hr
echo "  默认启动目标 : $(systemctl get-default)"
echo "  内存占用     : $(free -m | awk '/Mem/{print $3" / "$2" MB"}')"
echo "  图形进程     : $(ps -eo cmd | grep -cE '[X]org|[c]ompiz|[g]dm3') 个"
echo "  GPU 被谁占用 :"
gpu_holders

hr
echo " ✅ 完成！【不要重启】，现在就有效果了。"
echo
echo " 直接接着跑相机:"
echo "     cd ~/work/realsense && python3 run_realsense.py --print --frames 20 --save-dir out --stride 5 --min-cluster-points 800 --max-dist 1.2"
echo
echo " 想改回图形界面:"
echo "     sudo systemctl set-default graphical.target && sudo systemctl enable --now gdm3"
hr
