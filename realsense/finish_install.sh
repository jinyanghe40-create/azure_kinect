#!/bin/bash
# =====================================================================
#  RealSense SDK 收尾安装 —— 在【板子上】用 sudo 跑（编译已完成）
#
#  用法（在板子上）:
#      sudo bash ~/work/realsense/finish_install.sh
#
#  做四件事：
#    ① make install   把刚编好的 SDK 装到 /usr/local
#    ② ldconfig       刷新动态库缓存
#    ③ 装 udev 规则   ★ 关键！否则普通用户打不开 USB 设备
#    ④ 验证          库能不能加载、Python 绑定能不能 import
# =====================================================================
set -u
if [ "$(id -u)" != "0" ]; then
    echo "❌ 要用 sudo 跑:  sudo bash $0"
    exit 1
fi

REAL_USER=${SUDO_USER:-nano}
BUILD=/home/$REAL_USER/librealsense/build
SRC=/home/$REAL_USER/librealsense

hr() { echo "=============================================="; }

hr; echo " ① make install"; hr
cd "$BUILD" || { echo "❌ 找不到 $BUILD"; exit 1; }
make install 2>&1 | tail -8
echo "  ✅ 完成"

hr; echo " ② ldconfig（刷新动态库缓存）"; hr
ldconfig
echo "  ✅ 完成"

hr; echo " ③ 装 udev 规则（让普通用户能打开相机）"; hr
cp "$SRC/config/99-realsense-libusb.rules" /etc/udev/rules.d/
udevadm control --reload-rules
udevadm trigger --subsystem-match=usb    # ★ 只触发 USB，别用无参数的（那会触发所有设备，动作太大）
echo "  ✅ 已装 /etc/udev/rules.d/99-realsense-libusb.rules"
echo "     这条规则给 video/plugdev 组开放了 RealSense 的 USB 设备"
echo "     ⚠️ 需要【拔插一次相机】才生效"

hr; echo " ④ 验证"; hr
echo -n "  librealsense2 库:  "
if ldconfig -p | grep -q librealsense2; then
    ldconfig -p | grep librealsense2 | head -2 | sed 's/^/    /'
else
    echo "❌ 找不到"
fi

echo -n "  rs-enumerate-devices: "
if command -v rs-enumerate-devices >/dev/null; then
    echo "✅ $(command -v rs-enumerate-devices)"
else
    echo "❌ 不在 PATH"
fi

echo "  pyrealsense2（Python 绑定）:"
if su - "$REAL_USER" -c "python3 -c 'import pyrealsense2; print(\"    ✅ 可用，版本\", pyrealsense2.__version__)'" 2>/dev/null; then
    :
else
    echo "    ⚠️ 默认路径 import 不到，自动修一下"
    SA=$(find /usr/local -name "pyrealsense2*.so" 2>/dev/null | head -1)
    if [ -n "$SA" ]; then
        DIR=$(dirname "$SA")
        SP=$(su - "$REAL_USER" -c "python3 -c 'import site; print(site.getsitepackages()[0])'" 2>/dev/null)
        echo "    把 $DIR 加进 $SP/pyrealsense2_path.pth"
        mkdir -p "$SP"
        echo "$DIR" > "$SP/realsense-sdk.pth"
        chown "$REAL_USER:$REAL_USER" "$SP/realsense-sdk.pth" 2>/dev/null
        su - "$REAL_USER" -c "python3 -c 'import pyrealsense2; print(\"    ✅ 修好了，版本\", pyrealsense2.__version__)'" 2>/dev/null \
          || echo "    ❌ 还是不行，把上面输出发我"
    else
        echo "    ❌ 没找到 pyrealsense2*.so"
    fi
fi

hr
echo " ✅ 收尾完成！"
echo
echo " 接下来："
echo "   1. 【拔插一次相机】（udev 规则要重新加载设备）"
echo "   2. 在板子上跑:      rs-enumerate-devices | head -20"
echo "   3. 图形界面:        realsense-viewer"
echo "   4. 我们的脚本:      cd ~/work/realsense && python3 realsense_capture.py --list"
hr
