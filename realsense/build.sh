#!/bin/bash
# =====================================================================
#  编译 librealsense —— 在【板子上】跑（不需要 sudo）
#
#  用法（在电脑上）:
#      b run 'bash ~/work/realsense/build.sh'
#
#  为什么要 FORCE_RSUSB_BACKEND=true：
#      RealSense 在普通 PC 上用 V4L2 后端（内核 uvcvideo 驱动），
#      但那需要给内核打补丁，而 **Jetson 的内核是 NVIDIA 定制的，打不了**。
#      所以 Jetson 上必须走 RSUSB 后端 —— 用 libusb 直接跟相机通信，
#      完全不依赖内核驱动。这是 Intel 官方对 Jetson 的推荐做法。
#
#  编译时长：Jetson Nano 4 核大约 40~90 分钟。中途别断电。
# =====================================================================
set -e
SRC=$HOME/librealsense
JOBS=$(nproc)

if [ ! -d "$SRC" ]; then
    echo "❌ 找不到源码 $SRC"
    echo "   先: git clone --depth 1 --branch v2.54.2 https://github.com/IntelRealSense/librealsense.git ~/librealsense"
    exit 1
fi

cd "$SRC"
mkdir -p build && cd build

echo "=============================================="
echo " ① cmake 配置"
echo "=============================================="
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=true \
    -DBUILD_EXAMPLES=true \
    -DBUILD_GRAPHICAL_EXAMPLES=true \
    -DBUILD_PYTHON_BINDINGS=true \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_C_FLAGS="-march=native" \
    -DCMAKE_CXX_FLAGS="-march=native" \
    2>&1 | tail -25

echo
echo "=============================================="
echo " ② 编译（$JOBS 核并行，40~90 分钟，别断电）"
echo "=============================================="
date
make -j"$JOBS" 2>&1 | tail -20
date

echo
echo "=============================================="
echo " ✅ 编译完成"
echo "=============================================="
echo "下一步（需要 sudo，在板子上跑）："
echo "    cd ~/librealsense/build && sudo make install && sudo ldconfig"
echo "    sudo cp ~/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/"
echo "    sudo udevadm control --reload-rules && sudo udevadm trigger"
