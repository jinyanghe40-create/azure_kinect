#!/bin/bash
# =====================================================================
#  Jetson Nano 部署脚本 —— 安装 Azure Kinect 的 ARM64 库
# =====================================================================
#  这个脚本做三件事：
#    1. 把 ARM64 库装到系统目录
#    2. 建软链接（让程序能找到库）
#    3. 验证库能不能加载
#
#  用法（在 U 盘目录里）：
#      sudo bash deploy_setup.sh
#
#  为什么要 sudo：要往 /usr/lib 和 /usr/include 写文件
# =====================================================================

set -e   # 任何一条命令失败就停下（避免装一半留下烂摊子）

HERE="$(cd "$(dirname "$0")" && pwd)"
LIBDIR=/usr/lib/aarch64-linux-gnu
INCDIR=/usr/include

echo "=============================================="
echo " Jetson Nano —— Azure Kinect ARM64 库安装"
echo "=============================================="
echo "  包目录: $HERE"
echo "  库目标: $LIBDIR"
echo ""

# ---------- 检查是不是在正确的板子上 ----------
echo "[1/5] 检查环境"
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "  ❌ 当前架构是 $ARCH，不是 aarch64"
    echo "     这个脚本只能在 Jetson（ARM64）上跑"
    exit 1
fi
echo "  ✅ 架构: $ARCH"

if [ "$(id -u)" != "0" ]; then
    echo "  ❌ 需要 root 权限，请用: sudo bash deploy_setup.sh"
    exit 1
fi
echo "  ✅ 以 root 运行"

# 检查源文件都在
for f in libk4a.so.1.4.2 libk4arecord.so.1.4.2 libdepthengine.so.2.0; do
    if [ ! -f "$HERE/libs/$f" ]; then
        echo "  ❌ 缺少文件: libs/$f"
        exit 1
    fi
done
echo "  ✅ 三个库文件都在"

# ---------- 安装库 ----------
echo ""
echo "[2/5] 安装库到 $LIBDIR"
cp "$HERE/libs/libk4a.so.1.4.2"       "$LIBDIR/"
cp "$HERE/libs/libk4arecord.so.1.4.2" "$LIBDIR/"
cp "$HERE/libs/libdepthengine.so.2.0" "$LIBDIR/"
echo "  ✅ 三个 .so 已复制"

# ---------- 建软链接 ----------
echo ""
echo "[3/5] 建立软链接"
cd "$LIBDIR"
ln -sf libk4a.so.1.4.2       libk4a.so.1.4
ln -sf libk4a.so.1.4.2       libk4a.so
ln -sf libk4arecord.so.1.4.2 libk4arecord.so.1.4
ln -sf libk4arecord.so.1.4.2 libk4arecord.so
echo "  ✅ libk4a:       libk4a.so -> libk4a.so.1.4 -> libk4a.so.1.4.2"
echo "  ✅ libk4arecord: libk4arecord.so -> libk4arecord.so.1.4 -> ...1.4.2"

# 深度引擎额外放一份到 /lib（有些加载路径只找这里）
ln -sf "$LIBDIR/libdepthengine.so.2.0" /lib/aarch64-linux-gnu/libdepthengine.so.2.0
echo "  ✅ libdepthengine.so.2.0 额外链接到 /lib/aarch64-linux-gnu/"

# ---------- 更新库缓存 ----------
echo ""
echo "[4/5] 更新动态库缓存"
ldconfig
echo "  ✅ ldconfig 完成"

# ---------- 装头文件 ----------
echo ""
echo "[5/5] 安装头文件到 $INCDIR/k4a"
if [ -d "$HERE/include/k4a" ]; then
    mkdir -p "$INCDIR/k4a"
    cp "$HERE/include/k4a/"*.h   "$INCDIR/k4a/" 2>/dev/null || true
    cp "$HERE/include/k4a/"*.hpp "$INCDIR/k4a/" 2>/dev/null || true
    echo "  ✅ 头文件: $(ls "$INCDIR/k4a" | wc -l) 个"
else
    echo "  ⚠️  没找到 include/k4a，跳过（不影响 Python 使用）"
fi

echo ""
echo "=============================================="
echo " 安装完成！"
echo "=============================================="
echo ""
echo "下一步：跑验证脚本，确认库能不能用"
echo ""
echo "    python3 verify_libs.py"
echo ""
