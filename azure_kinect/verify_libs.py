#!/usr/bin/env python3
"""
验证脚本 —— 检查 k4a 库装对了没、能不能加载

这个脚本专门为"排查问题"设计：每一步都告诉你
  · 在检查什么
  · 正常应该是什么样
  · 不对的话怎么修

用法（在 U 盘目录里）：
    python3 verify_libs.py

不需要 root 权限。
"""
import ctypes
import os
import subprocess
import sys

LIBDIR = "/usr/lib/aarch64-linux-gnu"
LIBS = ["libk4a.so.1.4.2", "libk4arecord.so.1.4.2", "libdepthengine.so.2.0"]
SONAMES = {"libk4a.so.1.4.2": "libk4a.so.1.4",
           "libk4arecord.so.1.4.2": "libk4arecord.so.1.4",
           "libdepthengine.so.2.0": "libdepthengine.so.2.0"}

PASS, FAIL, WARN = "✅", "❌", "⚠️ "
problems = []


def hdr(n, title):
    print(f"\n{'─' * 62}")
    print(f"[{n}] {title}")
    print(f"{'─' * 62}")


def main():
    print("=" * 62)
    print(" Azure Kinect ARM64 库 —— 安装验证")
    print("=" * 62)

    # ---------- 1. 架构 ----------
    hdr(1, "架构检查")
    arch = os.uname().machine
    if arch == "aarch64":
        print(f"  {PASS} 架构 {arch}（Jetson 正确）")
    else:
        print(f"  {FAIL} 架构是 {arch}，不是 aarch64")
        print("     这个包只能在 Jetson / ARM64 上用")
        problems.append("架构不对")
        return 1

    # ---------- 2. 文件是否存在 ----------
    hdr(2, "库文件是否存在")
    for f in LIBS:
        p = os.path.join(LIBDIR, f)
        if os.path.isfile(p):
            mb = os.path.getsize(p) / 1048576
            print(f"  {PASS} {f:<28} {mb:5.1f} MB")
        else:
            print(f"  {FAIL} {f:<28} 缺失")
            print(f"     修复: sudo bash deploy_setup.sh")
            problems.append(f"{f} 缺失")

    # ---------- 3. 软链接 ----------
    hdr(3, "软链接是否正确")
    for real, link in SONAMES.items():
        linkp = os.path.join(LIBDIR, link)
        if os.path.islink(linkp):
            tgt = os.readlink(linkp)
            ok = os.path.basename(tgt) == real or tgt.endswith(real)
            print(f"  {PASS if ok else WARN} {link:<28} -> {tgt}")
        elif os.path.isfile(linkp):
            print(f"  {PASS} {link:<28} 是实体文件")
        else:
            print(f"  {FAIL} {link:<28} 不存在")
            print(f"     修复: cd {LIBDIR} && sudo ln -sf {real} {link}")
            problems.append(f"软链接 {link} 缺失")

    # ---------- 4. 依赖是否齐全（ldd）----------
    hdr(4, "依赖库是否齐全（这一步最容易出问题）")
    for f in ["libk4a.so.1.4.2", "libdepthengine.so.2.0"]:
        p = os.path.join(LIBDIR, f)
        if not os.path.isfile(p):
            continue
        print(f"\n  --- {f} ---")
        try:
            out = subprocess.run(["ldd", p], stdout=subprocess.PIPE, universal_newlines=True,
                                 timeout=20).stdout
        except Exception as e:
            print(f"  {WARN} ldd 跑不了: {e}")
            continue
        missing = [l.strip() for l in out.splitlines() if "not found" in l]
        if missing:
            print(f"  {FAIL} 缺少 {len(missing)} 个依赖:")
            for m in missing:
                print(f"       {m}")
            problems.append(f"{f} 缺依赖: {missing}")
        else:
            n = len([l for l in out.splitlines() if "=>" in l])
            print(f"  {PASS} 依赖齐全（{n} 个）")

    # ---------- 5. 实际加载（最关键）----------
    hdr(5, "实际加载测试（最关键）")
    for soname in ["libk4a.so.1.4", "libk4arecord.so.1.4"]:
        try:
            ctypes.CDLL(soname)
            print(f"  {PASS} {soname} 加载成功")
        except OSError as e:
            print(f"  {FAIL} {soname} 加载失败")
            print(f"     {e}")
            problems.append(f"{soname} 加载失败")

    # 深度引擎单独测（它是闭源的，最容易出问题）
    try:
        ctypes.CDLL("libdepthengine.so.2.0")
        print(f"  {PASS} libdepthengine.so.2.0 加载成功  ← 深度功能靠它")
    except OSError as e:
        print(f"  {FAIL} libdepthengine.so.2.0 加载失败  ← 深度会不可用")
        print(f"     {e}")
        print("     常见原因: 缺 libGL.so.1 或 libX11.so.6")
        print("     修复: sudo apt install libgl1-mesa-glx libx11-6")
        print("           （没网的话需要离线包）")
        problems.append("深度引擎加载失败")

    # ---------- 6. k4a API 是否可用 ----------
    hdr(6, "k4a API 符号检查")
    try:
        k4a = ctypes.CDLL("libk4a.so.1.4")
        need = ["k4a_device_get_installed_count", "k4a_device_open",
                "k4a_device_start_cameras", "k4a_device_get_capture",
                "k4a_calibration_3d_to_3d", "k4a_calibration_2d_to_3d",
                "k4a_calibration_color_2d_to_depth_2d",
                "k4a_transformation_depth_image_to_point_cloud"]
        ok = 0
        for s in need:
            if hasattr(k4a, s):
                ok += 1
            else:
                print(f"  {WARN} 缺符号 {s}")
        print(f"  {PASS if ok == len(need) else WARN} "
              f"{ok}/{len(need)} 个关键 API 可用")
        if ok < len(need):
            problems.append("部分 API 符号缺失")
    except OSError:
        print(f"  {FAIL} 无法打开 libk4a")
        problems.append("libk4a 打不开")

    # ---------- 7. 设备检查 ----------
    hdr(7, "设备连接检查")
    try:
        k4a = ctypes.CDLL("libk4a.so.1.4")
        k4a.k4a_device_get_installed_count.restype = ctypes.c_uint32
        n = k4a.k4a_device_get_installed_count()
        if n > 0:
            print(f"  {PASS} 检测到 {n} 个 Azure Kinect 设备")
        else:
            print(f"  {WARN} 没检测到设备")
            print("     如果 Kinect 没插，这是正常的")
            print("     如果插了但读不到，跑: lsusb | grep 045e")
    except Exception as e:
        print(f"  {WARN} 检测失败: {e}")

    # ---------- 总结 ----------
    print("\n" + "=" * 62)
    if not problems:
        print(" 🎉 全部通过！库装好了，可以开始用相机了")
        print("=" * 62)
        print("\n下一步：确认相机能出彩色和深度")
        print("     python3 code/k4a_smoke.py 30")
        return 0
    else:
        print(f" ⚠️  发现 {len(problems)} 个问题：")
        for p in problems:
            print(f"     · {p}")
        print("=" * 62)
        print("\n按上面的提示修复，然后重新跑本脚本")
        return 1


if __name__ == "__main__":
    sys.exit(main())
