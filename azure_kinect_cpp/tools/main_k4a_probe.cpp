// k4a_probe_cpp — port of azure_kinect/k4a_probe.py.
// Checks device count, open, serial number, and whether a depth stream really
// starts.  Uses the C API directly so it works even when libk4a.hpp is absent.

#include <cstdio>

#if KVC_HAVE_K4A

#include <k4a/k4a.h>

#include <cstring>

int main() {
    std::printf("============================================================\n");
    std::printf(" Azure Kinect 体检\n");
    std::printf("============================================================\n");

    const uint32_t n = k4a_device_get_installed_count();
    std::printf("① 认到的设备数: %u\n", n);
    if (n == 0) {
        std::printf("   ❌ 一个都没有 —— 检查 USB 线（要 USB3.0 线）和供电\n");
        std::printf("      板子上看: lsusb | grep 045e\n");
        return 1;
    }

    k4a_device_t dev = nullptr;
    k4a_result_t r = k4a_device_open(0, &dev);
    std::printf("② k4a_device_open -> %d %s\n", static_cast<int>(r),
                r == K4A_RESULT_SUCCEEDED ? "✅ 成功" : "❌ 失败");
    if (r != K4A_RESULT_SUCCEEDED) {
        std::printf("\n   失败的常见原因：\n");
        std::printf("     a) 缺 udev 规则 → 先跑 sudo bash "
                    "~/work/azure_kinect/k4a_udev_setup.sh\n");
        std::printf("     b) 被别的进程占用 → sudo lsof /dev/bus/usb/002/* | head\n");
        std::printf("     c) 供电不足 → Kinect 需要 12V 独立电源；\n");
        std::printf("        光靠 Jetson 的 USB 口带不动（尤其板子还是 micro-USB 供电时）\n");
        return 1;
    }

    char serial[256] = {0};
    size_t sz = sizeof(serial);
    const k4a_buffer_result_t br =
        k4a_device_get_serialnum(dev, serial, &sz);
    if (br == K4A_BUFFER_RESULT_SUCCEEDED) {
        std::printf("③ 序列号: %s\n", serial);
    } else {
        std::printf("③ 序列号: 读取失败(%d)\n", static_cast<int>(br));
    }

    // ④ 试着真开一路深度流 —— 这一步才证明相机是好的
    k4a_device_configuration_t cfg = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    cfg.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
    cfg.color_resolution = K4A_COLOR_RESOLUTION_720P;
    cfg.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;  // 640x576
    cfg.camera_fps = K4A_FRAMES_PER_SECOND_15;
    cfg.synchronized_images_only = 0;
    cfg.depth_delay_off_color_usec = 0;
    cfg.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
    cfg.subordinate_delay_off_master_usec = 0;
    cfg.disable_streaming_indicator = 0;

    r = k4a_device_start_cameras(dev, &cfg);
    std::printf("④ k4a_device_start_cameras -> %d %s\n", static_cast<int>(r),
                r == K4A_RESULT_SUCCEEDED ? "✅ 深度流开起来了！" : "❌ 开不了");
    if (r == K4A_RESULT_SUCCEEDED) {
        k4a_capture_t cap = nullptr;
        bool got = false;
        for (int i = 0; i < 50; ++i) {  // 最多等 5 秒
            if (k4a_device_get_capture(dev, &cap, 100) == K4A_WAIT_RESULT_SUCCEEDED) {
                got = true;
                break;
            }
        }
        if (got) {
            std::printf("   ✅ 拿到了第一帧 capture —— 相机完全正常\n");
            k4a_capture_release(cap);
        } else {
            std::printf("   ⚠️ 流开了但拿不到帧 —— 多半是供电/带宽问题\n");
        }
        k4a_device_stop_cameras(dev);
    } else {
        std::printf("\n   开不了流的原因：\n");
        std::printf("     · 没有 X 显示（深度引擎要 GLX）→ 用 run_with_x.sh\n");
        std::printf("     · 供电不足（最常见）—— Kinect 深度引擎要 12V 独立电源\n");
        std::printf("     · USB 不是 3.0 / 线材不行\n");
    }

    k4a_device_close(dev);
    std::printf("============================================================\n");
    return 0;
}

#else

int main() {
    std::printf("k4a_probe_cpp: 这个二进制没有编入 Azure Kinect SDK "
                "（KVC_WITH_K4A=0）\n");
    std::printf("  在 Jetson 上重新构建：bash "
                "~/work/azure_kinect_cpp/scripts/build_board.sh\n");
    return 1;
}

#endif
