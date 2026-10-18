// check_gravity_cpp — port of azure_kinect/check_gravity.py.
// Reads the accelerometer for 3 s and reports which axis points up in the
// depth camera frame.  Opens the device but does NOT start the cameras, so it
// works without a display; grab the depth engine separately if you need it.

#include <cstdio>
#include <cstdint>

#if KVC_HAVE_K4A

#include <k4a/k4a.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace {

struct Stats {
    double mean[3] = {0, 0, 0};
    double std[3] = {0, 0, 0};
};

}  // namespace

int main() {
    std::printf("k4a_imu_sample_t 大小 = %zu 字节（C 里是 48）\n",
                sizeof(k4a_imu_sample_t));

    const uint32_t n = k4a_device_get_installed_count();
    if (n == 0) {
        std::printf("❌ 没认到 Azure Kinect（检查 USB 3.0 线和 12V 电源）\n");
        return 1;
    }
    k4a_device_t dev = nullptr;
    if (k4a_device_open(0, &dev) != K4A_RESULT_SUCCEEDED) {
        std::printf("❌ k4a_device_open 失败（权限 / 占用 / 供电）\n");
        return 1;
    }

    k4a_calibration_t calib;
    bool has_calib =
        k4a_device_get_calibration(dev, K4A_DEPTH_MODE_NFOV_UNBINNED,
                                   K4A_COLOR_RESOLUTION_720P,
                                   &calib) == K4A_RESULT_SUCCEEDED;
    if (!has_calib) {
        std::printf("⚠️ 取标定失败 —— 只能给加速度计原始系的方向\n");
    }

    // SDK 1.4.2 refuses start_imu while neither camera stream is running
    // ("k4a_device_start_imu called while the color/depth camera is not
    // running is not supported").  The Python tool dodges this because
    // AzureSource starts both streams in its constructor.  Start the COLOR
    // stream here: it does not need the depth engine / GLX, so this tool
    // still runs without a DISPLAY.
    k4a_device_configuration_t cfg = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    cfg.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
    cfg.color_resolution = K4A_COLOR_RESOLUTION_720P;
    cfg.camera_fps = K4A_FRAMES_PER_SECOND_30;
    const bool color_ok =
        k4a_device_start_cameras(dev, &cfg) == K4A_RESULT_SUCCEEDED;
    if (!color_ok) {
        std::printf("⚠️ 起彩色流失败 —— 直接试 IMU\n");
    }

    if (k4a_device_start_imu(dev) != K4A_RESULT_SUCCEEDED) {
        std::printf("❌ 启动 IMU 失败（这台设备的 IMU 可能不可用）\n");
        if (color_ok) k4a_device_stop_cameras(dev);
        k4a_device_close(dev);
        return 1;
    }

    std::vector<k4a_imu_sample_t> samples;
    samples.reserve(400);
    std::printf("采样 3 秒（相机保持不动）…\n");
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count() < 3.0) {
        k4a_imu_sample_t s;
        if (k4a_device_get_imu_sample(dev, &s, 200) == K4A_WAIT_RESULT_SUCCEEDED) {
            samples.push_back(s);
        }
    }
    k4a_device_stop_imu(dev);
    if (color_ok) k4a_device_stop_cameras(dev);
    k4a_device_close(dev);

    if (samples.empty()) {
        std::printf("❌ 一个 IMU 样本都没拿到\n");
        return 1;
    }

    Stats acc;
    Stats gyr;
    for (const auto& s : samples) {
        acc.mean[0] += s.acc_sample.xyz.x;
        acc.mean[1] += s.acc_sample.xyz.y;
        acc.mean[2] += s.acc_sample.xyz.z;
        gyr.mean[0] += s.gyro_sample.xyz.x;
        gyr.mean[1] += s.gyro_sample.xyz.y;
        gyr.mean[2] += s.gyro_sample.xyz.z;
    }
    const double ns = static_cast<double>(samples.size());
    for (int i = 0; i < 3; ++i) {
        acc.mean[i] /= ns;
        gyr.mean[i] /= ns;
    }
    for (const auto& s : samples) {
        const double a[3] = {s.acc_sample.xyz.x, s.acc_sample.xyz.y,
                             s.acc_sample.xyz.z};
        for (int i = 0; i < 3; ++i) {
            const double d = a[i] - acc.mean[i];
            acc.std[i] += d * d;
        }
    }
    for (int i = 0; i < 3; ++i) acc.std[i] = std::sqrt(acc.std[i] / ns);

    const double mag = std::sqrt(acc.mean[0] * acc.mean[0] +
                                 acc.mean[1] * acc.mean[1] +
                                 acc.mean[2] * acc.mean[2]);

    std::printf("\n");
    std::printf("======================================================================\n");
    std::printf(" 加速度计样本 %zu 个，抖动 ±(%.3f, %.3f, %.3f)\n", samples.size(),
                acc.std[0], acc.std[1], acc.std[2]);
    std::printf("======================================================================\n");
    std::printf(" 平均加速度 (ax, ay, az) = (%+.3f, %+.3f, %+.3f) m/s²\n",
                acc.mean[0], acc.mean[1], acc.mean[2]);
    std::printf(" 模长 = %.2f m/s²   （重力约 9.81；接近说明读数是重力）\n",
                mag);
    double up[3] = {acc.mean[0], acc.mean[1], acc.mean[2]};
    if (mag > 1e-6) {
        up[0] /= mag;
        up[1] /= mag;
        up[2] /= mag;
        std::printf(" 单位化 = (%+.3f, %+.3f, %+.3f)\n", up[0], up[1], up[2]);
    }
    std::printf(" ↑ 这是在【加速度计自己的坐标系】里 —— IMU 和深度相机不是同一个系，\n");
    std::printf("   必须用标定里的外参转过去，否则会得出错误结论。\n");

    double gv[3] = {acc.mean[0], acc.mean[1], acc.mean[2]};
    if (has_calib) {
        const k4a_calibration_extrinsics_t e =
            calib.extrinsics[K4A_CALIBRATION_TYPE_ACCEL]
                            [K4A_CALIBRATION_TYPE_DEPTH];
        double R[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) R[i][j] = e.rotation[i * 3 + j];
        }
        const double tx = e.translation[0] / 1000.0;
        const double ty = e.translation[1] / 1000.0;
        const double tz = e.translation[2] / 1000.0;
        std::printf("\n");
        std::printf(" 加速度计→深度 的外参 |t| = %.1f mm\n",
                    std::sqrt(tx * tx + ty * ty + tz * tz) * 1000.0);
        std::printf(" R = \n");
        for (int i = 0; i < 3; ++i) {
            std::printf("   [%+.4f %+.4f %+.4f]\n", R[i][0], R[i][1], R[i][2]);
        }
        for (int i = 0; i < 3; ++i) {
            gv[i] = R[i][0] * acc.mean[0] + R[i][1] * acc.mean[1] +
                    R[i][2] * acc.mean[2];
        }
        const double gn =
            std::sqrt(gv[0] * gv[0] + gv[1] * gv[1] + gv[2] * gv[2]);
        std::printf("\n");
        std::printf(" ★★ 转到【深度相机系】后的重力向量 = "
                    "(%+.3f, %+.3f, %+.3f) m/s²\n",
                    gv[0], gv[1], gv[2]);
        std::printf("    单位化 = (%+.3f, %+.3f, %+.3f)\n", gv[0] / gn,
                    gv[1] / gn, gv[2] / gn);
    }
    std::printf("\n");
    std::printf(" ★ 相机【静止】时，加速度计读到的主要就是重力（指向地面）\n");
    std::printf("   所以这个向量的方向 = 相机坐标系里【向下】的方向\n");
    std::printf("\n");
    std::printf(" ══ 判读（以深度相机系为准；加速度计静止时读的是支撑力，\n");
    std::printf("     也就是指向【上】的方向）══\n");
    const double gnorm =
        std::sqrt(gv[0] * gv[0] + gv[1] * gv[1] + gv[2] * gv[2]);
    const char axis_names[3] = {'X', 'Y', 'Z'};
    for (int i = 0; i < 3; ++i) {
        const bool vertical = std::fabs(gv[i]) > 0.8 * gnorm;
        std::printf("    %c = %+8.3f m/s²%s\n", axis_names[i], gv[i],
                    vertical ? "  ← 竖直方向" : "");
    }
    int big = 0;
    for (int i = 1; i < 3; ++i) {
        if (std::fabs(gv[i]) > std::fabs(gv[big])) big = i;
    }
    std::printf("\n");
    std::printf(" ★ 重力主要落在 %c%c 轴上\n",
                gv[big] > 0 ? '+' : '-', axis_names[big]);
    if (big == 1) {
        if (gv[1] > 0) {
            std::printf("   → +Y 指向【上】→ 深度系的 +Y 是【向上】的\n");
            std::printf("     这【不是】标准光学系（标准是 +Y 向下），\n");
            std::printf("     说明相机是【倒装】的：加 --rotate 180 把它转正。\n");
        } else {
            std::printf("   → −Y 指向【上】，也就是 +Y 指向【下】✅\n");
            std::printf("     这就是标准光学系约定（+X 右 / +Y 下 / +Z 前），\n");
            std::printf("     我们的代码是对的，Y 为正 = 目标在相机下方。\n");
        }
    } else if (big == 2) {
        std::printf("   → 光轴（±Z）是【竖直】的！相机在俯视或仰视。\n");
        std::printf("     这时 +Y 是【水平】方向，报告里的 Y 不代表'高度差'。\n");
        if (gv[2] > 0) {
            std::printf("     +Z 向上 → 相机【朝上】看\n");
        } else {
            std::printf("     −Z 向下 → 相机【朝下】看（俯视桌面）\n");
        }
    } else {
        std::printf("   → %c 轴是竖直的 → 相机是【侧装】的，\n", axis_names[big]);
        std::printf("     +Y 不是竖直方向，报告里的 Y 不代表'高度差'。\n");
    }
    std::printf("\n 陀螺仪均值 = (%.3f, %.3f, %.3f)（静止时应接近 0）\n",
                gyr.mean[0], gyr.mean[1], gyr.mean[2]);
    std::printf("======================================================================\n");
    return 0;
}

#else

#include <cstdio>

int main() {
    std::printf("check_gravity_cpp: 这个二进制没有编入 Azure Kinect SDK "
                "（KVC_WITH_K4A=0）\n"
                "  在 Jetson 上重新构建：bash "
                "~/work/azure_kinect_cpp/scripts/build_board.sh\n");
    return 1;
}

#endif
