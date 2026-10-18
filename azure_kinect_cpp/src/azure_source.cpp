// Azure Kinect capture backend.  Direct port of azure_kinect/azure_capture.py;
// see that file for the derivations behind the rotation/extrinsics formulas.

#include "kvc/azure_source.hpp"

#include "kvc/math.hpp"

// The vendored export header was generated on Windows; on Linux the import
// attribute is meaningless and __declspec is a syntax error.
#ifndef K4A_STATIC_DEFINE
#define K4A_STATIC_DEFINE
#endif
#include <k4a/k4a.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>

namespace kvc {
namespace {

constexpr double kMinAccel = 8.5;
constexpr double kMaxAccel = 11.0;

std::string fixed4(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.4f", v);
    return buf;
}

int rot90_code(int k) {
    switch (k) {
        case 1: return cv::ROTATE_90_CLOCKWISE;
        case 2: return cv::ROTATE_180;
        default: return cv::ROTATE_90_COUNTERCLOCKWISE;
    }
}

void rotate_like_python(cv::Mat& m, int k) {
    if (!k || m.empty()) return;
    cv::Mat out;
    cv::rotate(m, out, rot90_code(k));
    m = out;
}

}  // namespace

struct AzureSource::Impl {
    k4a::device device;
    k4a::calibration calib;
    bool has_calib = false;
    std::optional<k4a::transformation> trans;
    bool started = false;

    ~Impl() {
        if (started) {
            try {
                device.stop_imu();
            } catch (...) {
            }
            try {
                device.stop_cameras();
            } catch (...) {
            }
        }
    }
};

AzureSource::AzureSource(const SourceConfig& cfg,
                         std::optional<BodyCalib> body_calib)
    : impl_(new Impl()), cfg_(cfg) {
    verbose_ = cfg.verbose;
    color_enabled_ = cfg.color;
    use_sdk_points_ = cfg.use_sdk_points;
    rotate_ = ((cfg.rotate % 360) + 360) % 360;

    const uint32_t n = k4a::device::get_installed_count();
    if (n == 0) {
        throw CaptureError(
            "没认到 Azure Kinect。\n"
            "  ① 插好 USB 3.0 线  ② 必须接自带 12V 电源\n"
            "  ③ sudo bash ~/work/azure_kinect/k4a_udev_setup.sh\n"
            "  ④ 无头环境还要有 X：bash ~/work/azure_kinect/run_with_x.sh");
    }
    try {
        impl_->device = k4a::device::open(static_cast<uint32_t>(cfg.device_index));
    } catch (const k4a::error& e) {
        throw CaptureError(
            std::string("k4a_device_open 失败：") + e.what() + "\n"
            "  · 权限：跑 99-k4a.rules\n"
            "  · 占用：sudo lsof /dev/bus/usb/002/*\n"
            "  · 供电：Kinect 需要 12V 独立电源");
    }
    try {
        serial_ = impl_->device.get_serialnum();
    } catch (...) {
        serial_ = "?";
    }

    k4a_device_configuration_t kcfg = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    kcfg.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
    kcfg.color_resolution =
        color_enabled_ ? static_cast<k4a_color_resolution_t>(cfg.color_resolution)
                       : K4A_COLOR_RESOLUTION_OFF;
    kcfg.depth_mode = static_cast<k4a_depth_mode_t>(cfg.depth_mode);
    kcfg.camera_fps = static_cast<k4a_fps_t>(cfg.fps);
    kcfg.synchronized_images_only = color_enabled_ ? 1 : 0;
    kcfg.depth_delay_off_color_usec = 0;
    kcfg.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
    kcfg.subordinate_delay_off_master_usec = 0;
    kcfg.disable_streaming_indicator = 0;

    // Calibration first: intrinsics and depth->color extrinsics.
    try {
        impl_->calib = impl_->device.get_calibration(
            kcfg.depth_mode, kcfg.color_resolution);
        impl_->has_calib = true;
    } catch (const k4a::error& e) {
        // ★ 标定是深度/彩色内参、depth→color 外参、以及无车体标定时
        //   重力对齐的唯一来源。继续跑只会让针孔兜底拿 fx=0 去除零，
        //   产出 inf/NaN 点云；而且 valid 只看 Z，垃圾点会被静默标成有效。
        //   Python 版这里同样是直接失败，不降级。
        throw CaptureError(
            std::string("取不到相机标定（get_calibration 失败）：") + e.what() +
            "\n  ★ 标定是内参/外参的唯一来源，没有它建不出点云。"
            "\n  · 相机没插好 / 供电不足 / 设备被占用都可能引起"
            "\n  · 检查相机连接与供电后重试");
    }

    if (impl_->has_calib) {
        const auto& d = impl_->calib.depth_camera_calibration;
        const auto& c = impl_->calib.color_camera_calibration;
        const auto& dp = d.intrinsics.parameters.param;
        const auto& cp = c.intrinsics.parameters.param;
        intr_depth_ = Intrinsics{dp.fx, dp.fy, dp.cx, dp.cy,
                                 d.resolution_width, d.resolution_height,
                                 "brown_conrady"};
        // ★ 最后一道防线：fx/fy 为 0 会在针孔兜底里除零，产出 inf/NaN。
        if (!(intr_depth_.fx > 0.0 && intr_depth_.fy > 0.0)) {
            throw CaptureError(
                "相机标定返回了无效深度内参（fx/fy ≤ 0）。"
                "\n  没有可靠内参就建不出点云；检查相机连接与供电后重试。");
        }
        intr_color_ = Intrinsics{cp.fx, cp.fy, cp.cx, cp.cy,
                                 c.resolution_width, c.resolution_height, ""};

        const k4a_calibration_extrinsics_t e =
            impl_->calib.extrinsics[K4A_CALIBRATION_TYPE_DEPTH]
                                   [K4A_CALIBRATION_TYPE_COLOR];
        Mat3 R;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R(i, j) = e.rotation[i * 3 + j];
            }
        }
        const Vec3 t(e.translation[0] / 1000.0, e.translation[1] / 1000.0,
                     e.translation[2] / 1000.0);
        extr_base_.R = R;
        extr_base_.t = t;

        // Point cloud and color are rotated by rot90; the projection
        // extrinsics must be conjugated the same way.  Computed ONCE.
        const Mat3 Rz = Rz_optical(static_cast<double>(rotate_));
        extr_.R = Rz * R * Rz.transpose();
        extr_.t = Rz * t;

        const int k = rotate_ / 90;
        if (k) {
            const double ccx = intr_color_.cx;
            const double ccy = intr_color_.cy;
            const double cw = intr_color_.width;
            const double ch = intr_color_.height;
            double pcx = 0.0, pcy = 0.0;
            if (k == 1) {
                pcx = ch - 1.0 - ccy;
                pcy = ccx;
            } else if (k == 2) {
                pcx = cw - 1.0 - ccx;
                pcy = ch - 1.0 - ccy;
            } else {
                pcx = ccy;
                pcy = cw - 1.0 - ccx;
            }
            px_fix_[0] = pcx - ccx;
            px_fix_[1] = pcy - ccy;
        }
    }

    // Start cameras, retrying like the Python version.
    bool started = false;
    for (int attempt = 0; attempt < 3; ++attempt) {
        try {
            impl_->device.start_cameras(&kcfg);
            started = true;
            break;
        } catch (const k4a::error&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    if (!started) {
        throw CaptureError(
            "k4a_device_start_cameras 失败。\n"
            "  ★ 最常见：没有 X 显示 → 深度引擎报 error 204\n"
            "     解决：bash ~/work/azure_kinect/run_with_x.sh\n"
            "  · 其次：供电不足（Kinect 要 12V 独立电源）");
    }
    impl_->started = true;

    if (impl_->has_calib) {
        try {
            impl_->trans.emplace(impl_->calib);
        } catch (const k4a::error& e) {
            if (verbose_) {
                std::cout << "[Kinect] ⚠️ 创建 transformation 失败："
                          << e.what() << "，点云退回针孔模型" << std::endl;
            }
        }
    }

    // Let auto-exposure settle, then discard three frames.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    for (int i = 0; i < 3; ++i) {
        k4a::capture cap;
        try {
            impl_->device.get_capture(&cap, std::chrono::milliseconds(2000));
        } catch (...) {
        }
    }

    if (body_calib.has_value()) {
        // Calibration is the physical truth: validate it and never read IMU.
        const Mat3 R = body_calib->R;
        const double res = upright_residual(R, static_cast<double>(rotate_));
        if (res >= 0.0) {
            throw CaptureError(
                "车体标定没通过『图像竖直』判据：\n"
                "  (R_cam_to_body @ Rz(" + std::to_string(rotate_) +
                ")ᵀ)[2,1] = " + (res >= 0 ? "+" : "") + fixed4(res) + " ≥ 0\n"
                "  含义：把图像翻滚还原后，『图像里的下』没有指向车体下方。\n"
                "  · 第二列 ≈ +1（rotate=180）→ 矩阵是按 post-Rz 系写的，\n"
                "    180° 会被重复计入，点云整个翻过去。\n"
                "  · 第三列不像光轴 → 多半是矩阵转置写反了。\n"
                "  自检： body_calib_cpp --check <文件>");
        }
        R_body_ = R;
        t_out_ = body_calib->t;
        calib_id_ = body_calib->id;
        has_body_ = true;
        gravity_ok_ = true;  // output is already physically aligned
    } else if (cfg.read_gravity) {
        read_gravity();
    }

    finish_rotations();

    if (verbose_) {
        std::cout << "[Kinect] 序列号 " << serial_ << "\n"
                  << "[Kinect] 深度内参 fx=" << intr_depth_.fx
                  << " fy=" << intr_depth_.fy << " cx=" << intr_depth_.cx
                  << " cy=" << intr_depth_.cy << " " << intr_depth_.width << "x"
                  << intr_depth_.height << "\n"
                  << "[Kinect] 彩色内参 fx=" << intr_color_.fx
                  << " fy=" << intr_color_.fy << " cx=" << intr_color_.cx
                  << " cy=" << intr_color_.cy << " " << intr_color_.width << "x"
                  << intr_color_.height << "\n"
                  << "[Kinect] 深度→彩色 |t| = "
                  << extr_base_.t.norm() * 1000.0 << " mm";
        if (has_body_) {
            std::cout << "\n[Kinect] 车体标定 '" << calib_id_ << "' 已装载："
                      << "光心在车体系 (" << t_out_.x() << ", " << t_out_.y()
                      << ", " << t_out_.z() << ") m";
        }
        std::cout << "\n[Kinect] 点云输出 R_out（车体系 X前/Y左/Z上）"
                  << (xform_needed_ ? "已启用" : "= 单位阵")
                  << (t_out_.cwiseAbs().maxCoeff() > 1e-9 ? "，含平移" : "")
                  << std::endl;
    }
}

AzureSource::~AzureSource() = default;

void AzureSource::read_gravity() {
    GravityInfo info;
    if (!impl_->has_calib) {
        info.err = "没有标定 → 没法把 IMU 系转到深度系";
        grav_info_ = info;
        if (verbose_) std::cout << "[Kinect] ⚠️ 重力对齐【跳过】："
                                << info.err << std::endl;
        return;
    }
    try {
        impl_->device.start_imu();
    } catch (const k4a::error& e) {
        info.err = std::string("k4a_device_start_imu 失败：") + e.what();
        grav_info_ = info;
        if (verbose_) std::cout << "[Kinect] ⚠️ 重力对齐【跳过】："
                                << info.err << std::endl;
        return;
    }

    std::vector<Vec3> accs;
    std::vector<Vec3> gyros;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count() < 1.5) {
        k4a_imu_sample_t s;
        try {
            if (impl_->device.get_imu_sample(&s, std::chrono::milliseconds(200))) {
                accs.emplace_back(s.acc_sample.xyz.x, s.acc_sample.xyz.y,
                                  s.acc_sample.xyz.z);
                gyros.emplace_back(s.gyro_sample.xyz.x, s.gyro_sample.xyz.y,
                                   s.gyro_sample.xyz.z);
            }
        } catch (const k4a::error& e) {
            info.err = std::string("读 IMU 出错：") + e.what();
            break;
        }
    }
    try {
        impl_->device.stop_imu();
    } catch (...) {
    }

    if (accs.size() < 50) {
        if (info.err.empty()) {
            info.err = "只拿到 " + std::to_string(accs.size()) + " 个 IMU 样本";
        }
        grav_info_ = info;
        if (verbose_) std::cout << "[Kinect] ⚠️ 重力对齐【跳过】："
                                << info.err << std::endl;
        return;
    }

    Vec3 mean = Vec3::Zero();
    for (const auto& a : accs) mean += a;
    mean /= static_cast<double>(accs.size());
    const double mag = mean.norm();

    Vec3 gmean = Vec3::Zero();
    for (const auto& g : gyros) gmean += g;
    if (!gyros.empty()) gmean /= static_cast<double>(gyros.size());

    // max over axes of the per-axis stddev
    Vec3 var = Vec3::Zero();
    for (const auto& a : accs) var += (a - mean).cwiseAbs2();
    var /= static_cast<double>(accs.size());
    const double jitter = std::sqrt(var.maxCoeff());

    info.n_samples = static_cast<int>(accs.size());
    info.accel_norm = mag;
    info.accel_jitter = jitter;
    info.gyro_norm = gmean.norm();

    if (mag < kMinAccel || mag > kMaxAccel) {
        info.err = "加速度模长 " + std::to_string(mag) +
                   " m/s² 不像重力（静置应≈9.8）→ 相机可能在动";
        grav_info_ = info;
        if (verbose_) std::cout << "[Kinect] ⚠️ 重力对齐【跳过】："
                                << info.err << std::endl;
        return;
    }

    const k4a_calibration_extrinsics_t e =
        impl_->calib.extrinsics[K4A_CALIBRATION_TYPE_ACCEL]
                               [K4A_CALIBRATION_TYPE_DEPTH];
    Mat3 R;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) R(i, j) = e.rotation[i * 3 + j];
    }
    Vec3 up = R * (mean / mag);
    const double n = up.norm();
    if (n < 0.5) {
        info.err = "accel→depth 外参退化";
        grav_info_ = info;
        if (verbose_) std::cout << "[Kinect] ⚠️ 重力对齐【跳过】："
                                << info.err << std::endl;
        return;
    }
    up /= n;

    const Mat3 Rz = Rz_optical(static_cast<double>(rotate_));
    const Vec3 up_out = Rz * up;
    R_grav_ = rotation_between(up_out, Vec3(0.0, -1.0, 0.0));
    has_grav_ = true;
    gravity_ok_ = true;
    info.ok = true;
    info.tilt_deg =
        std::acos(std::max(-1.0, std::min(1.0, -up_out.y()))) * 180.0 / M_PI;
    grav_info_ = info;

    if (verbose_) {
        std::cout << "[Kinect] 重力对齐：IMU " << info.n_samples << " 样本，|g|="
                  << info.accel_norm << " m/s²，抖动 " << info.accel_jitter * 100
                  << " cm/s²\n"
                  << "[Kinect] 相机相对竖直偏 " << info.tilt_deg << "°"
                  << (info.tilt_deg > 0.15 ? "（已把点云转正，Y=真实高度差）"
                                           : "（基本水平）")
                  << std::endl;
        if (info.gyro_norm > 0.15) {
            std::cout << "[Kinect]   ⚠️ 采样时陀螺仪 " << info.gyro_norm
                      << " rad/s ≠ 0 —— 相机在动，这次对齐不可信，摆稳后重启"
                      << std::endl;
        }
    }
}

std::optional<Vec3> AzureSource::read_up_depth(std::string* err) {
    auto fail = [&](const std::string& why) -> std::optional<Vec3> {
        if (err) *err = why;
        return std::nullopt;
    };
    if (!impl_->has_calib) return fail("没有标定 → 没法把 IMU 系转到深度系");
    try {
        impl_->device.start_imu();
    } catch (const k4a::error& e) {
        return fail(std::string("k4a_device_start_imu 失败：") + e.what());
    }

    std::vector<Vec3> accs;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count() < 1.5) {
        k4a_imu_sample_t s;
        try {
            if (impl_->device.get_imu_sample(&s, std::chrono::milliseconds(200))) {
                accs.emplace_back(s.acc_sample.xyz.x, s.acc_sample.xyz.y,
                                  s.acc_sample.xyz.z);
            }
        } catch (const k4a::error& e) {
            try {
                impl_->device.stop_imu();
            } catch (...) {
            }
            return fail(std::string("读 IMU 出错：") + e.what());
        }
    }
    try {
        impl_->device.stop_imu();
    } catch (...) {
    }

    if (accs.size() < 50) {
        return fail("只拿到 " + std::to_string(accs.size()) + " 个 IMU 样本");
    }
    Vec3 mean = Vec3::Zero();
    for (const auto& a : accs) mean += a;
    mean /= static_cast<double>(accs.size());
    const double mag = mean.norm();
    if (mag < 8.5 || mag > 11.0) {
        return fail("加速度模长 " + std::to_string(mag) +
                    " m/s² 不像重力（静置应≈9.8）→ 相机可能在动");
    }
    const k4a_calibration_extrinsics_t e =
        impl_->calib.extrinsics[K4A_CALIBRATION_TYPE_ACCEL]
                               [K4A_CALIBRATION_TYPE_DEPTH];
    Mat3 R;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) R(i, j) = e.rotation[i * 3 + j];
    }
    Vec3 up = R * (mean / mag);
    const double n = up.norm();
    if (n < 0.5) return fail("accel→depth 外参退化");
    return up / n;
}

void AzureSource::finish_rotations() {    const Mat3 Rz = Rz_optical(static_cast<double>(rotate_));
    if (has_body_) {
        // The calibration matrix is raw-depth -> body and already contains
        // the 180° inversion; multiplying by Rz again would double-count it.
        R_out_ = R_body_;
    } else {
        const Mat3 X = has_grav_ ? R_grav_ : Mat3::Identity();
        R_out_ = X * Rz;
    }
    xform_needed_ = (R_out_ - Mat3::Identity()).cwiseAbs().maxCoeff() > 1e-9 ||
                    t_out_.cwiseAbs().maxCoeff() > 1e-9;

    // ★ R_cam_/t_cam_ 只由 R_out_/t_out_ 决定，和有没有 SDK 标定无关。
    //   必须放在 early return 之前，否则没标定时 range/az/el 会整体算错。
    const Mat3 R_out_T = R_out_.transpose();
    R_cam_ = Rz * R_out_T;
    t_cam_ = -(R_cam_ * t_out_);

    if (!impl_->has_calib) return;
    const Mat3& R_dc = extr_base_.R;
    const Vec3& t_dc = extr_base_.t;
    extr_.R = Rz * R_dc * R_out_T;
    extr_.t = Rz * (t_dc - R_dc * R_out_T * t_out_);
}

std::optional<FrameData> AzureSource::get(int stride, int timeout_ms) {
    if (stride < 1) stride = 1;

    k4a::capture cap;
    try {
        if (!impl_->device.get_capture(&cap,
                                       std::chrono::milliseconds(timeout_ms))) {
            return std::nullopt;
        }
    } catch (const k4a::error& e) {
        throw CaptureError(std::string("k4a_device_get_capture 失败：") +
                           e.what());
    }

    k4a::image dimg = cap.get_depth_image();
    if (!dimg) return std::nullopt;
    const int w = dimg.get_width_pixels();
    const int h = dimg.get_height_pixels();
    const int dstride = dimg.get_stride_bytes();

    cv::Mat depth16(h, w, CV_16U);
    const uint8_t* dbuf = dimg.get_buffer();
    for (int y = 0; y < h; ++y) {
        std::memcpy(depth16.ptr<uint16_t>(y), dbuf + static_cast<size_t>(y) * dstride,
                    static_cast<size_t>(w) * sizeof(uint16_t));
    }
    cv::Mat depth_m;
    depth16.convertTo(depth_m, CV_32F, 0.001);

    // SDK point cloud (distortion-corrected) must be read before dimg dies.
    std::vector<int16_t> xyz_raw;
    int xyz_stride = 0;
    if (use_sdk_points_ && impl_->trans.has_value()) {
        try {
            k4a::image xyz =
                impl_->trans->depth_image_to_point_cloud(
                    dimg, K4A_CALIBRATION_TYPE_DEPTH);
            if (xyz) {
                xyz_stride = xyz.get_stride_bytes();
                const size_t bytes = static_cast<size_t>(xyz_stride) * h;
                xyz_raw.resize(bytes / sizeof(int16_t));
                std::memcpy(xyz_raw.data(), xyz.get_buffer(), bytes);
            }
        } catch (const k4a::error& e) {
            if (verbose_) {
                std::cout << "[Kinect] ⚠️ SDK 点云失败：" << e.what()
                          << "，本帧退回针孔模型" << std::endl;
            }
            xyz_raw.clear();
        }
    }

    FrameData f;
    f.depth_width = w;
    f.depth_height = h;
    f.intr_depth = intr_depth_;
    f.intr_depth.width = w;
    f.intr_depth.height = h;
    f.intr_color = intr_color_;

    // Color: MJPG -> BGR, then the same rot90 as the depth image.
    if (color_enabled_) {
        k4a::image cimg = cap.get_color_image();
        if (cimg) {
            const size_t csz = cimg.get_size();
            const cv::Mat jpg(1, static_cast<int>(csz), CV_8U,
                              const_cast<uint8_t*>(cimg.get_buffer()));
            f.color_bgr = cv::imdecode(jpg, cv::IMREAD_COLOR);
            rotate_like_python(f.color_bgr, rotate_ / 90);
        }
    }

    const int k = rotate_ / 90;
    // ★ depth_m 保持【未旋转】：下面的针孔兜底必须用原始栅格 + 原始内参
    //   建点，最后统一由 R_out_（已含 Rz）旋转一次 —— 和 SDK 点云路径一致。
    //   旋转后的图像只用于输出/可视化，放 f.depth_m（保持旧语义）。
    //   （2026-10-18 修：以前先 rotate 再建点再乘 R_out_，rotate≠0 时转两次。）
    f.depth_m = depth_m;                 // k==0 时共享；k!=0 时 rotate 会换新 Mat
    rotate_like_python(f.depth_m, k);

    if (!xyz_raw.empty()) {
        const size_t rows = static_cast<size_t>(h);
        const size_t cols = static_cast<size_t>(w);
        const size_t n = (rows / stride + 1) * (cols / stride + 1);
        f.points.reserve(n);
        f.valid.reserve(n);
        for (int y = 0; y < h; y += stride) {
            const int16_t* row =
                xyz_raw.data() + static_cast<size_t>(y) * xyz_stride / 2;
            for (int x = 0; x < w; x += stride) {
                const int16_t* q = row + static_cast<size_t>(x) * 3;
                Vec3 p(q[0] / 1000.0, q[1] / 1000.0, q[2] / 1000.0);
                if (xform_needed_) p = R_out_ * p + t_out_;
                // Deliberately after the transform: an invalid (0,0,0) point
                // becomes t_out and survives.  Preserved Python behaviour.
                f.valid.push_back(
                    (std::fabs(p.x()) + std::fabs(p.y()) + std::fabs(p.z())) >
                            1e-6
                        ? 1
                        : 0);
                f.points.push_back(p);
            }
        }
        return f;
    }

    // Pinhole fallback.  ★ Build from the UNROTATED grid (depth_m) with the
    // raw intrinsics (stride-rescaled), then apply R_out_ once — same as the
    // SDK path.  The Python original rotated the grid first and then applied
    // R_out_, which double-counted Rz whenever rotate != 0.
    // Validity is computed BEFORE the transform here (unlike the SDK path).
    cv::Mat depth_s;
    if (stride > 1) {
        depth_s.create((h + stride - 1) / stride, (w + stride - 1) / stride,
                       CV_32F);
        for (int y = 0, yy = 0; y < h; y += stride, ++yy) {
            for (int x = 0, xx = 0; x < w; x += stride, ++xx) {
                depth_s.at<float>(yy, xx) = depth_m.at<float>(y, x);
            }
        }
    } else {
        depth_s = depth_m;
    }

    const int sh = depth_s.rows;
    const int sw = depth_s.cols;
    const double fx = intr_depth_.fx / stride;
    const double fy = intr_depth_.fy / stride;
    const double cx = intr_depth_.cx / stride;
    const double cy = intr_depth_.cy / stride;
    f.points.reserve(static_cast<size_t>(sh) * sw);
    f.valid.reserve(static_cast<size_t>(sh) * sw);
    for (int v = 0; v < sh; ++v) {
        for (int u = 0; u < sw; ++u) {
            const float Z = depth_s.at<float>(v, u);
            const bool ok = Z > 0.0f && std::isfinite(Z);
            const double uu = static_cast<double>(u);
            const double vv = static_cast<double>(v);
            Vec3 p((uu - cx) * Z / fx, (vv - cy) * Z / fy, Z);
            if (xform_needed_) p = R_out_ * p + t_out_;
            f.valid.push_back(ok ? 1 : 0);
            f.points.push_back(p);
        }
    }
    return f;
}

}  // namespace kvc
