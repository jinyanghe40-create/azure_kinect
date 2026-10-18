#pragma once

#include "kvc/body_calib.hpp"
#include "kvc/types.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace kvc {

class CaptureError : public std::runtime_error {
public:
    explicit CaptureError(const std::string& what) : std::runtime_error(what) {}
};

// Numeric values match k4a_depth_mode_t / k4a_color_resolution_t /
// k4a_frames_per_second_t, so this header does not pull in the SDK.
struct SourceConfig {
    int depth_mode = 2;         // K4A_DEPTH_MODE_NFOV_UNBINNED
    int color_resolution = 1;   // K4A_COLOR_RESOLUTION_720P
    int fps = 1;                // K4A_FRAMES_PER_SECOND_15
    bool color = true;
    int rotate = 0;
    int device_index = 0;
    bool verbose = true;
    bool use_sdk_points = true;
    bool read_gravity = true;
};

struct FrameData {
    cv::Mat depth_m;     // CV_32F, rotated (np.rot90 equivalent), meters
    cv::Mat color_bgr;   // CV_8UC3, rotated; empty when color is off
    // ★ 未旋转传感器栅格的内参（全分辨率）。depth_m 是【旋转后】的图像；
    //   rotate=90/270 时二者不能配对使用（fx/fy、cx/cy、宽高都要换）。
    //   目前没有消费者，保留原语义，别拿它去反投影 depth_m。
    Intrinsics intr_depth;
    Intrinsics intr_color;
    int depth_width = 0;
    int depth_height = 0;

    // Stride-sampled points in the output frame (body frame when calibrated),
    // flattened row-major over the raw SDK grid.
    std::vector<Vec3> points;
    std::vector<uint8_t> valid;  // |x|+|y|+|z| > 1e-6 after the transform
};

struct GravityInfo {
    bool ok = false;
    int n_samples = 0;
    double accel_norm = 0.0;
    double accel_jitter = 0.0;
    double gyro_norm = 0.0;
    double tilt_deg = 0.0;
    std::string err;
};

// Thin RAII wrapper over the Azure Kinect SDK.  Mirrors azure_capture.py:
// same rotation handling, same extrinsics closing formula, same gravity
// fallback, same "valid computed after the transform" quirk.
class AzureSource {
public:
    explicit AzureSource(const SourceConfig& cfg,
                         std::optional<BodyCalib> body_calib = std::nullopt);
    ~AzureSource();

    AzureSource(const AzureSource&) = delete;
    AzureSource& operator=(const AzureSource&) = delete;

    // Returns nullopt on timeout.  Fatal SDK failures throw CaptureError.
    std::optional<FrameData> get(int stride = 1, int timeout_ms = 2000);

    // Output frame -> post-Rz camera optical frame.  Directions use R_cam
    // only; positions additionally use t_cam.
    Vec3 to_camera(const Vec3& p) const { return R_cam_ * (p) + t_cam_; }
    Vec3 direction_to_camera(const Vec3& v) const { return R_cam_ * v; }

    const Mat3& R_out() const { return R_out_; }
    const Vec3& t_out() const { return t_out_; }
    const Mat3& R_cam() const { return R_cam_; }
    const Vec3& t_cam() const { return t_cam_; }
    const Extrinsics& extrinsics() const { return extr_; }  // depth -> color
    const Intrinsics& intr_depth() const { return intr_depth_; }
    const Intrinsics& intr_color() const { return intr_color_; }
    int rotate() const { return rotate_; }
    double px_fix_x() const { return px_fix_[0]; }
    double px_fix_y() const { return px_fix_[1]; }

    bool gravity_ok() const { return gravity_ok_; }
    const GravityInfo& gravity_info() const { return grav_info_; }
    const std::string& serial() const { return serial_; }
    const std::string& calib_id() const { return calib_id_; }
    bool has_color() const { return color_enabled_; }

    // One-off accelerometer read for diagnostics (check_body_frame D): the
    // unit "up" vector in the raw depth optical frame.  Does not touch R_out
    // and never runs during production.  On failure returns nullopt and, when
    // `err` is given, fills it with the reason.
    std::optional<Vec3> read_up_depth(std::string* err = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void read_gravity();
    void finish_rotations();

    SourceConfig cfg_;
    int rotate_ = 0;
    bool verbose_ = true;
    bool color_enabled_ = true;
    bool use_sdk_points_ = true;
    bool gravity_ok_ = false;
    bool xform_needed_ = false;

    std::string serial_ = "?";
    std::string calib_id_;
    Intrinsics intr_depth_;
    Intrinsics intr_color_;
    Extrinsics extr_base_;  // raw depth -> color, before the Rz conjugation
    Extrinsics extr_;       // conjugated, used for 2D/3D matching
    bool has_body_ = false;
    bool has_grav_ = false;
    Mat3 R_body_ = Mat3::Identity();
    Mat3 R_grav_ = Mat3::Identity();
    Mat3 R_out_ = Mat3::Identity();
    Vec3 t_out_ = Vec3::Zero();
    Mat3 R_cam_ = Mat3::Identity();
    Vec3 t_cam_ = Vec3::Zero();
    double px_fix_[2] = {0.0, 0.0};
    GravityInfo grav_info_;
};

}  // namespace kvc
