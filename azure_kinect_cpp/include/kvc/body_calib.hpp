#pragma once

#include "kvc/math.hpp"
#include "kvc/types.hpp"

#include <nlohmann/json.hpp>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace kvc {

using ojson = nlohmann::ordered_json;

inline constexpr const char* kCalibSchema = "k4a_body_calib";
inline constexpr int kCalibVersion = 1;

// Missing / invalid calibration.  Callers must refuse to start rather than
// silently fall back to the camera frame.
class BodyCalibError : public std::runtime_error {
public:
    explicit BodyCalibError(const std::string& what) : std::runtime_error(what) {}
};

struct CheckResult {
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();
    int image_rotate_deg = 0;
    std::vector<std::string> warnings;
};

struct BodyCalib {
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();
    std::string id;
    int image_rotate_deg = 0;
    std::string path;
    std::vector<std::string> warnings;
    ojson raw;
};

// Validate a parsed calibration document.  Mirrors body_calib.check_calib:
// same rules, same order, same thresholds.
CheckResult check_calib(const ojson& d, std::optional<int> rotate = std::nullopt,
                        bool fix = false);

// Load and validate.  Missing file returns nullopt iff allow_missing, otherwise
// throws BodyCalibError.  rotate == nullopt means "do not assert".
std::optional<BodyCalib> load_body_calib(const std::string& path,
                                         std::optional<int> rotate = std::nullopt,
                                         bool allow_missing = false);

// Same as load_body_calib but resolves the default path when path is empty.
std::optional<BodyCalib> load_body_calib_default(
    const std::string& path, std::optional<int> rotate = std::nullopt,
    bool allow_missing = false);

// Create body_calib.json.  Lengths in millimeters, angles in degrees, exactly
// like body_calib.py --write.
ojson write_body_calib(const std::string& path, double yaw_deg, double pitch_deg,
                       double roll_deg, double forward_mm, double left_mm,
                       double height_mm, const std::string& calib_id = "",
                       const std::string& serial = "",
                       const std::string& depth_mode = "nfov",
                       const std::string& color = "720p",
                       const ojson& tape_extra = ojson::object());

// Human-readable report (body_calib.py --verify).
void print_report(const ojson& d, std::ostream& out);

// Path helpers.
std::string executable_dir();
std::string default_calib_path();

std::string fmt_fixed(double v, int decimals);
std::string fmt_sci(double v, int decimals);

}  // namespace kvc
