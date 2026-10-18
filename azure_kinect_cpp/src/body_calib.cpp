#include "kvc/body_calib.hpp"

#include <Eigen/SVD>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>

namespace kvc {
namespace {

int py_mod(int a, int b) {
    const int r = a % b;
    return (r < 0) ? r + b : r;
}

long long json_int(const ojson& d, const char* key) {
    const auto& v = d.at(key);
    if (v.is_number_integer()) return v.get<long long>();
    if (v.is_number_unsigned()) return static_cast<long long>(v.get<unsigned long long>());
    if (v.is_number_float()) return static_cast<long long>(v.get<double>());
    if (v.is_string()) {
        try {
            return std::stoll(v.get<std::string>());
        } catch (...) {
            throw BodyCalibError(std::string("字段 ") + key + " 不是整数");
        }
    }
    throw BodyCalibError(std::string("字段 ") + key + " 不是整数");
}

double json_num(const ojson& v, const char* where) {
    if (v.is_number()) return v.get<double>();
    if (v.is_string()) {
        try {
            return std::stod(v.get<std::string>());
        } catch (...) {
        }
    }
    throw BodyCalibError(std::string(where) + " 必须全是数字");
}

Mat3 parse_R(const ojson& v) {
    if (!v.is_array() || v.size() != 3) {
        throw BodyCalibError(
            "R_cam_to_body 形状不对，要 3×3"
            "（行主序 [[r00,r01,r02],[...],[...]]）");
    }
    Mat3 R;
    for (int i = 0; i < 3; ++i) {
        const auto& row = v[i];
        if (!row.is_array() || row.size() != 3) {
            throw BodyCalibError(
                "R_cam_to_body 形状不对，要 3×3"
                "（行主序 [[r00,r01,r02],[...],[...]]）");
        }
        for (int j = 0; j < 3; ++j) R(i, j) = json_num(row[j], "R_cam_to_body");
    }
    return R;
}

Vec3 parse_t(const ojson& v) {
    std::vector<double> flat;
    auto push = [&](const ojson& x) {
        if (x.is_array()) {
            for (const auto& y : x) {
                if (y.is_array()) {
                    throw BodyCalibError("t_cam_to_body_m 要 3 个数");
                }
                flat.push_back(json_num(y, "t_cam_to_body_m"));
            }
        } else {
            flat.push_back(json_num(x, "t_cam_to_body_m"));
        }
    };
    if (v.is_array()) {
        for (const auto& x : v) push(x);
    } else {
        throw BodyCalibError("t_cam_to_body_m 要 3 个数");
    }
    if (flat.size() != 3) throw BodyCalibError("t_cam_to_body_m 要 3 个数");
    return Vec3(flat[0], flat[1], flat[2]);
}

Mat3 reorthonormalize(const Mat3& R) {
    Eigen::JacobiSVD<Mat3> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    const Mat3 Vt = svd.matrixV().transpose();
    Mat3 Rn = U * Vt;
    if (Rn.determinant() < 0.0) {
        U.col(2) = -U.col(2);
        Rn = U * Vt;
    }
    return Rn;
}

std::string today_iso() {
    std::time_t t = std::time(nullptr);
    std::tm tm_buf{};
    localtime_r(&t, &tm_buf);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm_buf);
    return buf;
}

ojson load_json_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw BodyCalibError("标定文件读不了：" + path + "（文件不存在或无权限）");
    }
    try {
        ojson d;
        f >> d;
        return d;
    } catch (const std::exception& e) {
        throw BodyCalibError("标定文件读不了：" + path +
                             "（JSONDecodeError: " + e.what() + "）");
    }
}

std::string describe(const Vec3& v) {
    const char* ax[3] = {"前", "左", "上"};
    int i = 0;
    if (std::fabs(v[1]) > std::fabs(v[i])) i = 1;
    if (std::fabs(v[2]) > std::fabs(v[i])) i = 2;
    const char* sgn = v[i] > 0.0 ? "+" : "−";
    const double mag = std::fabs(v[i]);
    std::ostringstream os;
    if (mag > 0.999) {
        os << "几乎正" << sgn << ax[i];
    } else if (mag > 0.1) {
        os << "主要在" << sgn << ax[i] << "（" << fmt_fixed(mag, 2) << "）";
        std::string others;
        for (int k = 0; k < 3; ++k) {
            if (k == i || std::fabs(v[k]) <= 0.05) continue;
            if (!others.empty()) others += ", ";
            others += std::string(ax[k]) + (v[k] > 0 ? "+" : "") + fmt_fixed(v[k], 2);
        }
        if (!others.empty()) os << "，还有 " << others;
    } else {
        os << "（没有主导方向 —— 这个数可能有问题）";
    }
    return os.str();
}

}  // namespace

std::string fmt_fixed(double v, int decimals) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(decimals) << v;
    return os.str();
}

std::string fmt_sci(double v, int decimals) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(decimals) << v;
    return os.str();
}

namespace {
// Explicit sign, because std::showpos does not survive a trip through a string.
std::string fmt_signed(double v, int decimals) {
    // Python's f"{v:+.2f}" uses the sign bit: -0.0 prints "-0.00", not
    // "+-0.00".  `v >= 0` is true for -0.0, so test the sign bit instead.
    return (std::signbit(v) ? "" : "+") + fmt_fixed(v, decimals);
}

// Python's f"{s:<18}" pads by code point, not by byte.  The axis names
// contain multi-byte UTF-8, so byte padding would silently not pad at all.
std::string pad_right(const std::string& s, size_t width) {
    size_t cp = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) ++cp;
    }
    if (cp >= width) return s;
    return s + std::string(width - cp, ' ');
}
}  // namespace

CheckResult check_calib(const ojson& d, std::optional<int> rotate, bool fix) {
    CheckResult out;
    if (!d.is_object()) throw BodyCalibError("标定文件顶层不是 JSON 对象");

    if (!d.contains("schema") || !d["schema"].is_string() ||
        d["schema"].get<std::string>() != kCalibSchema) {
        std::string actual = d.contains("schema") ? d["schema"].dump() : "null";
        throw BodyCalibError("schema 不对：期望 '" + std::string(kCalibSchema) +
                             "'，实际 " + actual +
                             "。（是不是把 k4a_calibration.py 的输出当车体标定用了？）");
    }

    for (const char* k : {"id", "image_rotate_deg", "R_cam_to_body", "t_cam_to_body_m"}) {
        if (!d.contains(k)) throw BodyCalibError(std::string("标定缺字段 '") + k + "'");
    }

    const int rot = static_cast<int>(json_int(d, "image_rotate_deg"));
    if (py_mod(rot, 90) != 0) {
        throw BodyCalibError("image_rotate_deg=" + std::to_string(rot) +
                             " 不是 90 的整数倍");
    }
    if (rotate.has_value() && py_mod(*rotate, 360) != py_mod(rot, 360)) {
        throw BodyCalibError(
            "命令行 --rotate " + std::to_string(*rotate) +
            " 与标定的 image_rotate_deg " + std::to_string(rot) + " 不一致。\n"
            "  ★ 这是故意设的硬断言：服务单元里曾经漏掉 --rotate 180，\n"
            "    于是彩色图是倒的、点云却按 0° 解释 —— 图倒着看得见，\n"
            "    点云错得静默。两者必须来自同一个数。\n"
            "  要么改标定，要么改命令行，别让它俩各说各话。");
    }

    Mat3 R = parse_R(d["R_cam_to_body"]);
    const Vec3 t = parse_t(d["t_cam_to_body_m"]);

    const double det = R.determinant();
    const double orth = (R * R.transpose() - Mat3::Identity()).cwiseAbs().maxCoeff();
    if (std::fabs(det - 1.0) > 1e-6 || orth > 1e-6) {
        if (!fix) {
            throw BodyCalibError(
                "R 不是合法旋转：det=" + fmt_fixed(det, 9) +
                "（要 +1），|RRᵀ−I|max=" + fmt_sci(orth, 3) + "（要 <1e-6）。\n"
                "  用 --check --fix 可以让它按 SVD 重新正交化后写回。");
        }
        out.warnings.push_back(
            "R 数值上不是严格旋转（det=" + fmt_fixed(det, 9) +
            ", |RRᵀ−I|=" + fmt_sci(orth, 3) + "）→ 已按 SVD 重新正交化");
        R = reorthonormalize(R);
    }

    const double res = upright_residual(R, static_cast<double>(rot));
    if (res > -1e-9) {
        throw BodyCalibError(
            "没通过『图像竖直』判据：(R @ Rz(" + std::to_string(rot) +
            ")ᵀ)[2,1] = " + (res >= 0 ? "+" : "") + fmt_fixed(res, 4) + " ≥ 0\n"
            "  含义：把图像翻滚还原后，『图像里的下』没有指向车体下方。\n"
            "  · 第二列 ≈ +1：矩阵是按 post-Rz 系（③）写的 → 180° 会被\n"
            "    重复计入，点云整个翻过去，而且 det 仍是 +1，完全静默。\n"
            "    应当存 raw SDK 深度系 → 车体系的物理真相（①→④）。\n"
            "  · 第三列也不像光轴：多半是矩阵转置写反了。\n"
            "  提示： body_calib_cpp --verify <文件> 会把三个基向量\n"
            "        和推导出的 yaw/pitch/roll 都打出来，一眼看得出哪错了。");
    }
    if (res > -0.5) {
        out.warnings.push_back(
            "竖直判据只勉强通过（" + std::string(res >= 0 ? "+" : "") +
            fmt_fixed(res, 3) +
            "，理想接近 −1）—— 光轴接近水平但『下』方向偏了，检查 roll/矩阵是否写歪");
    }

    const double tn = t.norm();
    if (!(tn > 0.0 && tn <= 2.0)) {
        throw BodyCalibError("|t| = " + fmt_fixed(tn, 4) + " m 不在 (0, 2] 内。" +
                             "单位是不是搞错了（mm 当 m 写）？");
    }

    out.R = R;
    out.t = t;
    out.image_rotate_deg = rot;
    return out;
}

std::string executable_dir() {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return ".";
    buf[n] = '\0';
    std::string p(buf);
    const auto pos = p.find_last_of('/');
    return (pos == std::string::npos) ? std::string(".") : p.substr(0, pos);
}

namespace {
bool file_exists(const std::string& p) {
    std::ifstream f(p);
    return static_cast<bool>(f);
}
}  // namespace

std::string default_calib_path() {
    if (const char* env = std::getenv("K4A_BODY_CALIB")) {
        if (*env) return env;
    }
    const std::string dir = executable_dir();
    const std::string local = dir + "/body_calib.json";
    if (file_exists(local)) return local;
    const std::string sibling = dir + "/../body_calib.json";
    if (file_exists(sibling)) return sibling;
    return local;
}

std::optional<BodyCalib> load_body_calib(const std::string& path,
                                         std::optional<int> rotate,
                                         bool allow_missing) {
    const std::string p = path.empty() ? default_calib_path() : path;
    if (!file_exists(p)) {
        if (allow_missing) return std::nullopt;
        throw BodyCalibError(
            "没有车体标定文件：" + p + "\n"
            "  相机→车体必须先标一次（刚性固定，一次永久用）。\n"
            "  生成：  body_calib_cpp --write " + p +
            " --pitch .. --forward .. --left .. --height ..\n"
            "  临时跳过：加 --allow-uncalibrated（会退回相机系/重力对齐，"
            "输出【不是】车体坐标，别接电控）");
    }
    const ojson d = load_json_file(p);
    const CheckResult c = check_calib(d, rotate);

    BodyCalib out;
    out.R = c.R;
    out.t = c.t;
    out.id = d.value("id", std::string(""));
    out.image_rotate_deg = c.image_rotate_deg;
    out.path = p;
    out.warnings = c.warnings;
    out.raw = d;
    return out;
}

std::optional<BodyCalib> load_body_calib_default(const std::string& path,
                                                 std::optional<int> rotate,
                                                 bool allow_missing) {
    return load_body_calib(path, rotate, allow_missing);
}

ojson write_body_calib(const std::string& path, double yaw_deg, double pitch_deg,
                       double roll_deg, double forward_mm, double left_mm,
                       double height_mm, const std::string& calib_id,
                       const std::string& serial, const std::string& depth_mode,
                       const std::string& color, const ojson& tape_extra) {
    const Mat3 R = build_R(yaw_deg, pitch_deg, roll_deg);
    const Vec3 t(forward_mm / 1000.0, left_mm / 1000.0, height_mm / 1000.0);

    ojson d;
    d["schema"] = kCalibSchema;
    d["version"] = kCalibVersion;
    d["id"] = calib_id.empty() ? "unnamed" : calib_id;
    d["created"] = today_iso();
    d["body_frame"] = ojson{
        {"convention", "ROS REP-103: X前 / Y左 / Z上"},
        {"origin", "chassis geometric center (含高度)"},
        {"units", "position in m, angle in deg"},
    };
    d["camera"] = ojson{{"serial", serial},
                        {"depth_mode", depth_mode},
                        {"color", color}};
    d["image_rotate_deg"] = py_mod(static_cast<int>(roll_deg), 360);

    ojson Rj = ojson::array();
    for (int i = 0; i < 3; ++i) {
        ojson row = ojson::array();
        for (int j = 0; j < 3; ++j) row.push_back(python_round(R(i, j), 10));
        Rj.push_back(row);
    }
    d["R_cam_to_body"] = Rj;

    ojson tj = ojson::array();
    for (int i = 0; i < 3; ++i) tj.push_back(python_round(t[i], 6));
    d["t_cam_to_body_m"] = tj;

    ojson tape;
    tape["yaw_deg"] = yaw_deg;
    tape["pitch_deg"] = pitch_deg;
    tape["roll_deg"] = roll_deg;
    tape["forward_mm"] = forward_mm;
    tape["left_mm"] = left_mm;
    tape["height_mm"] = height_mm;
    tape["note"] = "角度用公式 R = Rz_body(yaw)·Ry_body(pitch)·C0·Rz_optical(roll)";
    if (tape_extra.is_object()) {
        for (auto it = tape_extra.begin(); it != tape_extra.end(); ++it) {
            tape[it.key()] = it.value();
        }
    }
    d["tape"] = tape;

    // Self-check before writing, then store the (possibly reorthonormalized) R.
    const CheckResult c = check_calib(d, py_mod(static_cast<int>(roll_deg), 360));
    ojson Rf = ojson::array();
    for (int i = 0; i < 3; ++i) {
        ojson row = ojson::array();
        for (int j = 0; j < 3; ++j) row.push_back(python_round(c.R(i, j), 10));
        Rf.push_back(row);
    }
    d["R_cam_to_body"] = Rf;
    ojson tf = ojson::array();
    for (int i = 0; i < 3; ++i) tf.push_back(python_round(c.t[i], 6));
    d["t_cam_to_body_m"] = tf;

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw BodyCalibError("写不了标定文件：" + path);
    f << d.dump(2) << "\n";
    return d;
}

void print_report(const ojson& d, std::ostream& out) {
    const CheckResult c = check_calib(d);
    const auto [yaw, pitch, roll] = decompose_R(c.R);

    out << "标定 " << d.value("id", std::string("?"))
        << "   文件旋转 image_rotate_deg = " << c.image_rotate_deg << "°\n";
    out << "  车体系：X前 / Y左 / Z上，原点 = 车体几何中心（带高度）\n\n";
    out << "  ★ R_cam_to_body 的三个基（列 = 该轴在车体系里的方向）\n";
    const char* names[3] = {"① +X  图像『右』", "① +Y  图像『下』", "① +Z  光轴"};
    for (int j = 0; j < 3; ++j) {
        const Vec3 v = c.R.col(j);
        const std::string nm = pad_right(names[j], 18);
        char line[200];
        std::snprintf(line, sizeof(line), "    %s → (%+.4f, %+.4f, %+.4f)   ",
                      nm.c_str(), v[0], v[1], v[2]);
        out << line << describe(v) << "\n";
    }
    out << "\n  反解回角度： yaw " << fmt_signed(yaw, 2)
        << "°   pitch " << fmt_signed(pitch, 2)
        << "°   roll " << fmt_signed(roll, 2) << "°\n";
    out << "    pitch > 0 = 光轴【向下俯视】； yaw > 0 = 光轴偏【左】\n\n";

    const double res = upright_residual(c.R, static_cast<double>(c.image_rotate_deg));
    out << "  ★ 竖直判据 (R @ Rz(" << c.image_rotate_deg << ")ᵀ)[2,1] = "
        << fmt_signed(res, 4) << "   （必须 < 0）"
        << (res < 0 ? "  ✅" : "  ❌") << "\n";
    const Vec3 down = (c.R * Rz_optical(static_cast<double>(c.image_rotate_deg)).transpose()).col(1);
    out << "    把图像翻滚还原后，『图像里的下』指向 " << describe(down) << "\n";
    const double again = upright_residual(
        c.R * Rz_optical(static_cast<double>(c.image_rotate_deg)),
        static_cast<double>(c.image_rotate_deg));
    out << "    再乘一次 Rz 会变成 " << fmt_signed(again, 4);
    if (again >= 0) {
        out << " → 会被判据拒掉 ✅（说明判据真的在起作用）\n";
    } else {
        out << " → ⚠️ 判据区分不出来\n";
    }
    out << "\n  光心位置（车心指向深度光心）： 前 " << fmt_signed(c.t[0] * 1000.0, 1)
        << " mm  左 " << fmt_signed(c.t[1] * 1000.0, 1)
        << " mm  高 " << fmt_signed(c.t[2] * 1000.0, 1)
        << " mm   |t| = " << fmt_fixed(c.t.norm(), 4) << " m\n";
    out << "  det(R) = " << fmt_fixed(c.R.determinant(), 9)
        << "    |RRᵀ−I|max = "
        << fmt_sci((c.R * c.R.transpose() - Mat3::Identity()).cwiseAbs().maxCoeff(), 2)
        << "\n";
    if (!c.warnings.empty()) {
        out << "\n";
        for (const auto& w : c.warnings) out << "  ⚠️  " << w << "\n";
    }
}

}  // namespace kvc
