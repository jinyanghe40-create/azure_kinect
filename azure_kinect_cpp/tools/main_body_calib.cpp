// body_calib_cpp — C++ port of azure_kinect/body_calib.py.
//
// Same modes, same validation, same file format, same exit codes.

#include "kvc/body_calib.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

using kvc::BodyCalibError;
using kvc::ojson;

namespace {

const char* kUsage =
    "usage: body_calib_cpp (--write PATH | --check PATH | --verify PATH |\n"
    "                       --pitch-from-target --L MM --drop MM) [options]\n"
    "\n"
    "车体标定文件（相机 → 车体 的固定刚体变换）\n"
    "\n"
    "  --write PATH        生成标定文件（长度用 mm）\n"
    "  --check PATH        校验标定文件\n"
    "  --verify PATH       打印可核对的报告\n"
    "  --pitch-from-target 按靶板测量算 pitch（配合 --L/--drop）\n"
    "\n"
    "  --yaw DEG           偏航角°（>0 偏左，默认 0）\n"
    "  --pitch DEG         俯仰角°（>0 下俯）\n"
    "  --roll DEG          绕光轴的图像翻滚°（默认 180）\n"
    "  --forward MM        光心在车心【前】方 mm\n"
    "  --left MM           光心在车心【左】方 mm\n"
    "  --height MM         光心比车心【高】 mm\n"
    "  --id ID             标定 id\n"
    "  --serial SN         相机序列号（留档）\n"
    "  --rotate DEG        断言 image_rotate_deg 等于这个值\n"
    "  --fix               --check 时允许 SVD 重新正交化\n";

struct Args {
    std::optional<std::string> write, check, verify;
    bool pitch_from_target = false;
    std::optional<double> L, drop;
    double yaw = 0.0;
    std::optional<double> pitch;
    double roll = 180.0;
    std::optional<double> forward, left, height;
    std::string id, serial;
    std::optional<int> rotate;
    bool fix = false;
};

[[noreturn]] void usage_error(const std::string& msg) {
    std::cerr << kUsage << "\nbody_calib_cpp: error: " << msg << "\n";
    std::exit(2);
}

double need_value(int argc, char** argv, int& i) {
    if (i + 1 >= argc) usage_error(std::string("argument ") + argv[i] + ": expected one argument");
    return std::stod(argv[++i]);
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "-h" || k == "--help") {
            std::cout << kUsage;
            std::exit(0);
        } else if (k == "--write") {
            if (i + 1 >= argc) usage_error("--write: expected PATH");
            a.write = argv[++i];
        } else if (k == "--check") {
            if (i + 1 >= argc) usage_error("--check: expected PATH");
            a.check = argv[++i];
        } else if (k == "--verify") {
            if (i + 1 >= argc) usage_error("--verify: expected PATH");
            a.verify = argv[++i];
        } else if (k == "--pitch-from-target") {
            a.pitch_from_target = true;
        } else if (k == "--L") {
            a.L = need_value(argc, argv, i);
        } else if (k == "--drop") {
            a.drop = need_value(argc, argv, i);
        } else if (k == "--yaw") {
            a.yaw = need_value(argc, argv, i);
        } else if (k == "--pitch") {
            a.pitch = need_value(argc, argv, i);
        } else if (k == "--roll") {
            a.roll = need_value(argc, argv, i);
        } else if (k == "--forward") {
            a.forward = need_value(argc, argv, i);
        } else if (k == "--left") {
            a.left = need_value(argc, argv, i);
        } else if (k == "--height") {
            a.height = need_value(argc, argv, i);
        } else if (k == "--id") {
            if (i + 1 >= argc) usage_error("--id: expected value");
            a.id = argv[++i];
        } else if (k == "--serial") {
            if (i + 1 >= argc) usage_error("--serial: expected value");
            a.serial = argv[++i];
        } else if (k == "--rotate") {
            a.rotate = static_cast<int>(need_value(argc, argv, i));
        } else if (k == "--fix") {
            a.fix = true;
        } else {
            usage_error("unrecognized arguments: " + k);
        }
    }
    const int modes = (a.write ? 1 : 0) + (a.check ? 1 : 0) +
                      (a.verify ? 1 : 0) + (a.pitch_from_target ? 1 : 0);
    if (modes == 0) usage_error("one of the arguments --write --check --verify --pitch-from-target is required");
    if (modes > 1) usage_error("argument --write/--check/--verify/--pitch-from-target: not allowed with each other");
    return a;
}

ojson read_json(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw BodyCalibError("标定文件读不了：" + path + "（文件不存在或无权限）");
    try {
        ojson d;
        f >> d;
        return d;
    } catch (const std::exception& e) {
        throw BodyCalibError("标定文件读不了：" + path +
                             "（JSONDecodeError: " + std::string(e.what()) + "）");
    }
}

void write_back(const std::string& path, const ojson& d, const kvc::CheckResult& c) {
    ojson out = d;
    ojson R = ojson::array();
    for (int i = 0; i < 3; ++i) {
        ojson row = ojson::array();
        for (int j = 0; j < 3; ++j) row.push_back(kvc::python_round(c.R(i, j), 10));
        R.push_back(row);
    }
    out["R_cam_to_body"] = R;
    ojson t = ojson::array();
    for (int i = 0; i < 3; ++i) t.push_back(kvc::python_round(c.t[i], 6));
    out["t_cam_to_body_m"] = t;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw BodyCalibError("写不了标定文件：" + path);
    f << out.dump(2) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    try {
        if (a.pitch_from_target) {
            if (!a.L || !a.drop) usage_error("--pitch-from-target 需要同时给 --L 和 --drop");
            const double p = std::atan2(*a.drop, *a.L) * 180.0 / M_PI;
            char line[256];
            std::snprintf(line, sizeof(line),
                          "靶板正前方 L=%.1f mm，靶心比传感器轴心低 %.1f mm\n"
                          "  → pitch = atan2(%.1f, %.1f) = %+.3f°（正 = 光轴下俯）\n"
                          "  生成时用： --pitch %.3f",
                          *a.L, *a.drop, *a.drop, *a.L, p, p);
            std::cout << line << std::endl;
            return 0;
        }

        if (a.verify) {
            const ojson d = read_json(*a.verify);
            kvc::print_report(d, std::cout);
            return 0;
        }

        if (a.check) {
            const ojson d = read_json(*a.check);
            const kvc::CheckResult c = kvc::check_calib(d, a.rotate, a.fix);
            std::cout << "✅ " << *a.check << " 合法\n";
            std::cout << "   image_rotate_deg = " << c.image_rotate_deg
                      << "°   det(R) = " << kvc::fmt_fixed(c.R.determinant(), 9) << "\n";
            const double res = kvc::upright_residual(c.R, c.image_rotate_deg);
            std::cout << "   竖直判据 = " << (res >= 0 ? "+" : "")
                      << kvc::fmt_fixed(res, 4) << "（<0 ✔）\n";
            std::cout << "   |t| = " << kvc::fmt_fixed(c.t.norm(), 4) << " m\n";
            for (const auto& w : c.warnings) std::cout << "   ⚠️  " << w << "\n";
            if (a.fix) {
                write_back(*a.check, d, c);
                std::cout << "   ↻ 已把正交化后的 R 写回 " << *a.check << "\n";
            }
            return 0;
        }

        // --write
        std::vector<std::string> missing;
        if (!a.pitch) missing.push_back("--pitch");
        if (!a.forward) missing.push_back("--forward");
        if (!a.left) missing.push_back("--left");
        if (!a.height) missing.push_back("--height");
        if (!missing.empty()) {
            std::string msg = "--write 还需要：";
            for (size_t i = 0; i < missing.size(); ++i) {
                if (i) msg += " ";
                msg += missing[i];
            }
            msg += "（长度都用 mm，跟卷尺一致）";
            usage_error(msg);
        }
        const ojson d = kvc::write_body_calib(
            *a.write, a.yaw, *a.pitch, a.roll, *a.forward, *a.left, *a.height,
            a.id, a.serial);
        std::cout << "✅ 已写 " << *a.write << "\n\n";
        kvc::print_report(d, std::cout);
        return 0;

    } catch (const BodyCalibError& e) {
        std::cerr << "❌ " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << std::endl;
        return 1;
    }
}
