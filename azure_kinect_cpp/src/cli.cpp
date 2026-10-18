#include "kvc/cli.hpp"

#include "kvc/body_calib.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kvc {
namespace {

bool file_exists(const std::string& p) {
    std::ifstream f(p);
    return static_cast<bool>(f);
}

const char* kUsage =
    "用法: run_azure_cpp [选项]\n"
    "\n"
    "Azure Kinect 全流程（C++）—— 彩色检测 + 深度几何 + 串口 JSONL\n"
    "\n"
    "  --engine PATH          TensorRT engine（默认自动找 models/best_trt.engine）\n"
    "  --score F              检测分数阈值（默认 0.4）\n"
    "  --edge M               方块边长兜底（默认 0.25）\n"
    "  --no-edge-auto         不从拟合面量边长\n"
    "  --min-dist M           近裁剪（默认 0.3）\n"
    "  --max-dist M           远裁剪（默认 1.5）\n"
    "  --plane-thresh M       平面阈值下限（默认 0.003）\n"
    "  --table-thresh M       桌面阈值（默认 0.010）\n"
    "  --voxel M              体素边长（默认 0.02）\n"
    "  --min-cluster-points N 最小簇点数（0=按 edge 推）\n"
    "  --face-select M        most_facing | largest\n"
    "  --depth-noise-coef C   自适应平面阈值系数（默认 0.008）\n"
    "  --max-faces N          每簇最多拟合几个面（默认 3）\n"
    "  --stride N             深度抽样步长（默认 2）\n"
    "  --depth-mode M         nfov | nfov2x2 | wfov | wfov2x2\n"
    "  --no-color             不取彩色（几何仍跑）\n"
    "  --rotate DEG           0 | 90 | 180 | 270（必须和标定文件一致）\n"
    "  --color-res M          720p | 1080p | 1440p | 1536p | 2160p | off\n"
    "  --calib PATH           车体标定（默认 $K4A_BODY_CALIB 或 body_calib.json）\n"
    "  --allow-uncalibrated   没有标定也跑（输出不是车体系，别接电控）\n"
    "  --no-gravity           不做重力对齐\n"
    "  --only-classified      只输出配上类别的物体\n"
    "  --print                每帧打一行文字\n"
    "  --json                 每帧打一行 JSON\n"
    "  --serial PATH          串口发送\n"
    "  --serial-dry           不接硬件，把 JSONL 打到 stdout\n"
    "  --baud N               波特率（默认 115200）\n"
    "  --hz F                 串口发送频率上限（默认 10）\n"
    "  --frames N             跑 N 帧后退出（0=一直跑）\n"
    "  --save-dir DIR         存带检测框的 JPEG\n"
    "  --save-every N         每 N 帧存一张（默认 1）\n"
    "  --no-split             关掉按 2D 检测框切簇\n"
    "  --mm                   对象长度字段用毫米（normal/角度不动）\n"
    "  --no-dedup             关掉吞并抑制\n"
    "  --geo-verbose          几何管线每步日志\n"
    "  --profile              打印各阶段耗时\n"
    "  -h, --help             显示本帮助\n";

[[noreturn]] void usage_error(const std::string& msg) {
    std::cerr << kUsage << "\nrun_azure_cpp: error: " << msg << "\n";
    std::exit(2);
}

long long parse_int(const std::string& s, const std::string& flag) {
    try {
        size_t pos = 0;
        const long long v = std::stoll(s, &pos);
        if (pos != s.size()) throw std::invalid_argument("trailing");
        return v;
    } catch (...) {
        usage_error(flag + " 需要一个整数，收到 '" + s + "'");
    }
}

double parse_double(const std::string& s, const std::string& flag) {
    try {
        size_t pos = 0;
        const double v = std::stod(s, &pos);
        if (pos != s.size()) throw std::invalid_argument("trailing");
        return v;
    } catch (...) {
        usage_error(flag + " 需要一个数，收到 '" + s + "'");
    }
}

}  // namespace

std::string default_engine_path() {
    const std::string dir = executable_dir();
    const std::vector<std::string> cands = {
        dir + "/models/best_trt.engine",
        dir + "/../models/best_trt.engine",
        dir + "/../../models/best_trt.engine",
    };
    for (const auto& p : cands) {
        if (file_exists(p)) return p;
    }
    if (const char* env = std::getenv("K4A_ENGINE")) {
        if (*env) return env;
    }
    // Not found: return the most likely board location so the error message
    // points at the right place.
    return cands.back();
}

int parse_run_args(int argc, char** argv, RunOptions* out) {
    // Support --key=value by splitting into two tokens.
    std::vector<std::string> av;
    av.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
            const size_t eq = a.find('=');
            if (eq != std::string::npos) {
                av.push_back(a.substr(0, eq));
                av.push_back(a.substr(eq + 1));
                continue;
            }
        }
        av.push_back(a);
    }

    RunOptions o;
    for (size_t i = 1; i < av.size(); ++i) {
        const std::string a = av[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= av.size()) {
                usage_error(std::string(flag) + " 缺少参数");
            }
            return av[++i];
        };

        if (a == "-h" || a == "--help") {
            std::cout << kUsage;
            return 1;
        } else if (a == "--engine") {
            o.engine = next("--engine");
        } else if (a == "--score") {
            o.score = parse_double(next("--score"), "--score");
        } else if (a == "--edge") {
            o.edge = parse_double(next("--edge"), "--edge");
        } else if (a == "--no-edge-auto") {
            o.edge_auto = false;
        } else if (a == "--min-dist") {
            o.min_dist = parse_double(next("--min-dist"), "--min-dist");
        } else if (a == "--max-dist") {
            o.max_dist = parse_double(next("--max-dist"), "--max-dist");
        } else if (a == "--plane-thresh") {
            o.plane_thresh =
                parse_double(next("--plane-thresh"), "--plane-thresh");
        } else if (a == "--table-thresh") {
            o.table_thresh =
                parse_double(next("--table-thresh"), "--table-thresh");
        } else if (a == "--voxel") {
            o.voxel = parse_double(next("--voxel"), "--voxel");
        } else if (a == "--min-cluster-points") {
            o.min_cluster_points = static_cast<int>(parse_int(
                next("--min-cluster-points"), "--min-cluster-points"));
        } else if (a == "--face-select") {
            o.face_select = next("--face-select");
            if (o.face_select != "most_facing" && o.face_select != "largest") {
                usage_error("--face-select 只能是 most_facing / largest");
            }
        } else if (a == "--depth-noise-coef") {
            o.depth_noise_coef =
                parse_double(next("--depth-noise-coef"), "--depth-noise-coef");
        } else if (a == "--max-faces") {
            o.max_faces =
                static_cast<int>(parse_int(next("--max-faces"), "--max-faces"));
        } else if (a == "--stride") {
            o.stride = static_cast<int>(parse_int(next("--stride"), "--stride"));
        } else if (a == "--depth-mode") {
            o.depth_mode = next("--depth-mode");
            if (o.depth_mode != "nfov" && o.depth_mode != "nfov2x2" &&
                o.depth_mode != "wfov" && o.depth_mode != "wfov2x2") {
                usage_error("--depth-mode 只能是 nfov/nfov2x2/wfov/wfov2x2");
            }
        } else if (a == "--no-color") {
            o.color = false;
        } else if (a == "--rotate") {
            o.rotate = static_cast<int>(parse_int(next("--rotate"), "--rotate"));
            if (o.rotate != 0 && o.rotate != 90 && o.rotate != 180 &&
                o.rotate != 270) {
                usage_error("--rotate 只能是 0/90/180/270");
            }
        } else if (a == "--color-res") {
            o.color_res = next("--color-res");
            if (o.color_res != "720p" && o.color_res != "1080p" &&
                o.color_res != "1440p" && o.color_res != "1536p" &&
                o.color_res != "2160p" && o.color_res != "off") {
                usage_error(
                    "--color-res 只能是 720p/1080p/1440p/1536p/2160p/off");
            }
        } else if (a == "--calib") {
            o.calib = next("--calib");
        } else if (a == "--allow-uncalibrated") {
            o.allow_uncalibrated = true;
        } else if (a == "--no-gravity") {
            o.gravity = false;
        } else if (a == "--only-classified") {
            o.only_classified = true;
        } else if (a == "--print") {
            o.show_text = true;
        } else if (a == "--json") {
            o.json_out = true;
        } else if (a == "--serial") {
            o.serial = next("--serial");
        } else if (a == "--serial-dry") {
            o.serial_dry = true;
        } else if (a == "--baud") {
            o.baud = static_cast<int>(parse_int(next("--baud"), "--baud"));
        } else if (a == "--hz") {
            o.hz = parse_double(next("--hz"), "--hz");
        } else if (a == "--frames") {
            o.frames =
                parse_int(next("--frames"), "--frames");
        } else if (a == "--save-dir") {
            o.save_dir = next("--save-dir");
        } else if (a == "--save-every") {
            o.save_every =
                static_cast<int>(parse_int(next("--save-every"), "--save-every"));
        } else if (a == "--no-split") {
            o.split_boxes = false;
        } else if (a == "--mm") {
            o.mm = true;
        } else if (a == "--no-dedup") {
            o.dedup = false;
        } else if (a == "--geo-verbose") {
            o.geo_verbose = true;
        } else if (a == "--profile") {
            o.profile = true;
        } else if (a == "--ros" || a == "--ros-hz" ||
                   a == "--ros-max-points" || a == "--no-ros-image") {
            usage_error(
                a + " 没有在 C++ 版里移植（ROS 发送仍用 Python 版）；"
                    "删掉这个参数再跑");
        } else {
            usage_error("未知参数: " + a);
        }
    }

    if (o.engine.empty()) o.engine = default_engine_path();
    if (o.stride < 1) o.stride = 1;
    if (o.save_every < 1) o.save_every = 1;
    if (o.max_faces < 1) o.max_faces = 1;
    *out = o;
    return 0;
}

}  // namespace kvc
