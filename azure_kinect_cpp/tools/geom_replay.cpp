// geom_replay: point cloud (PLY) + optional detection boxes -> geometry JSON.
//
// This is the C++ counterpart of `python3 common/pc_pipeline.py --replay FILE`,
// with the additions needed to exercise the 2D-box cluster split offline:
//   --boxes FILE        detections (JSON list of dicts or [x0,y0,x1,y1,cls,score])
//   --intr fx,fy,cx,cy  pinhole intrinsics used only to build the split uv
//   --size W,H          original color image size (rot_uv uses it)
//   --rotate 0|90|180|270
//   --px-fix U,V        principal-point correction from azure_capture
//
// Objects are emitted without their point arrays; every scalar field matches
// the Python pipeline's object dict.

#include "kvc/math.hpp"
#include "kvc/pc_pipeline.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::ordered_json;
using namespace kvc;

namespace {

struct Ply {
    std::vector<Vec3> pts;
};

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("无法打开 " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int type_size(const std::string& t) {
    if (t == "float" || t == "float32" || t == "int" || t == "uint") return 4;
    if (t == "double" || t == "float64") return 8;
    if (t == "uchar" || t == "uint8" || t == "char" || t == "int8") return 1;
    if (t == "short" || t == "ushort") return 2;
    return 4;
}

double read_scalar(const unsigned char* p, const std::string& t) {
    if (t == "float" || t == "float32") {
        float v;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }
    if (t == "double" || t == "float64") {
        double v;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }
    if (t == "uchar" || t == "uint8") return static_cast<double>(*p);
    if (t == "char" || t == "int8") return static_cast<double>(*reinterpret_cast<const int8_t*>(p));
    if (t == "short") { int16_t v; std::memcpy(&v, p, 2); return v; }
    if (t == "ushort") { uint16_t v; std::memcpy(&v, p, 2); return v; }
    if (t == "int") { int32_t v; std::memcpy(&v, p, 4); return v; }
    if (t == "uint") { uint32_t v; std::memcpy(&v, p, 4); return v; }
    return 0.0;
}

Ply read_ply(const std::string& path) {
    const std::string raw = read_file(path);
    const size_t hend = raw.find("end_header");
    if (hend == std::string::npos) throw std::runtime_error("不是合法的 PLY 文件");
    const std::string header = raw.substr(0, hend);
    size_t body_pos = hend + std::strlen("end_header");
    if (raw.compare(body_pos, 2, "\r\n") == 0) {
        body_pos += 2;
    } else if (raw.compare(body_pos, 1, "\n") == 0) {
        body_pos += 1;
    }

    size_t n = 0;
    std::vector<std::pair<std::string, std::string>> props;  // (type, name)
    bool in_vertex = false;
    bool ascii = false;
    bool binary_le = false;
    std::istringstream hs(header);
    std::string line;
    while (std::getline(hs, line)) {
        std::istringstream ls(line);
        std::vector<std::string> tok;
        std::string w;
        while (ls >> w) tok.push_back(w);
        if (tok.size() >= 3 && tok[0] == "element") {
            in_vertex = (tok[1] == "vertex");
            if (in_vertex) n = static_cast<size_t>(std::stoull(tok[2]));
        } else if (tok.size() >= 3 && tok[0] == "property" && in_vertex) {
            props.emplace_back(tok[1], tok[2]);
        }
        if (line.find("format ascii") != std::string::npos) ascii = true;
        if (line.find("format binary_little_endian") != std::string::npos)
            binary_le = true;
    }
    if (props.empty()) throw std::runtime_error("PLY 缺少属性");

    std::vector<size_t> offsets(props.size());
    size_t stride = 0;
    for (size_t i = 0; i < props.size(); ++i) {
        offsets[i] = stride;
        stride += static_cast<size_t>(type_size(props[i].first));
    }
    int ix = -1, iy = -1, iz = -1;
    for (size_t i = 0; i < props.size(); ++i) {
        if (props[i].second == "x") ix = static_cast<int>(i);
        if (props[i].second == "y") iy = static_cast<int>(i);
        if (props[i].second == "z") iz = static_cast<int>(i);
    }
    if (ix < 0 || iy < 0 || iz < 0) throw std::runtime_error("PLY 缺少 x/y/z 属性");

    Ply out;
    out.pts.reserve(n);
    if (binary_le) {
        const unsigned char* body =
            reinterpret_cast<const unsigned char*>(raw.data() + body_pos);
        const size_t avail = raw.size() - body_pos;
        if (avail < n * stride) throw std::runtime_error("PLY 数据长度不足");
        for (size_t i = 0; i < n; ++i) {
            const unsigned char* rec = body + i * stride;
            out.pts.emplace_back(
                read_scalar(rec + offsets[static_cast<size_t>(ix)], props[static_cast<size_t>(ix)].first),
                read_scalar(rec + offsets[static_cast<size_t>(iy)], props[static_cast<size_t>(iy)].first),
                read_scalar(rec + offsets[static_cast<size_t>(iz)], props[static_cast<size_t>(iz)].first));
        }
    } else if (ascii) {
        std::istringstream ss(raw.substr(body_pos));
        for (size_t i = 0; i < n; ++i) {
            double a = 0, b = 0, c = 0;
            if (!(ss >> a >> b >> c)) break;
            out.pts.emplace_back(a, b, c);
        }
    } else {
        throw std::runtime_error("不支持的 PLY 格式");
    }
    return out;
}

bool parse_vec3(const std::string& s, Vec3* out) {
    std::string t = s;
    std::replace(t.begin(), t.end(), ';', ',');
    std::vector<double> v;
    std::istringstream ss(t);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        try {
            v.push_back(std::stod(item));
        } catch (...) {
            return false;
        }
    }
    if (v.size() != 3) return false;
    *out = Vec3(v[0], v[1], v[2]);
    return true;
}

std::vector<Detection> read_boxes(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("无法打开 boxes 文件 " + path);
    nlohmann::json j;
    f >> j;
    std::vector<Detection> boxes;
    if (!j.is_array()) throw std::runtime_error("boxes JSON 必须是数组");
    for (const auto& e : j) {
        Detection d;
        if (e.is_array()) {
            if (e.size() < 4) continue;
            d.x0 = e[0].get<double>();
            d.y0 = e[1].get<double>();
            d.x1 = e[2].get<double>();
            d.y1 = e[3].get<double>();
            if (e.size() > 4) d.cls_id = e[4].get<int>();
            if (e.size() > 5) d.score = e[5].get<double>();
        } else if (e.is_object()) {
            d.x0 = e.value("x0", 0.0);
            d.y0 = e.value("y0", 0.0);
            d.x1 = e.value("x1", 0.0);
            d.y1 = e.value("y1", 0.0);
            if (!e.contains("cls") || e["cls"].is_null()) {
                d.cls_id = e.value("cls_id", 0);
            } else {
                d.cls_id = e["cls"].get<int>();
            }
            d.score = e.value("score", 0.0);
        } else {
            continue;
        }
        boxes.push_back(d);
    }
    return boxes;
}

void rot_uv(double* u, double* v, int k, double w0, double h0) {
    const double uu = *u, vv = *v;
    if (k == 1) {
        *u = h0 - 1.0 - vv;
        *v = uu;
    } else if (k == 2) {
        *u = w0 - 1.0 - uu;
        *v = h0 - 1.0 - vv;
    } else if (k == 3) {
        *u = vv;
        *v = w0 - 1.0 - uu;
    }
}

ordered_json vec_json(const Vec3& v, int nd) {
    return ordered_json::array({python_round(v[0], nd), python_round(v[1], nd),
                                python_round(v[2], nd)});
}

ordered_json object_json(const GeoObject& o) {
    ordered_json j;
    j["id"] = o.id;
    j["face_center_m"] = vec_json(o.face_center, 4);
    j["normal"] = vec_json(o.normal, 4);
    j["angle_deg"] = python_round(o.angle_deg, 2);
    j["n_points"] = o.n_points;
    j["n_faces"] = o.n_faces;
    j["face_rms_mm"] = python_round(o.face_rms_mm, 2);
    if (o.lean_deg) {
        j["lean_deg"] = python_round(*o.lean_deg, 2);
    } else {
        j["lean_deg"] = nullptr;
    }
    ordered_json fj = ordered_json::array();
    for (const Face& f : o.faces) {
        ordered_json e;
        e["normal"] = vec_json(f.normal, 4);
        e["center"] = vec_json(f.center, 4);
        e["angle_deg"] = python_round(f.angle_deg, 2);
        if (f.lean_deg) {
            e["lean_deg"] = python_round(*f.lean_deg, 2);
        } else {
            e["lean_deg"] = nullptr;
        }
        e["n_points"] = f.n_points;
        e["rms_mm"] = python_round(f.rms_mm, 2);
        fj.push_back(e);
    }
    j["faces"] = fj;
    if (o.cube_center) {
        j["cube_center_m"] = vec_json(*o.cube_center, 4);
    }
    j["n_face_pts"] = o.n_face_pts;
    j["nearest_m"] = python_round(o.nearest_m, 4);
    j["nearest_pt_m"] = vec_json(o.nearest_pt, 4);
    j["plane_thresh_used_mm"] = python_round(o.plane_thresh_used_mm, 2);
    j["sigma_z_mm"] = python_round(o.sigma_z_mm, 2);
    j["edge_used_m"] = python_round(o.edge_used_m, 4);
    j["edge_src"] = o.edge_src;
    if (!o.edge_faces_m.empty()) {
        ordered_json ev = ordered_json::array();
        for (double v : o.edge_faces_m) ev.push_back(python_round(v, 4));
        j["edge_faces_m"] = ev;
    }
    return j;
}

void usage() {
    std::fprintf(
        stderr,
        "用法: geom_replay --replay FILE [选项]\n"
        "  --edge M --no-edge-auto --min-dist M --max-dist M\n"
        "  --plane-thresh M --table-thresh M --voxel M --min-cluster-points N\n"
        "  --max-faces N --depth-noise-coef C --face-select most_facing|largest\n"
        "  --up-axis X,Y,Z --cam-axis X,Y,Z --cam-origin X,Y,Z\n"
        "  --boxes FILE --intr fx,fy,cx,cy --size W,H --rotate 0|90|180|270\n"
        "  --px-fix U,V --json\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string replay;
    std::string boxes_path;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    bool have_intr = false;
    int img_w = 0, img_h = 0;
    int rotate = 0;
    double pfx = 0.0, pfy = 0.0;
    bool json_only = false;

    PipelineArgs args;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s 需要参数\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--replay") replay = next("--replay");
        else if (a == "--edge") args.edge = std::stod(next("--edge"));
        else if (a == "--no-edge-auto") args.edge_auto = false;
        else if (a == "--min-dist") args.min_dist = std::stod(next("--min-dist"));
        else if (a == "--max-dist") args.max_dist = std::stod(next("--max-dist"));
        else if (a == "--plane-thresh") args.plane_thresh = std::stod(next("--plane-thresh"));
        else if (a == "--table-thresh") args.table_thresh = std::stod(next("--table-thresh"));
        else if (a == "--voxel") args.voxel = std::stod(next("--voxel"));
        else if (a == "--min-cluster-points")
            args.min_cluster_points = std::stoi(next("--min-cluster-points"));
        else if (a == "--max-faces") args.max_faces = std::stoi(next("--max-faces"));
        else if (a == "--depth-noise-coef")
            args.depth_noise_coef = std::stod(next("--depth-noise-coef"));
        else if (a == "--face-select") {
            const std::string v = next("--face-select");
            if (v == "largest") args.face_select = FaceSelect::Largest;
            else if (v == "most_facing") args.face_select = FaceSelect::MostFacing;
            else {
                std::fprintf(stderr, "--face-select 只能是 most_facing / largest\n");
                return 2;
            }
        } else if (a == "--up-axis") {
            Vec3 v;
            if (!parse_vec3(next("--up-axis"), &v)) {
                std::fprintf(stderr, "--up-axis 要三个数，形如 0,0,1\n");
                return 2;
            }
            args.up_axis = v;
        } else if (a == "--gravity") {
            args.up_axis = Vec3(0.0, 1.0, 0.0);
        } else if (a == "--cam-axis") {
            Vec3 v;
            if (!parse_vec3(next("--cam-axis"), &v)) {
                std::fprintf(stderr, "--cam-axis 要三个数，形如 0,0,1\n");
                return 2;
            }
            args.cam_axis = v;
        } else if (a == "--cam-origin") {
            Vec3 v;
            if (!parse_vec3(next("--cam-origin"), &v)) {
                std::fprintf(stderr, "--cam-origin 要三个数，形如 0,0,0\n");
                return 2;
            }
            args.cam_origin = v;
        } else if (a == "--boxes") boxes_path = next("--boxes");
        else if (a == "--intr") {
            const std::string v = next("--intr");
            if (std::sscanf(v.c_str(), "%lf,%lf,%lf,%lf", &fx, &fy, &cx, &cy) != 4) {
                std::fprintf(stderr, "--intr 形如 fx,fy,cx,cy\n");
                return 2;
            }
            have_intr = true;
        } else if (a == "--size") {
            const std::string v = next("--size");
            if (std::sscanf(v.c_str(), "%d,%d", &img_w, &img_h) != 2) {
                std::fprintf(stderr, "--size 形如 W,H\n");
                return 2;
            }
        } else if (a == "--rotate") {
            rotate = std::stoi(next("--rotate"));
            if (rotate != 0 && rotate != 90 && rotate != 180 && rotate != 270) {
                std::fprintf(stderr, "--rotate 只能是 0/90/180/270\n");
                return 2;
            }
        } else if (a == "--px-fix") {
            const std::string v = next("--px-fix");
            if (std::sscanf(v.c_str(), "%lf,%lf", &pfx, &pfy) != 2) {
                std::fprintf(stderr, "--px-fix 形如 U,V\n");
                return 2;
            }
        } else if (a == "--json") json_only = true;
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (replay.empty()) {
        usage();
        return 2;
    }
    (void)json_only;  // this tool always writes JSON to stdout

    std::vector<Vec3> pts;
    try {
        pts = read_ply(replay).pts;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "读取失败: %s\n", e.what());
        return 1;
    }

    if (!boxes_path.empty()) {
        try {
            args.split_boxes = read_boxes(boxes_path);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "读取 boxes 失败: %s\n", e.what());
            return 1;
        }
        if (have_intr) {
            const int k = rotate / 90;
            args.project = [=](const Vec3& p) -> std::optional<Eigen::Vector2d> {
                if (p[2] <= 1e-6) return std::nullopt;
                double u = fx * p[0] / p[2] + cx;
                double v = fy * p[1] / p[2] + cy;
                rot_uv(&u, &v, k, static_cast<double>(img_w),
                       static_cast<double>(img_h));
                if (!std::isfinite(u) || !std::isfinite(v)) return std::nullopt;
                return Eigen::Vector2d(u + pfx, v + pfy);
            };
        } else {
            std::fprintf(stderr, "--boxes 需要 --intr 才能投影切簇\n");
            return 2;
        }
    }

    const PipelineResult r = run_pipeline(pts, args, /*verbose=*/false);
    ordered_json out;
    ordered_json objs = ordered_json::array();
    for (const GeoObject& o : r.objects) objs.push_back(object_json(o));
    out["objects"] = objs;
    out["timing_ms"] = r.timing_ms;
    std::cout << out.dump() << std::endl;
    return 0;
}
