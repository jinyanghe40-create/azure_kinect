// Locks the JSONL wire contract: key order, compact encoding, seq semantics
// and the rate limiter.  The 20-key object mapping is produced by
// serial_obj(); this file asserts its exact order and the --mm whitelist.

#include "kvc/json_util.hpp"
#include "kvc/match.hpp"
#include "kvc/serial_out.hpp"
#include "test_common.hpp"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

using namespace kvc;

namespace {

// Redirect stdin's sibling fd 1 to a file for the lifetime of the capture.
class StdoutCapture {
public:
    explicit StdoutCapture(const std::string& path) : path_(path) {
        std::cout.flush();
        std::fflush(stdout);
        saved_ = ::dup(STDOUT_FILENO);
        file_ = std::fopen(path.c_str(), "w");
        if (file_) ::dup2(::fileno(file_), STDOUT_FILENO);
    }
    ~StdoutCapture() { restore(); }

    std::string restore() {
        if (saved_ < 0) return {};
        std::cout.flush();
        std::fflush(stdout);
        ::dup2(saved_, STDOUT_FILENO);
        ::close(saved_);
        saved_ = -1;
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
        std::ifstream f(path_);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

private:
    std::string path_;
    int saved_ = -1;
    FILE* file_ = nullptr;
};

std::vector<std::string> json_lines(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.front() == '{') out.push_back(line);
    }
    return out;
}

const std::vector<std::string> kObjKeys = {
    "cls_id", "cls",      "score",    "face",    "normal",   "cube",
    "range",  "az",       "el",       "nearest", "edge",     "edge_src",
    "n_faces", "pos_type", "quality", "n_face_pts", "cube_ok", "angle",
    "rms_mm", "lean"};

ojson sample_object() {
    // Same 20 keys, same order as the Python serial_obj().
    ojson o;
    o["cls_id"] = 2;
    o["cls"] = "instrument";
    o["score"] = 0.91;
    o["face"] = ojson::array({1.5, 1.9, 0.005});
    o["normal"] = ojson::array({0.0, 0.0, -1.0});
    o["cube"] = ojson::array({1.5, 1.9, 0.13});
    o["range"] = 2.42;
    o["az"] = -4.35;
    o["el"] = -3.08;
    o["nearest"] = 2.31;
    o["edge"] = 0.25;
    o["edge_src"] = "auto";
    o["n_faces"] = 3;
    o["pos_type"] = "cube";
    o["quality"] = "FAIR";
    o["n_face_pts"] = 1340;
    o["cube_ok"] = true;
    o["angle"] = 2.54;
    o["rms_mm"] = 2.1;
    o["lean"] = 89.7;
    return o;
}

ojson sample_extra() {
    ojson e;
    e["coord"] = "body";
    e["rev"] = 2;
    e["cam"] = "depth_optical";
    e["cam_unit"] = "m";
    e["calib"] = "test";
    e["obj_unit"] = "mm";
    return e;
}

void test_wire_format() {
    const std::string path = "/tmp/kvc_serial_contract.jsonl";
    std::string text;
    uint64_t sent = 0;
    {
        StdoutCapture cap(path);
        SerialOut out("", 115200, 0.0, false);
        for (int i = 0; i < 5; ++i) {
            std::vector<ojson> objs{sample_object()};
            out.send(static_cast<uint64_t>(100 + i), std::move(objs), 12.34,
                     sample_extra());
            // Let the writer drain so the 4-slot queue never drops.
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (out.sent() < static_cast<uint64_t>(i + 1) &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        sent = out.sent();
        out.close();
        text = cap.restore();
    }
    std::remove(path.c_str());

    CHECK(sent == 5u);
    const auto lines = json_lines(text);
    CHECK(lines.size() == 5u);
    if (lines.size() != 5u) return;

    const ojson first = ojson::parse(lines[0]);
    std::vector<std::string> top_keys;
    for (auto it = first.begin(); it != first.end(); ++it) top_keys.push_back(it.key());
    const std::vector<std::string> want_top = {"t", "frame", "seq", "n", "obj",
                                               "infer_ms", "coord", "rev", "cam",
                                               "cam_unit", "calib", "obj_unit"};
    CHECK(top_keys == want_top);
    CHECK(first["n"] == 1);
    CHECK(first["frame"] == 100);
    CHECK(first["seq"] == 1);
    CHECK(first["infer_ms"] == 12.3);
    CHECK(first["obj_unit"] == "mm");

    std::vector<std::string> obj_keys;
    for (auto it = first["obj"][0].begin(); it != first["obj"][0].end(); ++it) {
        obj_keys.push_back(it.key());
    }
    CHECK(obj_keys.size() == 20u);
    CHECK(obj_keys == kObjKeys);

    for (size_t i = 0; i < lines.size(); ++i) {
        const ojson rec = ojson::parse(lines[i]);
        CHECK(rec["seq"] == static_cast<int>(i + 1));
        CHECK(rec["frame"] == static_cast<int>(100 + i));
    }

    // Compact encoding: no whitespace outside strings.
    CHECK(lines[0].find(' ') == std::string::npos);
    CHECK(lines[0].find(": ") == std::string::npos);
}

void test_rate_limit_keeps_seq() {
    const std::string path = "/tmp/kvc_serial_rate.jsonl";
    std::string text;
    {
        StdoutCapture cap(path);
        SerialOut out("", 115200, 0.5, false);  // 2 s between sends
        out.send(1, {sample_object()}, std::nullopt, sample_extra());
        out.send(2, {sample_object()}, std::nullopt, sample_extra());
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (out.sent() < 1u &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(out.sent() == 1u);
        out.close();
        text = cap.restore();
    }
    std::remove(path.c_str());
    const auto lines = json_lines(text);
    CHECK(lines.size() == 1u);
    if (!lines.empty()) {
        const ojson rec = ojson::parse(lines[0]);
        CHECK(rec["frame"] == 1);
        CHECK(rec["seq"] == 1);
        CHECK(!rec.contains("infer_ms"));
    }
}

void test_rounding_in_json() {
    CHECK(compact_dump(vec_json(Vec3(0.1234567, -0.00004, 2.0), 4)) ==
          "[0.1235,-0.0,2.0]");
    CHECK(compact_dump(opt_json(std::nullopt, 3)) == "null");
    CHECK(compact_dump(opt_json(std::optional<double>(1.5), 0)) == "2.0");
    CHECK(compact_dump(opt_vec_json(std::nullopt, 3)) == "null");
}

void test_serial_obj_contract() {
    Match m;
    m.cls_id = 2;
    m.cls = "instrument";
    m.cn = "仪器";
    m.score = 0.876;
    m.face_center = Vec3(1.5, 1.9, 0.005);
    m.normal = Vec3(0.0, 0.0, -1.0);
    m.cube_center = Vec3(1.5, 1.9, 0.13);
    m.nearest_m = 1.2;
    m.edge_used = 0.25;
    m.edge_src = "auto";
    m.n_faces = 3;
    m.pos_type = "cube";
    m.n_face_pts = 1300;
    m.rms_mm = 2.5;
    m.angle = 5.0;
    m.lean_deg = 89.0;

    const ojson o = serial_obj(m, nullptr, nullptr);
    std::vector<std::string> keys;
    for (auto it = o.begin(); it != o.end(); ++it) keys.push_back(it.key());
    CHECK(keys == kObjKeys);

    CHECK(o["cls_id"] == 2);
    CHECK(o["cls"] == "instrument");
    CHECK(o["score"] == 0.876);
    CHECK(o["pos_type"] == "cube");
    CHECK(o["edge_src"] == "auto");
    CHECK(o["n_faces"] == 3);
    CHECK(o["quality"] == "FAIR");  // 1300 pts, 1.2 m -> not GOOD
    CHECK(o["n_face_pts"] == 1300);
    CHECK(o["cube_ok"] == true);
    CHECK(o["angle"] == 5.0);
    CHECK(o["rms_mm"] == 2.5);
    CHECK(o["lean"] == 89.0);

    // --mm whitelist: lengths x1000, angles / counts untouched.
    const ojson mm = to_mm(o);
    CHECK(compact_dump(mm["face"]) == "[1500.0,1900.0,5.0]");
    CHECK(compact_dump(mm["cube"]) == "[1500.0,1900.0,130.0]");
    CHECK(compact_dump(mm["normal"]) == "[0.0,0.0,-1.0]");
    CHECK(mm["range"] == python_round(o["range"].get<double>() * 1000.0, 1));
    CHECK(mm["nearest"] == 1200.0);
    CHECK(mm["edge"] == 250.0);
    CHECK(mm["az"] == o["az"]);
    CHECK(mm["el"] == o["el"]);
    CHECK(mm["angle"] == 5.0);
    CHECK(mm["rms_mm"] == 2.5);
    CHECK(mm["n_face_pts"] == 1300);
    CHECK(mm["cube_ok"] == true);
}

void test_serial_obj_camera_angle() {
    // With R_cam, angle is recomputed from the camera-frame normal; positions
    // stay in the output (body) frame and only range/az/el are re-based.
    Match m;
    m.cls = "tools";
    m.face_center = Vec3(0.0, 0.0, 1.0);
    m.normal = Vec3(0.0, 1.0, 0.0);
    m.n_faces = 1;
    m.pos_type = "face";
    m.n_face_pts = 600;
    m.nearest_m = 1.0;
    m.edge_used = 0.25;
    m.edge_src = "cli";
    const Mat3 R_cam = Mat3::Identity();
    const Vec3 t_cam(0.0, 0.0, 0.0);
    const ojson o = serial_obj(m, &R_cam, &t_cam);
    CHECK(o["angle"] == 90.0);           // |n·z| = 0
    CHECK(compact_dump(o["normal"]) == "[0.0,1.0,0.0]");
    CHECK(o["range"] == 1.0);
    CHECK(o["cube_ok"] == false);
    CHECK(o["quality"] == "FAIR");
}

}  // namespace

int main() {
    test_wire_format();
    test_rate_limit_keeps_seq();
    test_rounding_in_json();
    test_serial_obj_contract();
    test_serial_obj_camera_angle();
    return kvc_test::finish("test_serial_contract");
}
