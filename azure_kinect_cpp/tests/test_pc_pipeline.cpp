// Geometry pipeline tests.
//
// The oracle comes from common/pc_pipeline.py (tests/gen_geom_fixtures.py):
// each scene is a float32 point cloud plus the Python run_pipeline output.
// RANSAC draws a different random stream in C++ on purpose (the plan compares
// with tolerances, not bit-for-bit), so fields are matched nearest-neighbour
// and checked against physical tolerances.

#include "kvc/math.hpp"
#include "kvc/pc_pipeline.hpp"
#include "test_common.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

using namespace kvc;
using nlohmann::json;

#ifndef KVC_FIXTURE_DIR
#define KVC_FIXTURE_DIR "tests/fixtures"
#endif

namespace {

void test_empty_input() {
    PipelineArgs args;
    const PipelineResult r = run_pipeline({}, args, false);
    CHECK(r.objects.empty());
    CHECK(r.timing_ms >= 0.0);
}

void test_short_input_early_exit() {
    // Fewer than 100 points: the Python pipeline returns no objects.
    std::vector<Vec3> pts;
    for (int i = 0; i < 50; ++i) pts.emplace_back(0.01 * i, 0.0, 0.8);
    PipelineArgs args;
    const PipelineResult r = run_pipeline(pts, args, false);
    CHECK(r.objects.empty());
}

std::vector<Vec3> read_pts(const std::string& path, size_t n) {
    std::vector<float> buf(n * 3);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(buf.size() * sizeof(float)));
    std::vector<Vec3> pts(n);
    for (size_t i = 0; i < n; ++i) {
        pts[i] = Vec3(buf[i * 3 + 0], buf[i * 3 + 1], buf[i * 3 + 2]);
    }
    return pts;
}

Vec3 vec3_from_json(const json& j) {
    return Vec3(j[0].get<double>(), j[1].get<double>(), j[2].get<double>());
}

std::vector<Detection> read_boxes(const std::string& path) {
    std::ifstream f(path);
    json j;
    f >> j;
    std::vector<Detection> boxes;
    for (const auto& e : j) {
        Detection d;
        d.x0 = e["x0"].get<double>();
        d.y0 = e["y0"].get<double>();
        d.x1 = e["x1"].get<double>();
        d.y1 = e["y1"].get<double>();
        d.cls_id = e.value("cls", 0);
        d.score = e.value("score", 0.0);
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

PipelineArgs args_from_json(const json& j) {
    PipelineArgs a;
    a.edge = j.at("edge").get<double>();
    a.edge_auto = j.at("edge_auto").get<bool>();
    a.min_dist = j.at("min_dist").get<double>();
    a.max_dist = j.at("max_dist").get<double>();
    a.plane_thresh = j.at("plane_thresh").get<double>();
    a.table_thresh = j.at("table_thresh").get<double>();
    a.voxel = j.at("voxel").get<double>();
    a.min_cluster_points = j.at("min_cluster_points").get<int>();
    a.max_faces = j.at("max_faces").get<int>();
    a.depth_noise_coef = j.at("depth_noise_coef").get<double>();
    a.face_select = (j.at("face_select").get<std::string>() == "largest")
                        ? FaceSelect::Largest
                        : FaceSelect::MostFacing;
    if (!j.at("up_axis").is_null()) a.up_axis = vec3_from_json(j.at("up_axis"));
    if (!j.at("cam_axis").is_null()) a.cam_axis = vec3_from_json(j.at("cam_axis"));
    if (!j.at("cam_origin").is_null())
        a.cam_origin = vec3_from_json(j.at("cam_origin"));
    return a;
}

double angle_diff(double a, double b) {
    return std::abs(a - b);
}

void compare_scene(const json& entry, const std::string& dir) {
    const std::string name = entry["name"].get<std::string>();
    const size_t n = entry["n"].get<size_t>();
    const std::vector<Vec3> pts =
        read_pts(dir + "/" + entry["pts"].get<std::string>(), n);

    PipelineArgs args = args_from_json(entry["args"]);
    if (entry.contains("boxes")) {
        args.split_boxes = read_boxes(dir + "/" + entry["boxes"].get<std::string>());
        const auto intr = entry["intr"];
        const auto size = entry["size"];
        const int k = entry["rotate"].get<int>() / 90;
        const double pfx = entry["px_fix"][0].get<double>();
        const double pfy = entry["px_fix"][1].get<double>();
        const double fx = intr[0].get<double>(), fy = intr[1].get<double>();
        const double cx = intr[2].get<double>(), cy = intr[3].get<double>();
        const double w = size[0].get<double>(), h = size[1].get<double>();
        args.project = [=](const Vec3& p) -> std::optional<Eigen::Vector2d> {
            if (p[2] <= 1e-6) return std::nullopt;
            double u = fx * p[0] / p[2] + cx;
            double v = fy * p[1] / p[2] + cy;
            rot_uv(&u, &v, k, w, h);
            if (!std::isfinite(u) || !std::isfinite(v)) return std::nullopt;
            return Eigen::Vector2d(u + pfx, v + pfy);
        };
    }

    json expected;
    {
        std::ifstream f(dir + "/" + entry["expected"].get<std::string>());
        f >> expected;
    }
    const PipelineResult got = run_pipeline(pts, args, /*verbose=*/false);

    CHECK_MSG(got.objects.size() == expected.size(),
              name + ": object count " + std::to_string(got.objects.size()) +
                  " vs " + std::to_string(expected.size()));
    if (got.objects.size() != expected.size()) return;

    // Greedy nearest-neighbour matching on the main face centre.  Different
    // RANSAC draws move a face centre by a few mm on noisy scenes, so the
    // tolerances below describe algorithm stability, not just porting error.
    constexpr double kCenterTol = 0.006;   // 6 mm
    constexpr double kNormalTol = 0.030;
    constexpr double kAngleTol = 1.5;      // degrees
    constexpr double kEdgeTol = 0.002;     // 2 mm
    std::vector<char> used(got.objects.size(), 0);
    for (const auto& eo : expected) {
        const Vec3 ec = vec3_from_json(eo["face_center_m"]);
        size_t best_i = 0;
        double best_d = 1e9;
        for (size_t i = 0; i < got.objects.size(); ++i) {
            if (used[i]) continue;
            const double d = (got.objects[i].face_center - ec).norm();
            if (d < best_d) {
                best_d = d;
                best_i = i;
            }
        }
        used[best_i] = 1;
        const GeoObject& go = got.objects[best_i];
        CHECK_MSG(best_d <= kCenterTol,
                  name + ": face centre " + std::to_string(best_d * 1000.0) + " mm");
        const Vec3 en = vec3_from_json(eo["normal"]);
        CHECK_MSG((go.normal - en).cwiseAbs().maxCoeff() <= kNormalTol,
                  name + ": normal");
        CHECK_MSG(angle_diff(go.angle_deg, eo["angle_deg"].get<double>()) <= kAngleTol,
                  name + ": angle");
        CHECK_MSG(go.n_faces == eo["n_faces"].get<int>(), name + ": n_faces");
        CHECK_MSG(go.edge_src == eo["edge_src"].get<std::string>(),
                  name + ": edge_src");
        CHECK_MSG(std::abs(go.edge_used_m - eo["edge_used_m"].get<double>()) <= kEdgeTol,
                  name + ": edge");
        CHECK_MSG(std::abs(go.nearest_m - eo["nearest_m"].get<double>()) <= kCenterTol,
                  name + ": nearest");
        CHECK_MSG(std::abs(go.face_rms_mm - eo["face_rms_mm"].get<double>()) <=
                      0.10 * std::max(1.0, eo["face_rms_mm"].get<double>()),
                  name + ": rms");

        if (eo.contains("cube_center_m") && go.cube_center) {
            const double d = (*go.cube_center - vec3_from_json(eo["cube_center_m"])).norm();
            CHECK_MSG(d <= kCenterTol, name + ": cube centre");
        }
        if (!eo["lean_deg"].is_null() && go.lean_deg) {
            CHECK_MSG(angle_diff(*go.lean_deg, eo["lean_deg"].get<double>()) <= kAngleTol,
                      name + ": lean");
        } else {
            CHECK(eo["lean_deg"].is_null() == !go.lean_deg.has_value());
        }
        const int en_pts = eo["n_points"].get<int>();
        CHECK_MSG(std::abs(go.n_points - en_pts) <=
                      std::max(20, static_cast<int>(0.03 * en_pts)),
                  name + ": n_points");
    }
}

void test_python_fixtures() {
    const std::string dir = std::string(KVC_FIXTURE_DIR) + "/geom";
    std::ifstream f(dir + "/manifest.json");
    if (!f) {
        std::cout << "  (geometry fixtures absent; run tests/gen_geom_fixtures.py)"
                  << std::endl;
        return;
    }
    json manifest;
    f >> manifest;
    for (const auto& entry : manifest) {
        compare_scene(entry, dir);
    }
}

}  // namespace

int main() {
    test_empty_input();
    test_short_input_early_exit();
    test_python_fixtures();
    return kvc_test::finish("test_pc_pipeline");
}
