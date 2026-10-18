#include "kvc/pc_pipeline.hpp"

#include "kvc/geom_internal.hpp"
#include "kvc/math.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace kvc {
namespace {

using Clk = std::chrono::steady_clock;

constexpr double kPi = 3.14159265358979323846;

double elapsed_ms(const Clk::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
}

std::string fmt_axis(const std::optional<Vec3>& v) {
    if (!v) return "None";
    char buf[128];
    std::snprintf(buf, sizeof(buf), "[%g, %g, %g]", python_round((*v)[0], 3),
                  python_round((*v)[1], 3), python_round((*v)[2], 3));
    return buf;
}

std::string fmt_axis(const Vec3& v) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "[%g, %g, %g]", python_round(v[0], 3),
                  python_round(v[1], 3), python_round(v[2], 3));
    return buf;
}

std::vector<double> round_list(const std::vector<double>& v, int ndigits) {
    std::vector<double> out;
    out.reserve(v.size());
    for (double x : v) out.push_back(python_round(x, ndigits));
    return out;
}

}  // namespace

PipelineResult run_pipeline(const std::vector<Vec3>& pts_in,
                            const PipelineArgs& args, bool verbose) {
    const Clk::time_point t0 = Clk::now();
    auto log = [verbose](const char* fmt, auto... rest) {
        if (verbose) std::printf(fmt, rest...);
    };

    // ------------------------------------------------------------------
    // [0] Resolve the frame once.  Every later use of "up / optical axis /
    // camera position" must go through these three values; literal (0,0,1) or
    // (0,1,0) in this frame is the silent-error pattern the Python code warns
    // about.
    // ------------------------------------------------------------------
    std::optional<Vec3> up_axis;
    if (args.up_axis && args.up_axis->norm() >= 1e-12) {
        up_axis = args.up_axis->normalized();
    }
    Vec3 cam_axis = args.cam_axis ? *args.cam_axis : Vec3(0.0, 0.0, 1.0);
    if (cam_axis.norm() >= 1e-12) cam_axis.normalize();
    const Vec3 cam_origin = args.cam_origin;
    log("[0] 坐标系: 上=%s 光轴=%s 光心=%s\n", fmt_axis(up_axis).c_str(),
        fmt_axis(cam_axis).c_str(), fmt_axis(cam_origin).c_str());

    log("[1] 输入点云: %zu 点\n", pts_in.size());

    // ------------------------------------------------------------------
    // Distance filter: distance to the CAMERA (cam_origin), not the origin.
    // ------------------------------------------------------------------
    std::vector<Vec3> pts;
    pts.reserve(pts_in.size());
    for (const Vec3& p : pts_in) {
        const double d = (p - cam_origin).norm();
        if (d > args.min_dist && d < args.max_dist) pts.push_back(p);
    }
    log("[2] 距离滤波 %g-%gm -> %zu 点\n", args.min_dist, args.max_dist,
        pts.size());

    if (pts.size() < 100) {
        log("[!] 只有 %zu 个点在 %g-%gm 范围内，跳过几何分析\n", pts.size(),
            args.min_dist, args.max_dist);
        log("    常见原因：① 相机对着太远的地方 → 调大 --max-dist\n");
        log("              ② 画面里没有物体/太近 → 调小 --min-dist\n");
        log("              ③ 深度图有问题 → 跑 realsense/test_streams.py 确认\n");
        PipelineResult r;
        r.timing_ms = python_round(elapsed_ms(t0), 1);
        return r;
    }

    // ------------------------------------------------------------------
    // Remove the table: normal roughly aligned with "up" (sign-agnostic).
    // ------------------------------------------------------------------
    const Vec3 table_hint = up_axis ? *up_axis : Vec3(0.0, 1.0, 0.0);
    auto table = geom::ransac_plane(
        pts, args.table_thresh,
        /*iters=*/150, /*min_inliers=*/static_cast<int>(0.20 * pts.size()),
        table_hint, /*cos_tol=*/0.3);
    std::vector<Vec3> pts_obj;
    if (table) {
        size_t removed = 0;
        pts_obj.reserve(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            if (table->mask[i]) {
                ++removed;
            } else {
                pts_obj.push_back(pts[i]);
            }
        }
        log("[3] 去桌面 (阈值%.0fmm): 移除 %zu 点 (%.0f%%)，剩余 %zu 点\n",
            args.table_thresh * 1000.0, removed,
            100.0 * static_cast<double>(removed) / static_cast<double>(pts.size()),
            pts_obj.size());
        log("    桌面法向=[%g, %g, %g]  平面偏移 d=%.4f m\n",
            python_round(table->normal[0], 4), python_round(table->normal[1], 4),
            python_round(table->normal[2], 4), table->d);
    } else {
        pts_obj = pts;
        log("[3] 去桌面: 未找到可靠桌面平面（可能场景里没有）\n");
    }

    if (static_cast<int>(pts_obj.size()) < args.min_cluster_points) {
        PipelineResult r;
        r.timing_ms = python_round(elapsed_ms(t0), 1);
        return r;
    }

    // ------------------------------------------------------------------
    // Voxel clustering (+ optional 2D-box split).
    // ------------------------------------------------------------------
    const int min_pts =
        args.min_cluster_points > 0
            ? args.min_cluster_points
            : std::max(60, static_cast<int>(
                               (args.edge / args.voxel) * (args.edge / args.voxel) * 3));
    std::vector<int> labels = geom::voxel_cluster(pts_obj, args.voxel, min_pts);
    int n_clusters = 0;
    if (!labels.empty()) {
        n_clusters = *std::max_element(labels.begin(), labels.end()) + 1;
    }
    log("[4] 体素聚类 (voxel=%gm, min_pts=%d): %d 个簇\n", args.voxel, min_pts,
        n_clusters);

    if (!args.split_enabled || !args.project || args.split_boxes.empty() ||
        n_clusters == 0) {
        log("[4b] 按框切簇：跳过（投影函数=%s，检测框=%zu 个，簇=%d）\n",
            args.project ? "有" : "缺", args.split_boxes.size(), n_clusters);
    } else {
        std::vector<Eigen::Vector2d> uv(pts_obj.size());
        std::vector<char> uv_valid(pts_obj.size(), 0);
        bool projected = true;
        try {
            for (size_t i = 0; i < pts_obj.size(); ++i) {
                const auto p = args.project(pts_obj[i]);
                if (p) {
                    uv[i] = *p;
                    uv_valid[i] = 1;
                }
            }
        } catch (const std::exception& e) {
            log("[4b] 按检测框切簇：投影失败（%s），跳过\n", e.what());
            projected = false;
        }
        if (projected) {
            int n_new = n_clusters;
            labels = geom::split_clusters_by_boxes(
                pts_obj, labels, n_clusters, uv, uv_valid, args.split_boxes,
                std::max(60, min_pts / 4), &n_new, verbose);
            n_clusters = n_new;
        }
    }

    // ------------------------------------------------------------------
    // Per-cluster analysis.
    // ------------------------------------------------------------------
    std::vector<GeoObject> objects;
    int n_skipped = 0;
    for (int cid = 0; cid < n_clusters; ++cid) {
        std::vector<Vec3> cp;
        for (size_t i = 0; i < pts_obj.size(); ++i) {
            if (labels[i] == cid) cp.push_back(pts_obj[i]);
        }
        if (cp.empty()) continue;

        // Fast pre-filter: a cube's bounding box cannot be much larger than
        // 4*edge; desks/walls would otherwise eat seconds of RANSAC.
        Vec3 lo = cp.front();
        Vec3 hi = cp.front();
        for (const Vec3& p : cp) {
            lo = lo.cwiseMin(p);
            hi = hi.cwiseMax(p);
        }
        const Vec3 ext = hi - lo;
        if (ext.maxCoeff() > args.edge * 4.0) {
            ++n_skipped;
            continue;
        }

        // Adaptive plane threshold from the depth-noise model sigma_z ~ c*z^2.
        // z is the depth ALONG THE OPTICAL AXIS, not cp[:,2].
        std::vector<double> z_vals;
        z_vals.reserve(cp.size());
        for (const Vec3& p : cp) {
            z_vals.push_back(std::max((p - cam_origin).dot(cam_axis), 0.1));
        }
        const double z_med = geom::median(z_vals);
        const double sig_z = args.depth_noise_coef * z_med * z_med;
        const double thresh_use = std::max(args.plane_thresh, 2.0 * sig_z);

        std::vector<geom::ClusterFace> faces =
            geom::analyze_cluster(cp, thresh_use, args.max_faces, cam_origin);
        if (faces.empty()) continue;

        size_t main_i = 0;
        if (args.face_select == FaceSelect::MostFacing) {
            double best = -std::numeric_limits<double>::infinity();
            for (size_t k = 0; k < faces.size(); ++k) {
                const double v = std::abs(faces[k].normal.dot(cam_axis));
                if (v > best) {  // first maximum wins, like Python's min()
                    best = v;
                    main_i = k;
                }
            }
        }
        const geom::ClusterFace& main = faces[main_i];

        const double cos_angle = std::min(1.0, std::abs(main.normal.dot(cam_axis)));
        const double angle = std::acos(cos_angle) * 180.0 / kPi;

        GeoObject obj;
        obj.id = static_cast<int>(objects.size());
        obj.face_center = main.center;
        obj.normal = main.normal;
        obj.angle_deg = python_round(angle, 2);
        obj.n_points = static_cast<int>(cp.size());
        obj.n_faces = static_cast<int>(faces.size());
        obj.face_rms_mm = python_round(main.rms * 1000.0, 2);
        if (up_axis) {
            const double lean = std::asin(std::min(
                1.0, std::abs(main.normal.dot(*up_axis))));
            obj.lean_deg = python_round(lean * 180.0 / kPi, 2);
        }
        obj.pts = cp;

        obj.faces.reserve(faces.size());
        for (const geom::ClusterFace& f : faces) {
            Face out;
            out.normal = f.normal;
            out.center = f.center;
            out.n_points = f.n_points;
            out.rms_mm = python_round(f.rms * 1000.0, 2);
            out.angle_deg = python_round(
                std::acos(std::min(1.0, std::abs(f.normal.dot(cam_axis)))) *
                    180.0 / kPi,
                2);
            if (up_axis) {
                out.lean_deg = python_round(
                    std::asin(std::min(1.0, std::abs(f.normal.dot(*up_axis)))) *
                        180.0 / kPi,
                    2);
            }
            out.extent = f.extent;
            obj.faces.push_back(out);
        }

        // Edge length: measured on the faces when possible, --edge otherwise.
        std::optional<double> edge_meas;
        std::vector<double> edge_vals;
        std::string edge_src;
        geom::measure_edge(faces, cp, /*tol=*/0.005, /*min_pts=*/50,
                           /*spread_tol=*/0.25, &edge_meas, &edge_vals,
                           &edge_src);
        double edge_use;
        if (args.edge_auto && edge_meas) {
            edge_use = *edge_meas;
        } else {
            edge_use = args.edge;
            edge_src = "cli";
        }
        if (verbose) {
            std::printf("      边长: 各面实测 [");
            for (size_t k = 0; k < edge_vals.size(); ++k) {
                std::printf("%s%g", k ? ", " : "",
                            python_round(edge_vals[k] * 100.0, 1));
            }
            std::printf("] cm → 采用 %.1f cm（来源 %s", edge_use * 100.0,
                        edge_src.c_str());
            if (edge_src == "cli") {
                std::printf("，拿命令行兜底，不可信");
            } else {
                std::printf("，--edge 填的是 %.0f cm", args.edge * 100.0);
            }
            std::printf("\n");
        }

        obj.cube_center = geom::cube_center_from_faces(faces, edge_use, cp);

        obj.n_face_pts = main.n_points;

        // Distance to the CAMERA; nearest_pt is what a tape measure would hit.
        double nearest = std::numeric_limits<double>::infinity();
        size_t near_i = 0;
        for (size_t i = 0; i < cp.size(); ++i) {
            const double d = (cp[i] - cam_origin).norm();
            if (d < nearest) {
                nearest = d;
                near_i = i;
            }
        }
        obj.nearest_m = python_round(nearest, 4);
        obj.nearest_pt = cp[near_i];
        obj.plane_thresh_used_mm = python_round(thresh_use * 1000.0, 2);
        obj.sigma_z_mm = python_round(sig_z * 1000.0, 2);
        obj.edge_used_m = python_round(edge_use, 4);
        obj.edge_src = edge_src;
        obj.edge_faces_m = round_list(edge_vals, 4);

        objects.push_back(std::move(obj));
        log("    簇%d: %zu 点, %zu 个面, 面心=[%g, %g, %g], 法向=[%g, %g, %g], "
            "夹角=%.1f°, RMS=%.1fmm\n",
            cid, cp.size(), faces.size(),
            python_round(main.center[0], 4), python_round(main.center[1], 4),
            python_round(main.center[2], 4),
            python_round(main.normal[0], 3), python_round(main.normal[1], 3),
            python_round(main.normal[2], 3), angle, main.rms * 1000.0);
    }

    if (n_skipped) {
        log("    （跳过了 %d 个过大的簇：包围盒 > %.0fcm，多半是桌面/墙）\n",
            n_skipped, args.edge * 4.0 * 100.0);
    }

    PipelineResult r;
    r.objects = std::move(objects);
    r.timing_ms = python_round(elapsed_ms(t0), 1);
    return r;
}

}  // namespace kvc
