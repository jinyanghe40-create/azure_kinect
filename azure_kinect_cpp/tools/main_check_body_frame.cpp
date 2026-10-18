// check_body_frame_cpp — port of azure_kinect/check_body_frame.py.
// Acceptance C/D/F with the same thresholds and the same reading aids.
//
// NOTE on C/F: the Python tool reads `o.get("cube_center")`, but the pipeline
// emits `cube_center_m` (and `face_center_m`).  Both lookups are therefore
// None and the Python version raises IndexError.  This port uses the intended
// field: cube_center when it exists, otherwise face_center.

#include "kvc/azure_source.hpp"
#include "kvc/body_calib.hpp"
#include "kvc/diag_common.hpp"
#include "kvc/pc_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

std::vector<std::string> g_fails;

void verdict(const char* name, const std::vector<std::string>& diag) {
    std::printf("\n  ┌─ %s 失败时怎么读 ────────────────────────────\n", name);
    for (const auto& line : diag) std::printf("  │ %s\n", line.c_str());
    std::printf("  └──────────────────────────────────────────────────\n");
}

bool ok(const std::string& tag, bool cond, const std::string& detail = "") {
    std::printf("  %-40s %s  %s\n", tag.c_str(), cond ? "✅" : "❌",
                detail.c_str());
    if (!cond) g_fails.push_back(tag);
    return cond;
}

void press_enter(const char* prompt) {
    std::printf("%s", prompt);
    std::fflush(stdout);
    std::string line;
    std::getline(std::cin, line);
}

struct Opts {
    int rotate = 180;
    std::string calib;
    bool allow_uncalibrated = false;
    std::string only = "all";
    double tape_fwd = 1.00;
    double tape_left = 0.00;
    double also_far = 0.0;
    double tol_mm = 15.0;
    int n_tape_pts = 3;
    int stride = 2;
    double min_dist = 0.3;
    double max_dist = 1.5;
    double voxel = 0.02;
    double edge = 0.25;
    double plane_thresh = 0.003;
    double table_thresh = 0.010;
    int max_faces = 3;
    int min_cluster_points = 200;
    std::string face_select = "most_facing";
};

void usage(const char* argv0) {
    std::printf(
        "用法: %s [--rotate 0|90|180|270] [--calib PATH] [--only all|c|d|f]\n"
        "          [--tape-fwd M] [--tape-left M] [--also-far M] "
        "[--tol-mm MM]\n"
        "          [--n-tape-pts N] [--stride N] [--min-dist M] "
        "[--max-dist M]\n"
        "          [--voxel M] [--edge M] [--plane-thresh M] "
        "[--table-thresh M]\n"
        "          [--max-faces N] [--min-cluster-points N] "
        "[--face-select M] [--allow-uncalibrated]\n",
        argv0);
}

bool parse(int argc, char** argv, Opts* o) {
    if (const char* env = std::getenv("K4A_BODY_CALIB")) o->calib = env;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) return {};
            return argv[++i];
        };
        if (a == "-h" || a == "--help") return false;
        if (a == "--rotate") o->rotate = std::atoi(next().c_str());
        else if (a == "--calib") o->calib = next();
        else if (a == "--allow-uncalibrated") o->allow_uncalibrated = true;
        else if (a == "--only") o->only = next();
        else if (a == "--tape-fwd") o->tape_fwd = std::atof(next().c_str());
        else if (a == "--tape-left") o->tape_left = std::atof(next().c_str());
        else if (a == "--also-far") o->also_far = std::atof(next().c_str());
        else if (a == "--tol-mm") o->tol_mm = std::atof(next().c_str());
        else if (a == "--n-tape-pts") o->n_tape_pts = std::atoi(next().c_str());
        else if (a == "--stride") o->stride = std::atoi(next().c_str());
        else if (a == "--min-dist") o->min_dist = std::atof(next().c_str());
        else if (a == "--max-dist") o->max_dist = std::atof(next().c_str());
        else if (a == "--voxel") o->voxel = std::atof(next().c_str());
        else if (a == "--edge") o->edge = std::atof(next().c_str());
        else if (a == "--plane-thresh") o->plane_thresh = std::atof(next().c_str());
        else if (a == "--table-thresh") o->table_thresh = std::atof(next().c_str());
        else if (a == "--max-faces") o->max_faces = std::atoi(next().c_str());
        else if (a == "--min-cluster-points")
            o->min_cluster_points = std::atoi(next().c_str());
        else if (a == "--face-select") o->face_select = next();
        else {
            std::printf("未知参数: %s\n", a.c_str());
            return false;
        }
    }
    return true;
}

kvc::PipelineArgs gargs(const Opts& a, const kvc::AzureSource& src,
                        const std::optional<kvc::BodyCalib>& calib) {
    const kvc::FaceSelect fs = (a.face_select == "largest")
                                   ? kvc::FaceSelect::Largest
                                   : kvc::FaceSelect::MostFacing;
    return kvc::diag_pipeline_args(src, calib, a.min_dist, a.max_dist, a.voxel,
                                   a.edge, a.plane_thresh, a.table_thresh,
                                   a.max_faces, a.min_cluster_points, fs);
}

// C: the cube placed at a tape-measured point in front of the vehicle.
void check_ground_target(kvc::AzureSource& src, const Opts& a,
                         const std::optional<kvc::BodyCalib>& calib) {
    std::printf("\n── C. 地面标点（车正前方 %.2f m，横向 %+.2f m） ──\n",
                a.tape_fwd, a.tape_left);
    std::printf("   ★ 把方块放在那儿再按回车。位置用卷尺从【车体几何中心】量。\n");
    press_enter("   放好了按回车… ");
    const auto r = src.get(a.stride);
    if (!r) {
        ok("取帧", false, "取帧失败");
        return;
    }
    const auto res = kvc::run_pipeline(kvc::valid_points(*r), gargs(a, src, calib),
                                       false);
    if (res.objects.empty()) {
        ok("检出物体", false, "没检出 —— 方块在画面里吗？");
        return;
    }
    const kvc::GeoObject* best = &res.objects.front();
    for (const auto& o : res.objects) {
        if (o.n_points > best->n_points) best = &o;
    }
    const kvc::Vec3 p = best->cube_center ? *best->cube_center : best->face_center;

    const double d_fwd = p[0] - a.tape_fwd;
    const double d_left = p[1] - a.tape_left;
    std::printf("   实测车体系 (X,Y,Z) = (%+.4f, %+.4f, %+.4f) m\n", p[0], p[1],
                p[2]);
    std::printf("   真值               = (%+.4f, %+.4f, Z?) m\n", a.tape_fwd,
                a.tape_left);
    std::printf("   前向误差 %+.1f mm   横向误差 %+.1f mm\n", d_fwd * 1000.0,
                d_left * 1000.0);

    const double tol = a.tol_mm / 1000.0;
    char detail[64];
    std::snprintf(detail, sizeof(detail), "|Δ|=%.1fmm", std::fabs(d_fwd) * 1000.0);
    ok("前向 X 在容差内", std::fabs(d_fwd) < tol, detail);
    std::snprintf(detail, sizeof(detail), "|Δ|=%.1fmm", std::fabs(d_left) * 1000.0);
    ok("横向 Y 在容差内", std::fabs(d_left) < tol, detail);

    const bool flipped_x =
        std::fabs(p[0] + a.tape_fwd) < tol && a.tape_fwd > 0.3;
    const bool flipped_y =
        std::fabs(p[1] + a.tape_left) < tol && std::fabs(a.tape_left) > 0.1;
    if (flipped_x) {
        std::printf("   ❗ X 反号且量值对得上 —— 这是 180° 重复计入的指纹\n");
    }
    if (flipped_y) {
        std::printf("   ❗ Y 反号且量值对得上 —— 这是 180° 重复计入的指纹\n");
    }
    verdict("C", {
                      "X≈−真值 或 Y≈−真值 且量值吻合 → 标定矩阵里已含 180°，",
                      "   代码又绕光轴乘了一次 Rz。查 azure_capture._finish_rotations()",
                      "   的 `if self.R_body is not None: self.R_out = self.R_body` 分支",
                      "   （有标定时【不能】再乘 Rz）。",
                      "X/Y 交换 → 标定文件的 R 行列写反了（转置）。",
                      "都对但 Z 差一个固定值 → height_mm 量错（原点高度）。",
                  });

    if (a.also_far > 0.0) {
        std::printf("\n   再放一个在 %.2f m…\n", a.also_far);
        press_enter("   放好了按回车… ");
        const auto r2 = src.get(a.stride);
        if (r2) {
            const auto res2 = kvc::run_pipeline(kvc::valid_points(*r2),
                                                gargs(a, src, calib), false);
            if (!res2.objects.empty()) {
                const kvc::GeoObject* b2 = &res2.objects.front();
                for (const auto& o : res2.objects) {
                    if (o.n_points > b2->n_points) b2 = &o;
                }
                const kvc::Vec3 p2 =
                    b2->cube_center ? *b2->cube_center : b2->face_center;
                const double ratio = p2[0] / std::max(a.also_far, 1e-6);
                std::printf("   实测 X=%+.4f m，真值 %.2f m，比值 %.4f（应≈1.000）\n",
                            p2[0], a.also_far, ratio);
                char detail[32];
                std::snprintf(detail, sizeof(detail), "比值 %.4f", ratio);
                ok("尺度线性（远点）", std::fabs(ratio - 1.0) < 0.05, detail);
            }
        }
    }
}

// D: IMU absolute cross-check of R_out.
void check_imu(kvc::AzureSource& src, const Opts&) {
    std::printf("\n── D. IMU 绝对交叉验证（车摆水平）────────────────\n");
    std::printf("   离线的、读一次就够 —— 不违反『运行时不读 IMU』。\n");
    std::string err;
    const auto up_opt = src.read_up_depth(&err);
    if (!up_opt) {
        ok("读到 IMU", false, "IMU 读不到：" + err);
        return;
    }
    const kvc::Vec3 up = (*up_opt) / std::max(up_opt->norm(), 1e-12);
    const kvc::Vec3 got = src.R_out() * up;
    const kvc::Vec3 want(0.0, 0.0, 1.0);
    const double dot = std::max(-1.0, std::min(1.0, got.dot(want)));
    const double ang = std::acos(dot) * 180.0 / M_PI;
    std::printf("   IMU 的『上』在深度系 = [%.5f, %.5f, %.5f]\n", up[0], up[1],
                up[2]);
    std::printf("   R_out·up（车体系）   = [%.5f, %.5f, %.5f]\n", got[0], got[1],
                got[2]);
    std::printf("   与 (0,0,1) 的夹角     = %.3f°\n", ang);
    char detail[32];
    std::snprintf(detail, sizeof(detail), "%.3f°", ang);
    ok("残差 < 0.5°", ang < 0.5, detail);
    verdict("D", {
                      "落在 (0,0,−1)（夹角 ≈180°）→ 180° 重复计入，或 roll 写成 0。",
                      "夹角 ≈ 2 倍相机俯仰（~19°）→ 矩阵转置写反（det 仍为 +1，",
                      "   所以只有这一关能抓住）。",
                      "方向偏但角度不大 → 卷尺量的 pitch/yaw 有误差，回 body_calib",
                      "   用 IMU 反推 pitch 重写标定（--pitch-from-target）。",
                  });
}

// F: three tape-marked floor points.
void check_tape_points(kvc::AzureSource& src, const Opts& a,
                       const std::optional<kvc::BodyCalib>& calib) {
    std::printf("\n── F. 卷尺绝对比对（地板标 3 个已知点）──────────\n");
    std::printf("   ★ 在地板上用胶带标 3 个点，量出它们相对车体几何中心的\n");
    std::printf("     前向/横向坐标，然后逐个把方块放上去测。\n");
    std::vector<double> errs;
    for (int i = 0; i < a.n_tape_pts; ++i) {
        std::printf("  第%d点 前向(m),横向(m)（回车跳过）: ", i + 1);
        std::fflush(stdout);
        std::string raw;
        if (!std::getline(std::cin, raw)) break;
        while (!raw.empty() &&
               (raw.back() == '\r' || raw.back() == ' ' || raw.back() == '\t')) {
            raw.pop_back();
        }
        if (raw.empty()) break;
        // Accept both ASCII and full-width commas.
        std::string norm;
        for (size_t k = 0; k < raw.size();) {
            if (static_cast<unsigned char>(raw[k]) == 0xEF &&
                k + 2 < raw.size() &&
                static_cast<unsigned char>(raw[k + 1]) == 0xBC &&
                static_cast<unsigned char>(raw[k + 2]) == 0x8C) {
                norm += ',';
                k += 3;
            } else {
                norm += raw[k++];
            }
        }
        double f = 0.0;
        double l = 0.0;
        if (std::sscanf(norm.c_str(), "%lf,%lf", &f, &l) != 2) {
            std::printf("    格式不对，跳过\n");
            continue;
        }
        std::printf("    把方块放到第%d点，按回车… ", i + 1);
        std::fflush(stdout);
        std::string line;
        std::getline(std::cin, line);
        const auto r = src.get(a.stride);
        if (!r) continue;
        const auto res = kvc::run_pipeline(kvc::valid_points(*r),
                                           gargs(a, src, calib), false);
        if (res.objects.empty()) {
            std::printf("    ❌ 没检出\n");
            continue;
        }
        const kvc::GeoObject* best = &res.objects.front();
        for (const auto& o : res.objects) {
            if (o.n_points > best->n_points) best = &o;
        }
        const kvc::Vec3 p =
            best->cube_center ? *best->cube_center : best->face_center;
        const double e = std::hypot(p[0] - f, p[1] - l);
        std::printf("    第%d点 实测(%+.4f,%+.4f) 真值(%+.4f,%+.4f)  |Δ|=%.1f mm\n",
                    i + 1, p[0], p[1], f, l, e * 1000.0);
        errs.push_back(e);
    }
    if (errs.empty()) {
        std::printf("   ⏭  一个点都没测，跳过\n");
        return;
    }
    double worst = 0.0;
    for (double e : errs) worst = std::max(worst, e * 1000.0);
    char detail[64];
    std::snprintf(detail, sizeof(detail), "%.1f mm", worst);
    ok("3 点最大绝对误差 < 15mm", worst < 15.0, detail);
}

bool wants_help(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "-h" || s == "--help") return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    Opts a;
    if (wants_help(argc, argv)) {
        usage(argv[0]);
        return 0;
    }
    if (!parse(argc, argv, &a)) {
        usage(argv[0]);
        return 2;
    }

    std::printf("═══ 车体坐标系物理验收 C/D/F ═══\n");
    std::optional<kvc::BodyCalib> calib;
    try {
        calib = kvc::load_body_calib(a.calib, a.rotate, a.allow_uncalibrated);
    } catch (const kvc::BodyCalibError& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (!calib) {
        std::printf("\n❌ 没有标定文件 —— C/D/F 验的正是标定本身，没它毫无意义。\n");
        return 1;
    }
    std::printf("   标定 %s  R=\n", calib->id.c_str());
    for (int i = 0; i < 3; ++i) {
        std::printf("   [%+.5f %+.5f %+.5f]\n", calib->R(i, 0), calib->R(i, 1),
                    calib->R(i, 2));
    }
    std::printf("   t = [%.4f, %.4f, %.4f] m\n", calib->t[0], calib->t[1],
                calib->t[2]);

    kvc::SourceConfig cfg;
    cfg.rotate = a.rotate;
    cfg.verbose = true;
    std::unique_ptr<kvc::AzureSource> src;
    try {
        src.reset(new kvc::AzureSource(cfg, calib));
    } catch (const kvc::CaptureError& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    if (a.only == "all" || a.only == "d") check_imu(*src, a);
    if (a.only == "all" || a.only == "c") check_ground_target(*src, a, calib);
    if (a.only == "all" || a.only == "f") check_tape_points(*src, a, calib);

    std::printf("\n════════════════════════════════════════════════════\n");
    if (!g_fails.empty()) {
        std::printf("❌ %zu 条失败：\n", g_fails.size());
        for (const auto& t : g_fails) std::printf("   · %s\n", t.c_str());
        std::printf("\n  ⚠️ 这几条没过【不要上车】—— 它们抓的都是静默错误。\n");
        return 1;
    }
    std::printf("✅ 全绿（C/D/F 都过）\n");
    return 0;
}
