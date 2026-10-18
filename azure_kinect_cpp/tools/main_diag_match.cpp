// diag_match_cpp — port of azure_kinect/diag_match.py.
// One frame through detection -> geometry -> pairing, printing every stage so
// "obj=0" can be localised.  Writes /tmp/diag_match.jpg.

#include "kvc/azure_source.hpp"
#include "kvc/body_calib.hpp"
#include "kvc/cli.hpp"
#include "kvc/diag_common.hpp"
#include "kvc/match.hpp"
#include "kvc/pc_pipeline.hpp"
#include "kvc/preprocess.hpp"
#include "kvc/ssd_decoder.hpp"
#include "kvc/trt_engine.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr int kImgSize = 320;

struct Opts {
    int rotate = 180;
    std::string calib;
    bool allow_uncalibrated = false;
    int stride = 2;
    double min_dist = 0.3;
    double max_dist = 1.5;
    std::string engine;
};

void usage(const char* argv0) {
    std::printf(
        "用法: %s [--rotate 0|90|180|270] [--calib PATH]\n"
        "          [--allow-uncalibrated] [--stride N] "
        "[--min-dist M] [--max-dist M] [--engine PATH]\n",
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
        else if (a == "--stride") o->stride = std::atoi(next().c_str());
        else if (a == "--min-dist") o->min_dist = std::atof(next().c_str());
        else if (a == "--max-dist") o->max_dist = std::atof(next().c_str());
        else if (a == "--engine") o->engine = next();
        else {
            std::printf("未知参数: %s\n", a.c_str());
            return false;
        }
    }
    if (o->engine.empty()) o->engine = kvc::default_engine_path();
    return true;
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

    std::printf("[TRT] 加载 engine…\n");
    std::fflush(stdout);
    std::unique_ptr<kvc::Engine> eng;
    try {
        eng = kvc::make_engine(a.engine);
    } catch (const kvc::InferError& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    const std::vector<float> anchors = kvc::generate_anchors(kImgSize);
    std::printf("[TRT] 好了\n");
    std::fflush(stdout);

    std::optional<kvc::BodyCalib> calib;
    try {
        calib = kvc::load_body_calib(a.calib, a.rotate, a.allow_uncalibrated);
    } catch (const kvc::BodyCalibError& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

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

    std::optional<kvc::FrameData> frame;
    try {
        frame = src->get(a.stride);
    } catch (const kvc::CaptureError& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (!frame) {
        std::printf("❌ 取帧失败\n");
        return 1;
    }
    const cv::Mat& color = frame->color_bgr;
    if (color.empty()) {
        std::printf("❌ 没有彩色图（不能关掉彩色）\n");
        return 1;
    }
    const int W = color.cols;
    const int H = color.rows;
    std::printf("\n彩色图 %dx%d   rotate_k=0\n", W, H);

    // ---------- ① 2D 检测 ----------
    const std::vector<float> input =
        kvc::preprocess_bgr(color.data, W, H, static_cast<int>(color.step),
                            kImgSize);
    std::vector<float> cls_logits;
    std::vector<float> bbox_reg;
    try {
        eng->infer(input, &cls_logits, &bbox_reg);
    } catch (const kvc::InferError& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    kvc::SsdConfig ssd;
    std::vector<kvc::Detection> dets =
        kvc::postprocess(cls_logits, bbox_reg, anchors, ssd);
    const double sx = W / static_cast<double>(kImgSize);
    const double sy = H / static_cast<double>(kImgSize);
    for (kvc::Detection& d : dets) {
        d.x0 *= sx;
        d.y0 *= sy;
        d.x1 *= sx;
        d.y1 *= sy;
    }

    std::printf("\n① 2D 检测: %zu 个框\n", dets.size());
    for (const kvc::Detection& d : dets) {
        const char* name = (d.cls_id >= 0 && d.cls_id < kvc::kClassCount)
                               ? kvc::kClassNames[d.cls_id]
                               : "?";
        std::printf("   %-12s %.2f 框=(%6.0f,%6.0f)-(%6.0f,%6.0f)  "
                    "中心=(%6.0f,%6.0f)\n",
                    name, d.score, d.x0, d.y0, d.x1, d.y1, (d.x0 + d.x1) / 2.0,
                    (d.y0 + d.y1) / 2.0);
    }

    // ---------- ② 3D 几何 ----------
    const std::vector<kvc::Vec3> pts = kvc::valid_points(*frame);
    const kvc::PipelineArgs gargs =
        kvc::diag_pipeline_args(*src, calib, a.min_dist, a.max_dist);
    const kvc::PipelineResult res = kvc::run_pipeline(pts, gargs, false);

    std::printf("\n② 3D 几何: %zu 个物体\n", res.objects.size());
    for (const kvc::GeoObject& o : res.objects) {
        const kvc::Vec3 cube = o.cube_center.value_or(kvc::Vec3::Zero());
        std::printf("   %d点 %d面 面心=(%.3f,%.3f,%.3f) "
                    "体心=(%.3f,%.3f,%.3f) 夹角=%.2f°\n",
                    o.n_points, o.n_faces, o.face_center.x(),
                    o.face_center.y(), o.face_center.z(), cube.x(), cube.y(),
                    cube.z(), o.angle_deg);
    }

    // ---------- ③ 配对 ----------
    std::printf("\n③ 配对（把 3D 点投回彩色图，看落在哪个框里）\n");
    const kvc::Intrinsics& intr = src->intr_color();
    const kvc::Extrinsics& extr = src->extrinsics();
    const double pfx = src->px_fix_x();
    const double pfy = src->px_fix_y();

    auto project = [&](const kvc::Vec3& p3,
                       double* u, double* v) -> bool {
        const auto uv0 = kvc::project_to_image(p3, intr, &extr);
        if (!uv0) return false;
        const auto uv = kvc::rot_uv((*uv0)[0], (*uv0)[1], 0,
                                    static_cast<double>(intr.width),
                                    static_cast<double>(intr.height));
        *u = uv.first + pfx;
        *v = uv.second + pfy;
        return true;
    };

    cv::Mat vis = color.clone();
    for (const kvc::Detection& d : dets) {
        cv::rectangle(vis,
                      cv::Point(static_cast<int>(d.x0), static_cast<int>(d.y0)),
                      cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                      cv::Scalar(0, 255, 0), 3);
    }

    for (const kvc::GeoObject& o : res.objects) {
        const kvc::Vec3 p3 = o.cube_center ? *o.cube_center : o.face_center;
        double u = 0.0;
        double v = 0.0;
        if (!project(p3, &u, &v)) {
            std::printf("   投影失败\n");
            continue;
        }
        const kvc::Detection* hit = nullptr;
        for (const kvc::Detection& d : dets) {
            if (d.x0 <= u && u <= d.x1 && d.y0 <= v && v <= d.y1) {
                if (!hit || d.score > hit->score) hit = &d;
            }
        }
        std::printf("   3D 点 (%.3f,%.3f,%.3f) → 像素 (%6.1f,%6.1f)  ",
                    p3.x(), p3.y(), p3.z(), u, v);
        if (hit) {
            const char* name = (hit->cls_id >= 0 && hit->cls_id < kvc::kClassCount)
                                   ? kvc::kClassNames[hit->cls_id]
                                   : "?";
            std::printf("✅ 落在 %s (%.2f) 框里\n", name, hit->score);
        } else {
            std::printf("❌ 不在任何检测框里\n");
        }
        cv::drawMarker(vis, cv::Point(static_cast<int>(u), static_cast<int>(v)),
                       cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 40, 4);
    }

    cv::imwrite("/tmp/diag_match.jpg", vis);
    std::printf("\n✅ 已存 /tmp/diag_match.jpg（绿框=检测框，红叉=3D 投影）\n");
    std::printf("   判读：红叉应该在绿框里面。不在就是配对断了。\n");
    return 0;
}
