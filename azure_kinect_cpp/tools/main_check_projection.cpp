// check_projection_cpp — port of azure_kinect/check_projection.py.
// Projects the measured face/cube centres back into the colour image so a
// wrong sign or frame is visible at a glance.  Writes /tmp/proj_check.jpg.

#include "kvc/azure_source.hpp"
#include "kvc/body_calib.hpp"
#include "kvc/diag_common.hpp"
#include "kvc/match.hpp"
#include "kvc/pc_pipeline.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

struct Opts {
    int rotate = 180;
    std::string calib;
    bool allow_uncalibrated = false;
    int stride = 2;
};

void usage(const char* argv0) {
    std::printf("用法: %s [--rotate 0|90|180|270] [--calib PATH] "
                "[--allow-uncalibrated] [--stride N]\n",
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
        else {
            std::printf("未知参数: %s\n", a.c_str());
            return false;
        }
    }
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
        std::printf("❌ 没有彩色图\n");
        return 1;
    }

    const std::vector<kvc::Vec3> pts = kvc::valid_points(*frame);
    const kvc::PipelineArgs gargs = kvc::diag_pipeline_args(*src, calib);
    const kvc::PipelineResult res = kvc::run_pipeline(pts, gargs, false);
    if (res.objects.empty()) {
        std::printf("❌ 没检出物体 —— 方块在画面里吗？\n");
        return 1;
    }
    const kvc::GeoObject* best = &res.objects.front();
    for (const kvc::GeoObject& o : res.objects) {
        if (o.n_points > best->n_points) best = &o;
    }
    const kvc::Vec3& fc = best->face_center;
    const std::optional<kvc::Vec3>& cc = best->cube_center;

    const kvc::Intrinsics& ic = src->intr_color();
    const kvc::Extrinsics& extr = src->extrinsics();
    const double pfx = src->px_fix_x();
    const double pfy = src->px_fix_y();

    std::printf("\n");
    std::printf("==================================================================\n");
    std::printf(" 簇 %d 点  %d 面  夹角 %.2f°\n", best->n_points, best->n_faces,
                best->angle_deg);
    std::printf(" 面心 (X,Y,Z) = (%+.4f, %+.4f, %+.4f) m   |P| = %.4f\n",
                fc.x(), fc.y(), fc.z(), fc.norm());
    if (cc) {
        std::printf(" 体心 (X,Y,Z) = (%+.4f, %+.4f, %+.4f) m\n", cc->x(), cc->y(),
                    cc->z());
    }

    cv::Mat vis = color.clone();
    struct Item {
        const char* name;
        const std::optional<kvc::Vec3>* p;
        cv::Scalar color;
    };
    const std::optional<kvc::Vec3> face_opt = fc;
    const Item items[2] = {{"face", &face_opt, cv::Scalar(0, 0, 255)},
                           {"cube", &cc, cv::Scalar(255, 0, 255)}};
    for (const Item& it : items) {
        if (!it.p->has_value()) continue;
        const auto uv0 = kvc::project_to_image(**it.p, ic, &extr);
        if (!uv0) {
            std::printf(" %s: 投影失败（在相机后面）\n", it.name);
            continue;
        }
        const auto uv = kvc::rot_uv((*uv0)[0], (*uv0)[1], 0,
                                    static_cast<double>(ic.width),
                                    static_cast<double>(ic.height));
        const double u = uv.first + pfx;
        const double v = uv.second + pfy;
        const bool inside = 0 <= u && u < ic.width && 0 <= v && v < ic.height;
        std::printf(" %s: 投影像素 = (%.1f, %.1f)   画面 %dx%d    %s\n", it.name,
                    u, v, ic.width, ic.height,
                    inside ? "✅ 在画面内" : "❌ 出界");
        if (0 <= u && u < vis.cols && 0 <= v && v < vis.rows) {
            cv::drawMarker(vis, cv::Point(static_cast<int>(u),
                                          static_cast<int>(v)),
                           it.color, cv::MARKER_CROSS, 40, 4);
            cv::putText(vis, it.name,
                        cv::Point(static_cast<int>(u) + 22,
                                  static_cast<int>(v) - 12),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, it.color, 3);
        }
    }

    // 画面中心（光轴）
    const int cu = ic.width / 2;
    const int cvv = ic.height / 2;
    cv::drawMarker(vis, cv::Point(cu, cvv), cv::Scalar(0, 255, 0),
                   cv::MARKER_TILTED_CROSS, 30, 2);
    cv::putText(vis, "optical axis", cv::Point(cu + 18, cvv + 26),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    const char* out = "/tmp/proj_check.jpg";
    cv::imwrite(out, vis);
    std::printf("\n✅ 已存 %s（绿十字=光轴中心，红=面心，品红=体心）\n", out);
    std::printf("   判读：红十字应该正好落在方块朝向相机的那个面上\n");
    std::printf("==================================================================\n");
    return 0;
}
