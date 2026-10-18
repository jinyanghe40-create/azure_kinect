// Production loop.  Port of azure_kinect/run_azure.py main(), with the same
// stage order, the same log text, and the same JSONL contract.

#include "kvc/app_run.hpp"

#include "kvc/azure_source.hpp"
#include "kvc/body_calib.hpp"
#include "kvc/json_util.hpp"
#include "kvc/match.hpp"
#include "kvc/math.hpp"
#include "kvc/pc_pipeline.hpp"
#include "kvc/preprocess.hpp"
#include "kvc/serial_out.hpp"
#include "kvc/ssd_decoder.hpp"
#include "kvc/trt_engine.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace kvc {
namespace {

using Clk = std::chrono::steady_clock;

constexpr int kImgSize = 320;

volatile std::sig_atomic_t g_stop = 0;

void on_signal(int) { g_stop = 1; }

double ms_since(const Clk::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
}

double seconds_since(const Clk::time_point& t0) {
    return std::chrono::duration<double>(Clk::now() - t0).count();
}

int depth_mode_value(const std::string& name) {
    if (name == "nfov2x2") return 1;
    if (name == "wfov") return 3;
    if (name == "wfov2x2") return 4;
    return 2;  // nfov
}

int color_res_value(const std::string& name) {
    if (name == "1080p") return 2;
    if (name == "1440p") return 3;
    if (name == "1536p") return 4;
    if (name == "2160p") return 5;
    if (name == "off") return 0;
    return 1;  // 720p
}

bool mkdir_p(const std::string& path) {
    if (path.empty()) return false;
    struct stat st {};
    if (::stat(path.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
    size_t pos = path.find('/', 1);
    while (pos != std::string::npos) {
        const std::string part = path.substr(0, pos);
        ::mkdir(part.c_str(), 0755);
        pos = path.find('/', pos + 1);
    }
    return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

struct Stage {
    const char* name;
    double sum = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = 0.0;
    int n = 0;
    void add(double ms) {
        sum += ms;
        min = std::min(min, ms);
        max = std::max(max, ms);
        ++n;
    }
    double avg() const { return n ? sum / n : 0.0; }
};

ojson opt_num(const std::optional<double>& v) {
    return v ? ojson(*v) : ojson(nullptr);
}

// Python's json.dumps(..., ensure_ascii=False) defaults: `", "` and `": "`
// separators, single line.  nlohmann's compact dump has none, so insert the
// spaces outside string literals.
std::string python_dumps(const ojson& j) {
    const std::string compact = j.dump();
    std::string out;
    out.reserve(compact.size() + compact.size() / 8);
    bool in_string = false;
    bool escaped = false;
    for (const char c : compact) {
        if (in_string) {
            out.push_back(c);
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            out.push_back(c);
        } else if (c == ',') {
            out += ", ";
        } else if (c == ':') {
            out += ": ";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

}  // namespace

int run_app(const RunOptions& opt) {
    if (opt.engine.empty() || !std::ifstream(opt.engine)) {
        std::printf("[ERROR] 找不到 engine: %s\n", opt.engine.c_str());
        return 1;
    }
    if (!std::getenv("DISPLAY")) {
        std::printf(
            "[警告] 没有 DISPLAY —— Azure Kinect 深度引擎需要 X/GLX，"
            "会报 error 204\n");
        std::printf(
            "       用: sudo bash ~/work/azure_kinect/run_with_x.sh "
            "./run_azure_cpp ...\n");
    }

    // ---- body calibration: read, validate, refuse to start on failure ----
    std::optional<BodyCalib> calib;
    try {
        calib = load_body_calib(opt.calib, opt.rotate, opt.allow_uncalibrated);
    } catch (const BodyCalibError& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    if (opt.mm) {
        std::printf(
            "[单位] --mm：对象输出用【毫米】。"
            "（点云仍是米 —— 只有 objects 字典变 mm，头里带 obj_unit）\n");
    }

    std::unique_ptr<AzureSource> src;
    try {
        SourceConfig cfg;
        cfg.depth_mode = depth_mode_value(opt.depth_mode);
        cfg.color_resolution = color_res_value(opt.color_res);
        cfg.color = opt.color;
        cfg.rotate = opt.rotate;
        cfg.read_gravity = opt.gravity;
        cfg.verbose = true;
        src.reset(new AzureSource(cfg, calib));
    } catch (const CaptureError& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    std::printf("[TRT] 加载 engine…\n");
    std::unique_ptr<Engine> engine;
    try {
        engine = make_engine(opt.engine);
    } catch (const InferError& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    const std::vector<float> anchors = generate_anchors(kImgSize);
    SsdConfig ssd;
    ssd.score_thresh = static_cast<float>(opt.score);

    // ---- geometry arguments ----
    PipelineArgs gargs;
    gargs.edge = opt.edge;
    gargs.edge_auto = opt.edge_auto;
    gargs.min_dist = opt.min_dist;
    gargs.max_dist = opt.max_dist;
    gargs.plane_thresh = opt.plane_thresh;
    gargs.table_thresh = opt.table_thresh;
    gargs.voxel = opt.voxel;
    gargs.min_cluster_points = opt.min_cluster_points;
    gargs.max_faces = opt.max_faces;
    gargs.depth_noise_coef = opt.depth_noise_coef;
    gargs.face_select = (opt.face_select == "largest") ? FaceSelect::Largest
                                                       : FaceSelect::MostFacing;
    if (calib) {
        gargs.up_axis = Vec3(0.0, 0.0, 1.0);
    } else if (src->gravity_ok()) {
        gargs.up_axis = Vec3(0.0, 1.0, 0.0);
    }
    gargs.cam_axis = src->R_out().col(2);
    gargs.cam_origin = src->t_out();

    gargs.split_enabled = opt.split_boxes;
    if (opt.split_boxes) {
        const Intrinsics intr = src->intr_color();
        const Extrinsics extr = src->extrinsics();
        const double pfx = src->px_fix_x();
        const double pfy = src->px_fix_y();
        // The cloud was rotated together with the color image, so rot_k is 0.
        gargs.project = [intr, extr, pfx, pfy](
                            const Vec3& p) -> std::optional<Eigen::Vector2d> {
            const Vec3 q = extr.R * p + extr.t;
            if (q[2] <= 1e-6) return std::nullopt;
            double u = intr.fx * q[0] / q[2] + intr.cx;
            double v = intr.fy * q[1] / q[2] + intr.cy;
            const auto r = rot_uv(u, v, 0, static_cast<double>(intr.width),
                                  static_cast<double>(intr.height));
            u = r.first + pfx;
            v = r.second + pfy;
            if (!std::isfinite(u) || !std::isfinite(v)) return std::nullopt;
            return Eigen::Vector2d(u, v);
        };
    }

    // ---- serial ----
    std::unique_ptr<SerialOut> ser;
    if (!opt.serial.empty() || opt.serial_dry) {
        ser.reset(new SerialOut(opt.serial_dry ? std::string() : opt.serial,
                                opt.baud, opt.hz));
    }

    // Top-level header.  coord/rev tell the controller which frame the
    // positions are in; obj_unit only describes obj[].
    ojson top_extra;
    top_extra["coord"] = calib ? "body" : "camera";
    top_extra["rev"] = 2;
    if (calib) {
        top_extra["cam"] = vec_json(src->t_out(), 4);
    } else {
        top_extra["cam"] = ojson::array({0.0, 0.0, 0.0});
    }
    top_extra["cam_unit"] = "m";
    top_extra["calib"] = calib ? ojson(calib->id) : ojson(nullptr);
    top_extra["obj_unit"] = opt.mm ? "mm" : "m";

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    long long n = 0;
    const Clk::time_point t_start = Clk::now();
    Clk::time_point t_loop = t_start;
    double t_inf = 0.0;
    double t_geo = 0.0;

    Stage p_get{"取帧/点云"}, p_pre{"预处理"}, p_trt{"TensorRT"},
        p_post{"检测后处理"}, p_geo{"几何"}, p_match{"配对"}, p_total{"总计"};

    int rc = 0;
    while (!g_stop && (opt.frames == 0 || n < opt.frames)) {
        const Clk::time_point t_frame = Clk::now();
        std::optional<FrameData> frame;
        try {
            frame = src->get(opt.stride);
        } catch (const CaptureError& e) {
            std::fprintf(stderr, "%s\n", e.what());
            rc = 1;
            break;
        }
        if (!frame) continue;
        ++n;
        if (n == 1) t_loop = Clk::now();
        p_get.add(ms_since(t_frame));

        if (frame->color_bgr.empty()) {
            std::printf("[ERROR] 没拿到彩色图（--no-color？）\n");
            break;
        }
        const int W = frame->color_bgr.cols;
        const int H = frame->color_bgr.rows;

        // Keep only valid points, exactly like `pts[valid]` in Python.
        std::vector<Vec3> pts_valid;
        pts_valid.reserve(frame->points.size());
        for (size_t i = 0; i < frame->points.size(); ++i) {
            if (frame->valid[i]) pts_valid.push_back(frame->points[i]);
        }

        // ---- ① 2D detection ----
        const Clk::time_point t_det = Clk::now();
        const Clk::time_point t_pre = Clk::now();
        const std::vector<float> input = preprocess_bgr(
            frame->color_bgr.data, W, H,
            static_cast<int>(frame->color_bgr.step), kImgSize);
        p_pre.add(ms_since(t_pre));

        std::vector<float> cls_logits;
        std::vector<float> bbox_reg;
        const Clk::time_point t_trt = Clk::now();
        engine->infer(input, &cls_logits, &bbox_reg);
        p_trt.add(ms_since(t_trt));

        const Clk::time_point t_post = Clk::now();
        std::vector<Detection> dets =
            postprocess(cls_logits, bbox_reg, anchors, ssd);
        const double sx = W / static_cast<double>(kImgSize);
        const double sy = H / static_cast<double>(kImgSize);
        for (Detection& d : dets) {
            d.x0 *= sx;
            d.y0 *= sy;
            d.x1 *= sx;
            d.y1 *= sy;
        }
        if (opt.dedup) {
            const size_t n0 = dets.size();
            dets = dedup_contained(std::move(dets));
            if (opt.geo_verbose && dets.size() != n0) {
                std::printf("[dets] 吞并抑制：%zu 框 → %zu 框\n", n0,
                            dets.size());
            }
        }
        p_post.add(ms_since(t_post));
        t_inf = ms_since(t_det);

        // ---- ② 3D geometry ----
        const Clk::time_point t_geo0 = Clk::now();
        gargs.split_boxes = opt.split_boxes ? dets : std::vector<Detection>();
        const PipelineResult geo = run_pipeline(pts_valid, gargs, opt.geo_verbose);
        p_geo.add(ms_since(t_geo0));
        t_geo = ms_since(t_geo0);

        // ---- ③ pairing ----
        const Clk::time_point t_match0 = Clk::now();
        const Mat3& R_cam = src->R_cam();
        const Vec3& t_cam = src->t_cam();
        std::vector<Match> objs = match_classes(
            dets, geo.objects, src->intr_color(), &src->extrinsics(), 0,
            src->px_fix_x(), src->px_fix_y());
        if (opt.only_classified) {
            objs.erase(std::remove_if(objs.begin(), objs.end(),
                                      [](const Match& o) { return o.cls.empty(); }),
                       objs.end());
        }
        p_match.add(ms_since(t_match0));

        // ---- ④ output ----
        if (opt.json_out) {
            ojson rec;
            rec["frame"] = n;
            rec["t"] = python_round(seconds_since(t_start), 3);
            rec["infer_ms"] = python_round(t_inf, 1);
            rec["geo_ms"] = python_round(t_geo, 1);
            rec["w"] = W;
            rec["h"] = H;
            rec["n"] = objs.size();
            ojson arr = ojson::array();
            for (const Match& o : objs) {
                ojson j;
                j["cls"] = o.cls.empty() ? ojson(nullptr) : ojson(o.cls);
                j["cn"] = o.cn.empty() ? ojson(nullptr) : ojson(o.cn);
                j["score"] = opt_num(o.score);
                j["xyz"] = vec_json(o.xyz, 4);
                j["angle"] = o.angle;  // already rounded by the geometry stage
                j["lean"] = opt_num(o.lean_deg);
                j["pos_type"] = o.pos_type;
                arr.push_back(std::move(j));
            }
            rec["obj"] = std::move(arr);
            std::printf("%s\n", python_dumps(rec).c_str());
            std::fflush(stdout);
        } else if (opt.show_text) {
            const double k = opt.mm ? 1000.0 : 1.0;
            const char* unit = opt.mm ? "mm" : "m";
            std::string body;
            for (const Match& o : objs) {
                const Vec3 p = o.face_center.value_or(o.xyz);
                char xyz[160];
                if (opt.mm) {
                    std::snprintf(xyz, sizeof(xyz), "%.0f,%.0f,%.0f",
                                  p[0] * k, p[1] * k, p[2] * k);
                } else {
                    std::snprintf(xyz, sizeof(xyz), "%.3f,%.3f,%.3f",
                                  p[0] * k, p[1] * k, p[2] * k);
                }
                char item[512];
                std::snprintf(item, sizeof(item),
                              "%s%s %.2f 面心=(%s)%s 夹角=%.1f°",
                              o.cn.empty() ? "?" : o.cn.c_str(),
                              o.cls.c_str(), o.score.value_or(0.0), xyz, unit,
                              o.angle);
                std::string s(item);
                if (o.lean_deg) {
                    char lean[64];
                    std::snprintf(lean, sizeof(lean), " 倾角=%.1f°",
                                  *o.lean_deg);
                    s += lean;
                }
                if (!body.empty()) body += " | ";
                body += s;
            }
            if (body.empty()) body = "-";
            std::printf("[%05lld] 检测%5.1fms 几何%6.1fms obj=%zu  %s\n", n,
                        t_inf, t_geo, objs.size(), body.c_str());
            std::fflush(stdout);
        }

        if (ser) {
            std::vector<ojson> ser_objs;
            ser_objs.reserve(objs.size());
            for (const Match& o : objs) {
                ojson j = serial_obj(o, &R_cam, &t_cam);
                if (opt.mm) j = to_mm(j);
                ser_objs.push_back(std::move(j));
            }
            ser->send(static_cast<uint64_t>(n), std::move(ser_objs), t_inf,
                      top_extra);
        }

        if (!opt.save_dir.empty() && n % opt.save_every == 0) {
            mkdir_p(opt.save_dir);
            cv::Mat vis = frame->color_bgr.clone();
            for (const Detection& d : dets) {
                const Bgr c = kClassColors[((d.cls_id % kClassCount) +
                                            kClassCount) %
                                           kClassCount];
                cv::rectangle(vis, cv::Point(static_cast<int>(d.x0),
                                             static_cast<int>(d.y0)),
                              cv::Point(static_cast<int>(d.x1),
                                        static_cast<int>(d.y1)),
                              cv::Scalar(c.b, c.g, c.r), 3);
            }
            char name[64];
            std::snprintf(name, sizeof(name), "/f%05lld.jpg", n);
            cv::imwrite(opt.save_dir + name, vis);
        }

        p_total.add(ms_since(t_frame));
    }

    if (ser) ser->close();
    src.reset();

    const double dt = n ? std::max(1e-6, seconds_since(t_loop)) : 1.0;
    std::printf("\n[DONE] %lld 帧  %.1f FPS（不含启动）  "
                "（检测 %.1fms + 几何 %.1fms）\n",
                n, static_cast<double>(n) / dt, t_inf, t_geo);

    if (opt.profile && n > 0) {
        std::printf("[profile] 帧数 %lld\n", n);
        for (const Stage* s : {&p_get, &p_pre, &p_trt, &p_post, &p_geo,
                               &p_match, &p_total}) {
            if (!s->n) continue;
            std::printf("[profile] %-10s avg=%6.1f  min=%6.1f  max=%6.1f ms\n",
                        s->name, s->avg(), s->min, s->max);
        }
    }
    return rc;
}

}  // namespace kvc
