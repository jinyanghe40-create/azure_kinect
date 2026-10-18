#include "kvc/match.hpp"

#include "kvc/json_util.hpp"
#include "kvc/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace kvc {

const char* const kClassNames[kClassCount] = {"tools", "medicine",
                                              "instrument", "food"};
const char* const kClassCn[kClassCount] = {"工具", "药品", "仪器", "食品"};
const Bgr kClassColors[kClassCount] = {{231, 76, 60},
                                       {200, 200, 60},
                                       {60, 140, 240},
                                       {60, 210, 60}};

std::pair<double, double> rot_uv(double u, double v, int k, double w0,
                                 double h0) {
    if (k == 1) return {h0 - 1.0 - v, u};
    if (k == 2) return {w0 - 1.0 - u, h0 - 1.0 - v};
    if (k == 3) return {v, w0 - 1.0 - u};
    return {u, v};
}

std::optional<Eigen::Vector2d> project_to_image(const Vec3& p,
                                                const Intrinsics& intr,
                                                const Extrinsics* extr) {
    Vec3 q = p;
    if (extr) q = extr->R * p + extr->t;
    if (q[2] <= 1e-6) return std::nullopt;
    return Eigen::Vector2d(intr.fx * q[0] / q[2] + intr.cx,
                           intr.fy * q[1] / q[2] + intr.cy);
}

std::string quality_of(int n_face_pts, const std::optional<double>& nearest_m,
                       const std::optional<double>& rms_mm) {
    if (rms_mm && *rms_mm > 5.0) return "COARSE";
    if (n_face_pts >= 1200 && nearest_m && *nearest_m <= 1.05) return "GOOD";
    if (n_face_pts >= 500 && nearest_m && *nearest_m <= 1.30) return "FAIR";
    return "COARSE";
}

std::vector<Detection> dedup_contained(std::vector<Detection> dets,
                                       double cover) {
    std::stable_sort(dets.begin(), dets.end(),
                     [](const Detection& a, const Detection& b) {
                         return a.score > b.score;
                     });
    auto area = [](const Detection& d) {
        return std::max(0.0, d.x1 - d.x0) * std::max(0.0, d.y1 - d.y0);
    };
    std::vector<Detection> kept;
    for (const Detection& d : dets) {
        const double a = area(d);
        bool swallowed = false;
        for (const Detection& k : kept) {
            const double ix0 = std::max(d.x0, k.x0);
            const double iy0 = std::max(d.y0, k.y0);
            const double ix1 = std::min(d.x1, k.x1);
            const double iy1 = std::min(d.y1, k.y1);
            const double inter =
                std::max(0.0, ix1 - ix0) * std::max(0.0, iy1 - iy0);
            if (inter <= 0.0) continue;
            const double karea = area(k);
            if (inter / std::max(1e-6, std::min(a, karea)) >= cover) {
                swallowed = true;
                break;
            }
        }
        if (!swallowed) kept.push_back(d);
    }
    return kept;
}

std::vector<Match> match_classes(const std::vector<Detection>& dets,
                                 const std::vector<GeoObject>& objects,
                                 const Intrinsics& intr,
                                 const Extrinsics* extr, int rot_k,
                                 double px_fix_x, double px_fix_y) {
    std::vector<Match> out;
    out.reserve(objects.size());
    for (const GeoObject& o : objects) {
        // Prefer the cube centre (all three faces fitted), else the main face.
        const bool use_cube = o.cube_center.has_value();
        const Vec3 p3 = use_cube ? *o.cube_center : o.face_center;

        const auto uv0 = project_to_image(p3, intr, extr);
        if (!uv0) continue;
        const auto uv = rot_uv((*uv0)[0], (*uv0)[1], rot_k,
                               static_cast<double>(intr.width),
                               static_cast<double>(intr.height));
        const double u = uv.first + px_fix_x;
        const double v = uv.second + px_fix_y;

        const Detection* best = nullptr;
        for (const Detection& d : dets) {
            if (d.x0 <= u && u <= d.x1 && d.y0 <= v && v <= d.y1) {
                if (!best || d.score > best->score) best = &d;
            }
        }

        Match m;
        m.pos_type = use_cube ? "cube" : "face";
        if (best) {
            m.cls_id = best->cls_id;
            if (best->cls_id >= 0 && best->cls_id < kClassCount) {
                m.cls = kClassNames[best->cls_id];
                m.cn = kClassCn[best->cls_id];
            }
            m.score = python_round(best->score, 3);
        }
        m.n_face_pts = o.n_face_pts;
        m.nearest_m = o.nearest_m;
        m.nearest_pt = o.nearest_pt;
        m.edge_used = o.edge_used_m;
        m.edge_src = o.edge_src;
        m.edge_faces = o.edge_faces_m;
        m.xyz = Vec3(python_round(p3[0], 4), python_round(p3[1], 4),
                     python_round(p3[2], 4));
        m.face_center = o.face_center;
        m.cube_center = o.cube_center;
        m.angle = o.angle_deg;
        m.lean_deg = o.lean_deg;
        m.normal = o.normal;
        for (const Face& f : o.faces) {
            Face c;
            c.normal = f.normal;
            c.center = f.center;
            m.faces.push_back(c);
        }
        m.n_points = o.n_points;
        m.n_faces = o.n_faces;
        m.rms_mm = o.face_rms_mm;
        m.u = u;
        m.v = v;
        out.push_back(std::move(m));
    }
    // Python: out.sort(key=lambda d: -(d["score"] or 0)) — stable.
    std::stable_sort(out.begin(), out.end(),
                     [](const Match& a, const Match& b) {
                         return a.score.value_or(0.0) > b.score.value_or(0.0);
                     });
    return out;
}

namespace {

struct Sph {
    double range = 0.0;
    double az = 0.0;
    double el = 0.0;
};

Sph sph_of(const Vec3& p) {
    Sph s;
    const double r = p.norm();
    if (r < 1e-9) {
        s.range = 0.0;
        s.az = 0.0;
        s.el = 0.0;
        return s;
    }
    s.range = python_round(r, 4);
    s.az = python_round(std::atan2(p[0], p[2]) * 180.0 / M_PI, 2);
    s.el = python_round(std::atan2(-p[1], std::hypot(p[0], p[2])) * 180.0 /
                            M_PI,
                        2);
    return s;
}

ojson face_json(const Vec3& v) { return vec_json(v, 4); }

}  // namespace

ojson serial_obj(const Match& m, const Mat3* R_cam, const Vec3* t_cam) {
    auto cam_point = [&](const Vec3& p) {
        return R_cam ? (*R_cam * p + (t_cam ? *t_cam : Vec3::Zero())) : p;
    };
    auto cam_dir = [&](const Vec3& n) { return R_cam ? (*R_cam * n) : n; };

    ojson o;
    o["cls_id"] = m.cls_id ? ojson(*m.cls_id) : ojson(nullptr);
    o["cls"] = m.cls.empty() ? ojson(nullptr) : ojson(m.cls);
    o["score"] = m.score ? ojson(*m.score) : ojson(nullptr);

    o["face"] = m.face_center ? face_json(*m.face_center) : ojson(nullptr);
    o["normal"] = face_json(m.normal);
    o["cube"] = m.cube_center ? face_json(*m.cube_center) : ojson(nullptr);

    if (m.face_center) {
        const Sph s = sph_of(cam_point(*m.face_center));
        o["range"] = s.range;
        o["az"] = s.az;
        o["el"] = s.el;
    } else {
        o["range"] = nullptr;
        o["az"] = nullptr;
        o["el"] = nullptr;
    }

    o["nearest"] = m.nearest_m ? ojson(*m.nearest_m) : ojson(nullptr);
    o["edge"] = m.edge_used ? ojson(*m.edge_used) : ojson(nullptr);
    o["edge_src"] = m.edge_src.empty() ? ojson(nullptr) : ojson(m.edge_src);
    o["n_faces"] = m.n_faces;
    o["pos_type"] = m.pos_type;
    o["quality"] =
        quality_of(m.n_face_pts.value_or(0), m.nearest_m, m.rms_mm);
    o["n_face_pts"] = m.n_face_pts ? ojson(*m.n_face_pts) : ojson(nullptr);
    o["cube_ok"] = (m.n_faces == 3 && m.edge_src == "auto");

    if (R_cam) {
        const Vec3 nc = cam_dir(m.normal);
        o["angle"] = python_round(
            std::acos(std::min(1.0, std::abs(nc[2]))) * 180.0 / M_PI, 2);
    } else {
        o["angle"] = python_round(m.angle, 2);
    }
    o["rms_mm"] = python_round(m.rms_mm, 2);
    o["lean"] = m.lean_deg ? ojson(*m.lean_deg) : ojson(nullptr);
    return o;
}

ojson to_mm(const ojson& serial) {
    // Python run_azure._to_mm whitelist.  normal / az / el / angle / lean /
    // rms_mm / n_* are never scaled.
    auto is_vec_key = [](const std::string& k) {
        return k == "face" || k == "cube" || k == "xyz" ||
               k == "face_center" || k == "cube_center" || k == "nearest_pt";
    };
    auto is_scalar_key = [](const std::string& k) {
        return k == "range" || k == "nearest" || k == "edge";
    };

    ojson out = ojson::object();
    for (auto it = serial.begin(); it != serial.end(); ++it) {
        const std::string k = it.key();
        const ojson& v = it.value();
        if (v.is_null()) {
            out[k] = nullptr;
        } else if (is_vec_key(k) && v.is_array()) {
            ojson a = ojson::array();
            for (const auto& x : v) {
                a.push_back(python_round(x.get<double>() * 1000.0, 1));
            }
            out[k] = std::move(a);
        } else if (is_scalar_key(k) && v.is_number()) {
            out[k] = python_round(v.get<double>() * 1000.0, 1);
        } else if (k == "faces" && v.is_array()) {
            ojson lst = ojson::array();
            for (const auto& f : v) {
                if (f.is_object() && f.contains("center") &&
                    !f["center"].is_null() && f["center"].is_array()) {
                    ojson g = f;
                    ojson c = ojson::array();
                    for (const auto& x : f["center"]) {
                        c.push_back(
                            python_round(x.get<double>() * 1000.0, 1));
                    }
                    g["center"] = std::move(c);
                    lst.push_back(std::move(g));
                } else {
                    lst.push_back(f);
                }
            }
            out[k] = std::move(lst);
        } else {
            out[k] = v;
        }
    }
    return out;
}

}  // namespace kvc
