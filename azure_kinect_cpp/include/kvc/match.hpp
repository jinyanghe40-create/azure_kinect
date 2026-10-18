#pragma once

#include "kvc/body_calib.hpp"
#include "kvc/types.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace kvc {

// ---------------------------------------------------------------------------
// Classes: same ids / names / colours as common/trt_infer.py CFG.
// ---------------------------------------------------------------------------
inline constexpr int kClassCount = 4;
extern const char* const kClassNames[kClassCount];
extern const char* const kClassCn[kClassCount];

struct Bgr {
    int b, g, r;
};
extern const Bgr kClassColors[kClassCount];

// ---------------------------------------------------------------------------
// Image projection (run_realsense.py project_to_image / rot_uv)
// ---------------------------------------------------------------------------
std::pair<double, double> rot_uv(double u, double v, int k, double w0,
                                 double h0);

// Depth-frame point -> original color pixel.  extr is depth->color; pass
// nullptr for an already-aligned cloud.
std::optional<Eigen::Vector2d> project_to_image(const Vec3& p,
                                                const Intrinsics& intr,
                                                const Extrinsics* extr);

// quality_of(): GOOD / FAIR / COARSE.  rms_mm is in millimetres.
std::string quality_of(int n_face_pts, const std::optional<double>& nearest_m,
                       const std::optional<double>& rms_mm);

// ---------------------------------------------------------------------------
// Match: one geometry object paired with a 2D detection.
// Mirrors the dict built by run_realsense.match_classes().
// ---------------------------------------------------------------------------
struct Match {
    std::optional<int> cls_id;
    std::string cls;  // empty when no box contains the projection
    std::string cn;
    std::optional<double> score;  // already rounded to 3
    std::string pos_type;         // "cube" | "face"

    Vec3 xyz = Vec3::Zero();  // rounded to 4
    std::optional<Vec3> face_center;
    std::optional<Vec3> cube_center;
    Vec3 normal = Vec3::Zero();

    std::optional<int> n_face_pts;
    std::optional<double> nearest_m;
    Vec3 nearest_pt = Vec3::Zero();
    std::optional<double> edge_used;
    std::string edge_src;
    std::vector<double> edge_faces;

    double angle = 0.0;  // GeoObject angle_deg (output frame)
    std::optional<double> lean_deg;
    std::vector<Face> faces;  // normal + center, nothing else used

    int n_points = 0;
    int n_faces = 0;
    double rms_mm = 0.0;

    double u = 0.0;  // projected pixel in the rotated color image
    double v = 0.0;
};

// Stable descending-score sort, like Python's sorted(key=lambda t: -t[5]).
std::vector<Detection> dedup_contained(std::vector<Detection> dets,
                                       double cover = 0.80);

std::vector<Match> match_classes(const std::vector<Detection>& dets,
                                 const std::vector<GeoObject>& objects,
                                 const Intrinsics& intr,
                                 const Extrinsics* extr, int rot_k,
                                 double px_fix_x, double px_fix_y);

// ---------------------------------------------------------------------------
// Serial contract (run_realsense.serial_obj / run_azure._to_mm)
// ---------------------------------------------------------------------------
// R_cam/t_cam map output-frame positions to the camera optical frame:
//   p_cam = R_cam * p + t_cam.  Pass nullptr for identity/zero.
ojson serial_obj(const Match& m, const Mat3* R_cam, const Vec3* t_cam);

// Whitelisted length fields -> millimetres (Python round(x*1000, 1)).
ojson to_mm(const ojson& serial);

}  // namespace kvc
