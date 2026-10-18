#pragma once

// Internal building blocks of the point-cloud geometry pipeline.  They mirror
// common/pc_pipeline.py function-for-function; only pc_pipeline.cpp (run_pipeline)
// is part of the public surface.

#include "kvc/types.hpp"

#include <optional>
#include <string>
#include <vector>

namespace kvc {
namespace geom {

using Mask = std::vector<char>;

struct PlaneFit {
    Vec3 normal = Vec3::Zero();
    double d = 0.0;
    Mask mask;  // inliers over the input points, |p.n + d| < thresh
};

// Port of ransac_plane().  `normal_hint` is a soft constraint that is checked
// again after the PCA tightening pass (the Python code learned this the hard
// way: refinement can rotate the normal onto a real object face).
std::optional<PlaneFit> ransac_plane(const std::vector<Vec3>& pts, double thresh,
                                     int iters, int min_inliers,
                                     const std::optional<Vec3>& normal_hint,
                                     double cos_tol = 0.5,
                                     int max_fit_pts = 20000);

Vec3 orient_toward_camera(const Vec3& n, const std::vector<Vec3>& inl,
                          const Vec3& cam_origin);

// 26-neighbourhood connected components on a 0.02 m voxel grid.  Labels are
// 0..k-1 in the same order scipy's connected_components would produce; -1 means
// unassigned.  Returns an empty vector when the Python version would return
// np.array([]) (degenerate grid).
std::vector<int> voxel_cluster(const std::vector<Vec3>& pts, double voxel,
                               int min_points);

// Port of split_clusters_by_boxes().  `uv` and `uv_valid` are parallel to pts;
// invalid entries are the C++ equivalent of NaN (they can never fall inside a
// box and are dropped unless a box is close enough).
std::vector<int> split_clusters_by_boxes(
    const std::vector<Vec3>& pts, const std::vector<int>& labels,
    int n_clusters, const std::vector<Eigen::Vector2d>& uv,
    const std::vector<char>& uv_valid, const std::vector<Detection>& boxes,
    int min_pts, int* n_out, bool verbose);

struct ClusterFace {
    Vec3 normal = Vec3::Zero();
    Vec3 center = Vec3::Zero();
    int n_points = 0;
    double rms = 0.0;     // meters
    double extent = 0.0;  // meters
};

// Port of analyze_cluster(): up to max_faces mutually near-perpendicular
// planes, normals oriented toward cam_origin, sorted by n_points descending.
std::vector<ClusterFace> analyze_cluster(const std::vector<Vec3>& pts,
                                         double dist_thresh, int max_faces,
                                         const Vec3& cam_origin);

double face_edge_len(const std::vector<Vec3>& dp, const Vec3& e1,
                     const Vec3& e2);

// Port of measure_edge().  `edge` is nullopt when nothing could be measured.
void measure_edge(const std::vector<ClusterFace>& faces,
                  const std::vector<Vec3>& pts, double tol, int min_pts,
                  double spread_tol, std::optional<double>* edge,
                  std::vector<double>* vals, std::string* src);

std::optional<Vec3> cube_center_from_faces(const std::vector<ClusterFace>& faces,
                                           double edge_len,
                                           const std::vector<Vec3>& cluster_pts);

// np.median equivalent (even counts average the two middle values).
double median(std::vector<double> v);

}  // namespace geom
}  // namespace kvc
