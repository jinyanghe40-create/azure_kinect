#pragma once

#include "kvc/types.hpp"

#include <functional>
#include <optional>
#include <vector>

namespace kvc {

// Projection used by the 2D-box cluster split.  Returns image pixel (u, v) in
// the rotated color frame, or nullopt when the point is behind/invalid.
using UvFn = std::function<std::optional<Eigen::Vector2d>(const Vec3&)>;

struct PipelineArgs {
    double edge = 0.25;
    bool edge_auto = true;
    double min_dist = 0.3;
    double max_dist = 1.5;
    double plane_thresh = 0.003;
    double table_thresh = 0.010;
    double voxel = 0.02;
    int min_cluster_points = 0;
    int max_faces = 3;
    double depth_noise_coef = 0.008;
    FaceSelect face_select = FaceSelect::MostFacing;

    std::optional<Vec3> up_axis;
    std::optional<Vec3> cam_axis;
    Vec3 cam_origin = Vec3::Zero();

    bool split_enabled = true;
    std::vector<Detection> split_boxes;
    UvFn project;  // required for splitting
};

struct PipelineResult {
    std::vector<GeoObject> objects;
    double timing_ms = 0.0;
};

// Port of common/pc_pipeline.run_pipeline.  Input points are already in the
// output (body/camera) frame and already validity-masked by the caller.
PipelineResult run_pipeline(const std::vector<Vec3>& pts, const PipelineArgs& args,
                            bool verbose = false);

}  // namespace kvc
