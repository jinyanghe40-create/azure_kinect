#pragma once

// Shared setup for the acceptance tools.  They build the same geometry
// arguments from the live source state (never guessed).

#include "kvc/azure_source.hpp"
#include "kvc/body_calib.hpp"
#include "kvc/pc_pipeline.hpp"

#include <optional>
#include <string>
#include <vector>

namespace kvc {

// Mirrors check_body_frame._gargs / run_azure's geometry_args block:
//   up_axis    calibration -> body +Z; gravity aligned -> camera +Y; else none
//   cam_axis   optical axis in the output frame = R_out[:, 2]
//   cam_origin camera centre in the output frame = t_out
inline PipelineArgs diag_pipeline_args(const AzureSource& src,
                                       const std::optional<BodyCalib>& calib,
                                       double min_dist = 0.3,
                                       double max_dist = 1.5,
                                       double voxel = 0.02,
                                       double edge = 0.25,
                                       double plane_thresh = 0.003,
                                       double table_thresh = 0.010,
                                       int max_faces = 3,
                                       int min_cluster_points = 200,
                                       FaceSelect face_select =
                                           FaceSelect::MostFacing) {
    PipelineArgs g;
    g.min_dist = min_dist;
    g.max_dist = max_dist;
    g.voxel = voxel;
    g.edge = edge;
    g.plane_thresh = plane_thresh;
    g.table_thresh = table_thresh;
    g.max_faces = max_faces;
    g.min_cluster_points = min_cluster_points;
    g.face_select = face_select;
    g.edge_auto = true;
    g.depth_noise_coef = 0.008;
    // The diagnostic tools do not split clusters by 2D boxes.
    g.split_enabled = false;
    if (calib) {
        g.up_axis = Vec3(0.0, 0.0, 1.0);
    } else if (src.gravity_ok()) {
        g.up_axis = Vec3(0.0, 1.0, 0.0);
    }
    g.cam_axis = src.R_out().col(2);
    g.cam_origin = src.t_out();
    return g;
}

// FrameData::points is stride-sampled over the raw grid; pts[valid] in Python.
inline std::vector<Vec3> valid_points(const FrameData& f) {
    std::vector<Vec3> out;
    out.reserve(f.points.size());
    for (size_t i = 0; i < f.points.size(); ++i) {
        if (i < f.valid.size() && f.valid[i]) out.push_back(f.points[i]);
    }
    return out;
}

}  // namespace kvc
