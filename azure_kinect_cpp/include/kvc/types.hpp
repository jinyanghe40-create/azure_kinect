#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kvc {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

struct Intrinsics {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    int width = 0;
    int height = 0;
    std::string model;  // "brown_conrady" for depth; empty for color
};

struct Extrinsics {
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();
};

// Detector box in the (already rotated) color image, pixels.
struct Detection {
    double x0 = 0.0;
    double y0 = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
    int cls_id = 0;
    double score = 0.0;
};

enum class FaceSelect { MostFacing, Largest };

// One fitted plane of a cluster.  All lengths in meters, angles in degrees.
struct Face {
    Vec3 normal = Vec3::Zero();
    Vec3 center = Vec3::Zero();
    int n_points = 0;
    double rms_mm = 0.0;
    double angle_deg = 0.0;
    std::optional<double> lean_deg;
    double extent = 0.0;
};

// Geometry-pipeline output, before class matching / serial mapping.
struct GeoObject {
    int id = 0;
    Vec3 face_center = Vec3::Zero();
    Vec3 normal = Vec3::Zero();
    double angle_deg = 0.0;
    int n_points = 0;
    int n_faces = 0;
    double face_rms_mm = 0.0;
    std::optional<double> lean_deg;

    std::vector<Face> faces;
    std::optional<Vec3> cube_center;

    int n_face_pts = 0;
    double nearest_m = 0.0;
    Vec3 nearest_pt = Vec3::Zero();
    double plane_thresh_used_mm = 0.0;
    double sigma_z_mm = 0.0;
    double edge_used_m = 0.0;
    std::string edge_src;               // "auto" | "auto-max" | "cli"
    std::vector<double> edge_faces_m;

    // Not serialized: the cluster's own points (kept for diagnostics/ROS-less use).
    std::vector<Vec3> pts;
};

enum class Quality { GOOD, FAIR, COARSE };

inline const char* quality_string(Quality q) {
    switch (q) {
        case Quality::GOOD: return "GOOD";
        case Quality::FAIR: return "FAIR";
        default: return "COARSE";
    }
}

}  // namespace kvc
