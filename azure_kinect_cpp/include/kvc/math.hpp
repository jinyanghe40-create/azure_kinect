#pragma once

#include "kvc/types.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>

#include <tuple>
#include <vector>

namespace kvc {

// ---------------------------------------------------------------------------
// Rotations (all angles in degrees, matching body_calib.py)
// ---------------------------------------------------------------------------

// Rotation about the optical axis (image roll).  Mirrors body_calib.Rz_optical
// and azure_capture._Rz: for rotate=180 this produces an exact-ish [[-1,0,0],
// [0,-1,0],[0,0,1]] (cos/sin of pi are not snapped to integers).
Mat3 Rz_optical(double rotate_deg);

Mat3 Ry(double deg);       // body pitch helper (standard Ry)
Mat3 Rz_body(double deg);  // yaw about body +Z

// Raw SDK depth optical frame (X right / Y down / Z forward) -> body frame
// (X forward / Y left / Z up), with roll baked in.
Mat3 build_R(double yaw_deg = 0.0, double pitch_deg = 0.0, double roll_deg = 180.0);

// Inverse of build_R.  Note the non-commuting order Ry(-pitch) @ Rz(-yaw).
std::tuple<double, double, double> decompose_R(const Mat3& R);

// (R @ Rz(rotate)^T)[2,1]; must be < 0 for a valid camera->body matrix.
double upright_residual(const Mat3& R, double rotate_deg);

// Minimal rotation taking direction a to direction b (Rodrigues).  a and b
// need not be normalized; the degenerate cases mirror azure_capture.
Mat3 rotation_between(const Vec3& a, const Vec3& b);

// The "upright, forward, no roll" base mapping.
extern const Mat3 C0_UPRIGHT;

// ---------------------------------------------------------------------------
// Python-compatible numeric helpers
// ---------------------------------------------------------------------------

// Python round(x, ndigits) uses round-half-to-even.  nearbyint under the
// default FE_TONEAREST mode gives the same tie behaviour after scaling.
// This is used ONLY when building output records, never inside the pipeline.
double python_round(double x, int ndigits);

std::vector<double> round_vec(const Vec3& v, int ndigits);

double clamp(double x, double lo, double hi);

}  // namespace kvc
