#include "kvc/math.hpp"

#include <cmath>

namespace kvc {

const Mat3 C0_UPRIGHT = [] {
    Mat3 m;
    m << 0.0, 0.0, 1.0,
        -1.0, 0.0, 0.0,
         0.0, -1.0, 0.0;
    return m;
}();

Mat3 Rz_optical(double rotate_deg) {
    // Python: k = int(rotate_deg) // 90 (floor division), then radians(k*90).
    const int k = static_cast<int>(std::floor(rotate_deg / 90.0));
    if (k == 0) return Mat3::Identity();
    const double th = M_PI * 90.0 * static_cast<double>(k) / 180.0;
    const double c = std::cos(th);
    const double s = std::sin(th);
    Mat3 m;
    m << c, -s, 0.0,
         s,  c, 0.0,
         0.0, 0.0, 1.0;
    return m;
}

Mat3 Ry(double deg) {
    const double t = deg * M_PI / 180.0;
    const double c = std::cos(t);
    const double s = std::sin(t);
    Mat3 m;
    m << c, 0.0, s,
         0.0, 1.0, 0.0,
        -s, 0.0, c;
    return m;
}

Mat3 Rz_body(double deg) {
    const double t = deg * M_PI / 180.0;
    const double c = std::cos(t);
    const double s = std::sin(t);
    Mat3 m;
    m << c, -s, 0.0,
         s,  c, 0.0,
         0.0, 0.0, 1.0;
    return m;
}

Mat3 build_R(double yaw_deg, double pitch_deg, double roll_deg) {
    return Rz_body(yaw_deg) * Ry(pitch_deg) * C0_UPRIGHT * Rz_optical(roll_deg);
}

std::tuple<double, double, double> decompose_R(const Mat3& R) {
    const Vec3 a = R.col(2);
    const double pitch = std::asin(clamp(-a[2], -1.0, 1.0)) * 180.0 / M_PI;
    const double yaw = std::atan2(a[1], a[0]) * 180.0 / M_PI;
    const Vec3 d = R.col(1);
    const Vec3 u = Ry(-pitch) * Rz_body(-yaw) * d;
    const double roll = std::atan2(u[1], -u[2]) * 180.0 / M_PI;
    return {yaw, pitch, roll};
}

double upright_residual(const Mat3& R, double rotate_deg) {
    const Mat3 m = R * Rz_optical(rotate_deg).transpose();
    return m(2, 1);
}

Mat3 rotation_between(const Vec3& a_in, const Vec3& b_in) {
    Vec3 a = a_in;
    Vec3 b = b_in;
    const double na = a.norm();
    const double nb = b.norm();
    if (na < 1e-12 || nb < 1e-12) return Mat3::Identity();
    a /= na;
    b /= nb;

    const Vec3 v = a.cross(b);
    const double c = a.dot(b);
    const double s = v.norm();

    if (s < 1e-9) {
        if (c > 0.0) return Mat3::Identity();
        // 180 degrees: rotate about any axis perpendicular to a.
        Vec3 tmp(1.0, 0.0, 0.0);
        if (std::fabs(a[0]) > 0.9) tmp = Vec3(0.0, 1.0, 0.0);
        Vec3 axis = a.cross(tmp);
        const double an = axis.norm();
        if (an < 1e-12) return Mat3::Identity();
        axis /= an;
        return 2.0 * axis * axis.transpose() - Mat3::Identity();
    }

    Eigen::Matrix3d vx;
    vx <<     0.0, -v[2],  v[1],
             v[2],   0.0, -v[0],
            -v[1],  v[0],   0.0;
    const Mat3 vx2 = vx * vx;
    return Mat3::Identity() + vx + vx2 * ((1.0 - c) / (s * s));
}

double python_round(double x, int ndigits) {
    if (!std::isfinite(x)) return x;
    double p = 1.0;
    for (int i = 0; i < ndigits; ++i) p *= 10.0;
    // Scale, round half-to-even, unscale.  Multiplication/division by powers of
    // ten is exact until 1e22 for doubles, so this matches Python's decimal
    // round for all values this pipeline emits (meters/millimeters/degrees).
    return std::nearbyint(x * p) / p;
}

std::vector<double> round_vec(const Vec3& v, int ndigits) {
    return {python_round(v[0], ndigits), python_round(v[1], ndigits),
            python_round(v[2], ndigits)};
}

double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

}  // namespace kvc
