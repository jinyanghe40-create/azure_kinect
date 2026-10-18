#include "kvc/math.hpp"
#include "test_common.hpp"

#include <Eigen/Geometry>
#include <random>

using namespace kvc;

namespace {

Mat3 random_rotation(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Eigen::Quaterniond q(u(rng), u(rng), u(rng), u(rng));
    q.normalize();
    return q.toRotationMatrix();
}

double wrap180(double a) {
    while (a > 180.0) a -= 360.0;
    while (a < -180.0) a += 360.0;
    return a;
}

void test_build_and_guard() {
    CHECK_MAT_NEAR(build_R(0, 0, 0), C0_UPRIGHT, 1e-15);

    const Mat3 R = build_R(0.0, 9.7, 180.0);
    CHECK(R(2, 1) > 0.0);
    CHECK(upright_residual(R, 180.0) < 0.0);
    // -cos(9.7°) ≈ -0.9857: "down" still dominates after undoing the roll.
    CHECK_NEAR(upright_residual(R, 180.0), -std::cos(9.7 * M_PI / 180.0), 1e-12);

    // Double-counting 180 degrees must be rejected by the guard formula.
    const Mat3 doubled = R * Rz_optical(180.0);
    CHECK(upright_residual(doubled, 180.0) >= 0.0);
}

void test_decompose_roundtrip() {
    for (double yaw = -40.0; yaw <= 40.0 + 1e-9; yaw += 10.0) {
        for (double pitch = -40.0; pitch <= 40.0 + 1e-9; pitch += 10.0) {
            for (double roll : {0.0, 90.0, 180.0, 270.0}) {
                const Mat3 R = build_R(yaw, pitch, roll);
                const auto [y2, p2, r2] = decompose_R(R);
                CHECK_NEAR(wrap180(y2 - yaw), 0.0, 1e-9);
                CHECK_NEAR(wrap180(p2 - pitch), 0.0, 1e-9);
                CHECK_NEAR(wrap180(r2 - roll), 0.0, 1e-9);
            }
        }
    }
}

void test_projection_consistency() {
    std::mt19937 rng(20261015u);
    std::normal_distribution<double> n(0.0, 0.8);
    std::normal_distribution<double> jitter(0.0, 0.05);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    for (double rotate : {0.0, 90.0, 180.0, 270.0}) {
        const Mat3 Rz = Rz_optical(rotate);
        for (int trial = 0; trial < 40; ++trial) {
            const Mat3 R_out = random_rotation(rng);
            const Vec3 t_out(0.25 + jitter(rng), jitter(rng), 0.60 + jitter(rng));
            const Mat3 R_dc = random_rotation(rng);
            const Vec3 t_dc(-0.032, 0.002, -0.001);
            const Vec3 p_sdk(n(rng), n(rng), n(rng));

            const Mat3 R_e = Rz * R_dc * R_out.transpose();
            const Vec3 t_e = Rz * (t_dc - R_dc * R_out.transpose() * t_out);

            const Vec3 truth = Rz * (R_dc * p_sdk + t_dc);
            const Vec3 p_body = R_out * p_sdk + t_out;
            const Vec3 got = R_e * p_body + t_e;
            CHECK_NEAR((got - truth).cwiseAbs().maxCoeff(), 0.0, 1e-9);

            // The transpose trap: using R_out instead of R_out^T must differ.
            const Mat3 bad_R = Rz * R_dc * R_out;
            const Vec3 bad = bad_R * p_body + t_e;
            CHECK((bad - truth).norm() > 1e-3);

            // Translation must reach the cloud and round-trip.
            const Vec3 back = R_out.transpose() * (p_body - t_out);
            CHECK_NEAR((back - p_sdk).cwiseAbs().maxCoeff(), 0.0, 1e-12);

            // Direction transforms must not translate.
            const Vec3 dir = Vec3(u(rng), u(rng), u(rng)).normalized();
            const Vec3 cam_dir = R_e * dir;
            CHECK_NEAR(cam_dir.norm(), 1.0, 1e-12);
        }
    }
}

void test_rotation_between() {
    const Vec3 a = Vec3(0.0, 1.0, 0.0);  // raw optical +Y is down
    const Vec3 b = Vec3(0.0, -1.0, 0.0);
    const Mat3 R = rotation_between(a, b);
    CHECK_NEAR((R * a - b).norm(), 0.0, 1e-12);
    CHECK_NEAR(R.determinant(), 1.0, 1e-9);

    const Mat3 I = rotation_between(Vec3(0, 1, 0), Vec3(0, 2, 0));
    CHECK_MAT_NEAR(I, Mat3::Identity(), 1e-12);

    std::mt19937 rng(7);
    for (int i = 0; i < 50; ++i) {
        const Vec3 x = Vec3(rng(), rng(), rng()).normalized();
        const Vec3 y = Vec3(rng(), rng(), rng()).normalized();
        const Mat3 Rx = rotation_between(x, y);
        CHECK_NEAR((Rx * x - y).norm(), 0.0, 1e-9);
        CHECK_NEAR(Rx.determinant(), 1.0, 1e-9);
    }
}

void test_rounding() {
    CHECK_NEAR(python_round(0.5, 0), 0.0, 0.0);
    CHECK_NEAR(python_round(1.5, 0), 2.0, 0.0);
    CHECK_NEAR(python_round(2.5, 0), 2.0, 0.0);
    CHECK_NEAR(python_round(-0.5, 0), 0.0, 0.0);
    CHECK_NEAR(python_round(-1.5, 0), -2.0, 0.0);
    CHECK_NEAR(python_round(1.23456, 4), 1.2346, 1e-12);
    CHECK_NEAR(python_round(1.0, 4), 1.0, 0.0);
}

}  // namespace

int main() {
    test_build_and_guard();
    test_decompose_roundtrip();
    test_projection_consistency();
    test_rotation_between();
    test_rounding();
    return kvc_test::finish("test_math");
}
