#include "kvc/body_calib.hpp"
#include "test_common.hpp"

#include <cstdio>
#include <fstream>
#include <functional>

using namespace kvc;

namespace {

ojson make_doc(const Mat3& R, const Vec3& t, int rotate = 180) {
    ojson d;
    d["schema"] = "k4a_body_calib";
    d["version"] = 1;
    d["id"] = "unit-test";
    d["image_rotate_deg"] = rotate;
    ojson Rj = ojson::array();
    for (int i = 0; i < 3; ++i) {
        ojson row = ojson::array();
        for (int j = 0; j < 3; ++j) row.push_back(R(i, j));
        Rj.push_back(row);
    }
    d["R_cam_to_body"] = Rj;
    d["t_cam_to_body_m"] = ojson::array({t[0], t[1], t[2]});
    return d;
}

bool throws_body_error(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const BodyCalibError&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void test_accepts_real_numbers() {
    const Mat3 R = build_R(0.0, 9.7, 180.0);
    const ojson d = make_doc(R, Vec3(0.25, 0.0, 0.60), 180);
    const CheckResult c = check_calib(d, 180);
    CHECK(c.image_rotate_deg == 180);
    CHECK_NEAR(c.R.determinant(), 1.0, 1e-12);
    CHECK(upright_residual(c.R, 180) < 0.0);
    CHECK_NEAR((c.t - Vec3(0.25, 0.0, 0.60)).norm(), 0.0, 1e-15);
}

void test_rejections() {
    const Mat3 R = build_R(0.0, 9.7, 180.0);
    const Vec3 t(0.25, 0.0, 0.60);

    // 1. post-Rz convention (180 counted twice)
    CHECK(throws_body_error([&] {
        check_calib(make_doc(R * Rz_optical(180.0), t, 180), 180);
    }));
    // 2. transpose
    CHECK(throws_body_error([&] { check_calib(make_doc(R.transpose(), t, 180), 180); }));
    // 3. reflection
    Mat3 refl = Mat3::Identity();
    refl(2, 2) = -1.0;
    CHECK(throws_body_error([&] { check_calib(make_doc(refl * R, t, 180), 180); }));
    // 4. --rotate mismatch
    CHECK(throws_body_error([&] { check_calib(make_doc(R, t, 180), 0); }));
    // 5. mm written as m
    CHECK(throws_body_error([&] {
        check_calib(make_doc(R, Vec3(250.0, 0.0, 600.0), 180), 180);
    }));
    // 6. wrong schema
    CHECK(throws_body_error([&] {
        ojson d = make_doc(R, t, 180);
        d["schema"] = "k4a_calibration";
        check_calib(d, 180);
    }));
    // 7. flat matrix
    CHECK(throws_body_error([&] {
        ojson d = make_doc(R, t, 180);
        d["R_cam_to_body"] = ojson::array({0, 0, 1, -1, 0, 0, 0, -1, 0});
        check_calib(d, 180);
    }));
}

void test_write_and_load() {
    const std::string path = "/tmp/kvc_test_body_calib.json";
    const ojson d = write_body_calib(path, 0.0, 9.7, 180.0, 250.0, 0.0, 600.0,
                                     "unit-test");
    CHECK(d["schema"] == "k4a_body_calib");
    CHECK(d["image_rotate_deg"] == 180);

    std::ifstream f(path);
    CHECK(static_cast<bool>(f));
    const auto calib = load_body_calib(path, 180, false);
    CHECK(calib.has_value());
    if (calib) {
        CHECK(calib->id == "unit-test");
        CHECK(calib->image_rotate_deg == 180);
        CHECK_NEAR(calib->t[0], 0.25, 1e-6);
        CHECK_NEAR(calib->t[2], 0.60, 1e-6);
    }
    std::remove(path.c_str());

    // allow_missing
    CHECK(!load_body_calib("/tmp/kvc_definitely_missing.json", std::nullopt, true)
               .has_value());
    CHECK(throws_body_error([] {
        load_body_calib("/tmp/kvc_definitely_missing.json", std::nullopt, false);
    }));
}

}  // namespace

int main() {
    test_accepts_real_numbers();
    test_rejections();
    test_write_and_load();
    return kvc_test::finish("test_body_calib");
}
