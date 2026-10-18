#include "kvc/ssd_decoder.hpp"
#include "test_common.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace kvc;

#ifndef KVC_FIXTURE_DIR
#define KVC_FIXTURE_DIR "tests/fixtures"
#endif

namespace {

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return static_cast<bool>(f);
}

std::vector<float> read_f32(const std::string& path, size_t count) {
    std::vector<float> out(count);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    return out;
}

void test_python_fixtures() {
    const std::string dir = KVC_FIXTURE_DIR;
    const std::string anchors_path = dir + "/anchors.f32";
    if (!file_exists(anchors_path)) {
        std::cout << "  (fixtures absent; run tests/gen_fixtures.py)" << std::endl;
        return;
    }

    const std::vector<float> py_anchors = read_f32(anchors_path, 3234u * 4u);
    const std::vector<float> anchors = generate_anchors(320);
    CHECK(anchors.size() == py_anchors.size());
    double max_anchor_err = 0.0;
    for (size_t i = 0; i < anchors.size() && i < py_anchors.size(); ++i) {
        max_anchor_err = std::max(max_anchor_err,
                                  std::fabs(static_cast<double>(anchors[i]) -
                                            py_anchors[i]));
    }
    // 320-pixel coordinates; sub-1e-4 differences are pure float ordering.
    CHECK_MSG(max_anchor_err <= 1e-5,
              "max anchor diff " + std::to_string(max_anchor_err));

    for (int seed : {101, 202, 303}) {
        const std::string cls_path = dir + "/cls_" + std::to_string(seed) + ".f32";
        const std::string reg_path = dir + "/reg_" + std::to_string(seed) + ".f32";
        const std::string det_path = dir + "/dets_" + std::to_string(seed) + ".json";
        if (!file_exists(cls_path) || !file_exists(reg_path) ||
            !file_exists(det_path)) {
            continue;
        }
        const auto cls = read_f32(cls_path, 3234u * 5u);
        const auto reg = read_f32(reg_path, 3234u * 4u);
        const auto got = postprocess(cls, reg, anchors, SsdConfig{});

        std::ifstream jf(det_path);
        nlohmann::json want = nlohmann::json::parse(jf);
        CHECK(got.size() == want.size());
        const size_t n = std::min(got.size(), want.size());
        for (size_t i = 0; i < n; ++i) {
            CHECK(got[i].cls_id == want[i]["cls"].get<int>());
            CHECK_NEAR(got[i].score, want[i]["score"].get<double>(), 1e-5);
            CHECK_NEAR(got[i].x0, want[i]["x0"].get<double>(), 1e-2);
            CHECK_NEAR(got[i].y0, want[i]["y0"].get<double>(), 1e-2);
            CHECK_NEAR(got[i].x1, want[i]["x1"].get<double>(), 1e-2);
            CHECK_NEAR(got[i].y1, want[i]["y1"].get<double>(), 1e-2);
        }
    }
}

void test_anchor_count_and_values() {
    const std::vector<float> a = generate_anchors(320);
    CHECK(a.size() == 3234u * 4u);

    // Level 0 (20x20), location (0,0), anchor 0: scale 0.2 at (0.025, 0.025).
    CHECK_NEAR(a[0], -24.0, 1e-5);
    CHECK_NEAR(a[1], -24.0, 1e-5);
    CHECK_NEAR(a[2], 40.0, 1e-5);
    CHECK_NEAR(a[3], 40.0, 1e-5);

    // Anchor 1 is scale sqrt(0.2*0.35).  Python clips (cx,cy,w,h), so the
    // corners may still be negative.
    const float s2 = std::sqrt(0.2f * 0.35f);
    CHECK_NEAR(a[4], (0.025 - 0.5 * s2) * 320.0, 1e-4);
    CHECK_NEAR(a[5], (0.025 - 0.5 * s2) * 320.0, 1e-4);
    CHECK_NEAR(a[6], (0.025 + 0.5 * s2) * 320.0, 1e-4);
    CHECK_NEAR(a[7], (0.025 + 0.5 * s2) * 320.0, 1e-4);

    // Second location (j=1) is x-shifted by 1/20 of the image.
    CHECK_NEAR(a[6 * 4 + 0], -8.0, 1e-4);              // (0.075-0.1)*320
    CHECK_NEAR(a[6 * 4 + 2], 56.0, 1e-4);              // (0.075+0.1)*320

    // Last level (1x1) starts at index 539*6-6 = 3228.
    const size_t last = 3228 * 4;
    CHECK_NEAR(a[last + 0], 8.0, 1e-5);
    CHECK_NEAR(a[last + 1], 8.0, 1e-5);
    CHECK_NEAR(a[last + 2], 312.0, 1e-5);
    CHECK_NEAR(a[last + 3], 312.0, 1e-5);

    // Anchors are non-decreasing in (row, col) within a level: index 6*20
    // is the first anchor of row 1 and must be y-shifted.
    const size_t row1 = 6 * 20 * 4;
    CHECK_NEAR(a[row1 + 0], -24.0, 1e-5);
    CHECK_NEAR(a[row1 + 1], -8.0, 1e-5);   // (0.075-0.1)*320
    CHECK_NEAR(a[row1 + 3], 56.0, 1e-5);
}

void test_postprocess_single() {
    const std::vector<float> anchors = {100.0f, 100.0f, 140.0f, 120.0f};
    const std::vector<float> logits = {0.0f, 10.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> reg = {0.0f, 0.0f, 0.0f, 0.0f};
    const auto dets = postprocess(logits, reg, anchors, SsdConfig{});
    CHECK(dets.size() == 1u);
    if (!dets.empty()) {
        CHECK(dets[0].cls_id == 0);
        CHECK_NEAR(dets[0].score, 0.999818, 1e-6);
        CHECK_NEAR(dets[0].x0, 100.0, 1e-5);
        CHECK_NEAR(dets[0].y0, 100.0, 1e-5);
        CHECK_NEAR(dets[0].x1, 140.0, 1e-5);
        CHECK_NEAR(dets[0].y1, 120.0, 1e-5);
    }
}

void test_postprocess_box_decode() {
    const std::vector<float> anchors = {100.0f, 100.0f, 140.0f, 120.0f};
    // dx=dy=0; dw=5*ln 3 -> width x3; dh=0.
    const float ln3 = std::log(3.0f);
    const std::vector<float> logits = {0.0f, 3.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> reg = {0.0f, 0.0f, 5.0f * ln3, 0.0f};
    const auto dets = postprocess(logits, reg, anchors, SsdConfig{});
    CHECK(dets.size() == 1u);
    if (!dets.empty()) {
        CHECK_NEAR(dets[0].x0, 60.0, 1e-3);
        CHECK_NEAR(dets[0].x1, 180.0, 1e-3);
        CHECK_NEAR(dets[0].y0, 100.0, 1e-4);
        CHECK_NEAR(dets[0].y1, 120.0, 1e-4);
    }
}

void test_threshold_and_nms() {
    // Two overlapping anchors, same class: lower score is suppressed.
    {
        const std::vector<float> anchors = {
            0.0f, 0.0f, 100.0f, 100.0f,
            5.0f, 5.0f, 105.0f, 105.0f,
        };
        const std::vector<float> logits = {
            0.0f, 10.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 3.0f, 0.0f, 0.0f, 0.0f,
        };
        const std::vector<float> reg(8, 0.0f);
        const auto dets = postprocess(logits, reg, anchors, SsdConfig{});
        CHECK(dets.size() == 1u);
        if (!dets.empty()) CHECK_NEAR(dets[0].score, 0.999818, 1e-6);
    }
    // Same boxes, different classes: both survive NMS.
    {
        const std::vector<float> anchors = {
            0.0f, 0.0f, 100.0f, 100.0f,
            5.0f, 5.0f, 105.0f, 105.0f,
        };
        const std::vector<float> logits = {
            0.0f, 10.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 10.0f, 0.0f, 0.0f,
        };
        const std::vector<float> reg(8, 0.0f);
        const auto dets = postprocess(logits, reg, anchors, SsdConfig{});
        CHECK(dets.size() == 2u);
    }
    // Uniform logits -> 0.2 < 0.4 threshold.
    {
        const std::vector<float> anchors = {0.0f, 0.0f, 100.0f, 100.0f};
        const std::vector<float> logits(5, 0.0f);
        const std::vector<float> reg(4, 0.0f);
        const auto dets = postprocess(logits, reg, anchors, SsdConfig{});
        CHECK(dets.empty());
    }
}

void test_shape_mismatch_throws() {
    bool threw = false;
    try {
        postprocess(std::vector<float>(5, 0.0f), std::vector<float>(4, 0.0f),
                    std::vector<float>(4, 0.0f), SsdConfig{});
        // one anchor, but 5 logits for... one anchor is fine; use 2 anchors.
        postprocess(std::vector<float>(5, 0.0f), std::vector<float>(4, 0.0f),
                    std::vector<float>(8, 0.0f), SsdConfig{});
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    test_anchor_count_and_values();
    test_postprocess_single();
    test_postprocess_box_decode();
    test_threshold_and_nms();
    test_shape_mismatch_throws();
    test_python_fixtures();
    return kvc_test::finish("test_ssd_decoder");
}
