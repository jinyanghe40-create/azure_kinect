#include "kvc/ssd_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace kvc {
namespace {

constexpr int kFeatureSizes[6][2] = {{20, 20}, {10, 10}, {5, 5}, {3, 3}, {2, 2}, {1, 1}};
// Python computes these in double and only then casts the table to float32.
constexpr double kScales[7] = {0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0};
constexpr int kAnchorsPerLoc = 6;

float clipf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

float iou(const Detection& a, const Detection& b) {
    const float iw = std::max(0.0f, static_cast<float>(std::min(a.x1, b.x1) - std::max(a.x0, b.x0)));
    const float ih = std::max(0.0f, static_cast<float>(std::min(a.y1, b.y1) - std::max(a.y0, b.y0)));
    const float inter = iw * ih;
    const float aa = std::max(0.0f, static_cast<float>(a.x1 - a.x0)) *
                     std::max(0.0f, static_cast<float>(a.y1 - a.y0));
    const float ab = std::max(0.0f, static_cast<float>(b.x1 - b.x0)) *
                     std::max(0.0f, static_cast<float>(b.y1 - b.y0));
    return inter / std::max(aa + ab - inter, 1e-9f);
}

}  // namespace

std::vector<float> generate_anchors(int img_size) {
    const int n_levels = 6;
    size_t total = 0;
    for (const auto& fs : kFeatureSizes) total += static_cast<size_t>(fs[0]) * fs[1] * kAnchorsPerLoc;

    std::vector<float> anchors(total * 4);
    size_t k = 0;
    for (int lv = 0; lv < n_levels; ++lv) {
        const int fh = kFeatureSizes[lv][0];
        const int fw = kFeatureSizes[lv][1];
        const double s = kScales[lv];
        const double s2 = std::sqrt(s * kScales[lv + 1]);
        const double sqrt2 = std::sqrt(2.0);
        const double sqrt3 = std::sqrt(3.0);
        const double wh[6][2] = {
            {s, s},
            {s2, s2},
            {s * sqrt2, s / sqrt2},
            {s / sqrt2, s * sqrt2},
            {s * sqrt3, s / sqrt3},
            {s / sqrt3, s * sqrt3},
        };
        for (int i = 0; i < fh; ++i) {
            for (int j = 0; j < fw; ++j) {
                for (int a = 0; a < kAnchorsPerLoc; ++a) {
                    // Python clips the (cx, cy, w, h) representation to [0,1],
                    // NOT the corners.  So anchors near the border keep their
                    // negative x0/y0, and widths above 1 are clamped.
                    const float cx = clipf(
                        static_cast<float>((j + 0.5) / static_cast<double>(fw)),
                        0.0f, 1.0f);
                    const float cy = clipf(
                        static_cast<float>((i + 0.5) / static_cast<double>(fh)),
                        0.0f, 1.0f);
                    const float w = clipf(static_cast<float>(wh[a][0]), 0.0f, 1.0f);
                    const float h = clipf(static_cast<float>(wh[a][1]), 0.0f, 1.0f);
                    const float x0 = (cx - 0.5f * w) * static_cast<float>(img_size);
                    const float y0 = (cy - 0.5f * h) * static_cast<float>(img_size);
                    const float x1 = (cx + 0.5f * w) * static_cast<float>(img_size);
                    const float y1 = (cy + 0.5f * h) * static_cast<float>(img_size);
                    anchors[k * 4 + 0] = x0;
                    anchors[k * 4 + 1] = y0;
                    anchors[k * 4 + 2] = x1;
                    anchors[k * 4 + 3] = y1;
                    ++k;
                }
            }
        }
    }
    return anchors;
}

std::vector<Detection> postprocess(const std::vector<float>& cls_logits,
                                  const std::vector<float>& bbox_reg,
                                  const std::vector<float>& anchors,
                                  const SsdConfig& cfg) {
    const size_t n = anchors.size() / 4;
    if (n == 0 || cls_logits.size() != n * 5 || bbox_reg.size() != n * 4) {
        throw std::runtime_error("postprocess: shape mismatch");
    }

    const float clip_limit = std::log(1000.0f / 16.0f);
    constexpr float w[4] = {10.0f, 10.0f, 5.0f, 5.0f};

    std::vector<Detection> cand;
    cand.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const float* a = &anchors[i * 4];
        const float aw = a[2] - a[0];
        const float ah = a[3] - a[1];
        const float acx = a[0] + 0.5f * aw;
        const float acy = a[1] + 0.5f * ah;

        const float* r = &bbox_reg[i * 4];
        const float dx = r[0] / w[0];
        const float dy = r[1] / w[1];
        const float dw = std::min(r[2] / w[2], clip_limit);
        const float dh = std::min(r[3] / w[3], clip_limit);
        const float pcx = dx * aw + acx;
        const float pcy = dy * ah + acy;
        const float pw = std::exp(dw) * aw;
        const float ph = std::exp(dh) * ah;

        // softmax over the 5 logits (background + 4 foreground)
        const float* lg = &cls_logits[i * 5];
        float mx = lg[0];
        for (int c = 1; c < 5; ++c) mx = std::max(mx, lg[c]);
        float sum = 0.0f;
        float p[5];
        for (int c = 0; c < 5; ++c) {
            p[c] = std::exp(lg[c] - mx);
            sum += p[c];
        }
        int best = 0;
        float best_score = -1.0f;
        for (int c = 1; c < 5; ++c) {
            const float sc = p[c] / sum;
            if (sc > best_score) {  // first max wins
                best_score = sc;
                best = c;
            }
        }
        if (best_score >= cfg.score_thresh) {
            Detection d;
            d.x0 = pcx - 0.5f * pw;
            d.y0 = pcy - 0.5f * ph;
            d.x1 = pcx + 0.5f * pw;
            d.y1 = pcy + 0.5f * ph;
            d.cls_id = best - 1;
            d.score = best_score;
            cand.push_back(d);
        }
    }

    // Per-class NMS, classes processed in ascending order.
    std::vector<Detection> kept_by_class;
    for (int cls = 0; cls < 4; ++cls) {
        std::vector<Detection> own;
        for (const auto& d : cand) {
            if (d.cls_id == cls) own.push_back(d);
        }
        std::stable_sort(own.begin(), own.end(),
                         [](const Detection& a, const Detection& b) {
                             return a.score > b.score;
                         });
        std::vector<char> dead(own.size(), 0);
        for (size_t i = 0; i < own.size(); ++i) {
            if (dead[i]) continue;
            kept_by_class.push_back(own[i]);
            for (size_t j = i + 1; j < own.size(); ++j) {
                if (dead[j]) continue;
                if (iou(own[i], own[j]) >= cfg.nms_iou) dead[j] = 1;
            }
        }
    }

    std::stable_sort(kept_by_class.begin(), kept_by_class.end(),
                     [](const Detection& a, const Detection& b) {
                         return a.score > b.score;
                     });
    if (static_cast<int>(kept_by_class.size()) > cfg.max_det) {
        kept_by_class.resize(cfg.max_det);
    }
    return kept_by_class;
}

std::vector<Detection> postprocess(const std::vector<float>& cls_logits,
                                  const std::vector<float>& bbox_reg,
                                  const SsdConfig& cfg) {
    return postprocess(cls_logits, bbox_reg, generate_anchors(cfg.img_size), cfg);
}

}  // namespace kvc
