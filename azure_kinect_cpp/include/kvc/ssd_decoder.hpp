#pragma once

#include "kvc/types.hpp"

#include <vector>

namespace kvc {

struct SsdConfig {
    int img_size = 320;
    float score_thresh = 0.4f;
    float nms_iou = 0.55f;
    int max_det = 20;
};

// 3234 anchored boxes in 320x320 pixel coordinates, row-major (y, x), the six
// aspect/size ratios in the exact order used by ssd_decoder.py.
std::vector<float> generate_anchors(int img_size = 320);

// cls_logits: (N, 5) row-major; bbox_reg: (N, 4); anchors: (N, 4).
// Returns detections in 320x320 coordinates with 0-based foreground class ids.
std::vector<Detection> postprocess(const std::vector<float>& cls_logits,
                                  const std::vector<float>& bbox_reg,
                                  const std::vector<float>& anchors,
                                  const SsdConfig& cfg);

// Convenience: anchors generated with cfg.img_size.
std::vector<Detection> postprocess(const std::vector<float>& cls_logits,
                                  const std::vector<float>& bbox_reg,
                                  const SsdConfig& cfg);

}  // namespace kvc
