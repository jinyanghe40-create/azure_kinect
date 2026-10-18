#pragma once

#include <cstdint>
#include <vector>

namespace kvc {

// Detector preprocessing, byte-for-byte equivalent to trt_infer.preprocess:
//   BGR -> RGB, stretch-resize to size x size (INTER_LINEAR on uint8),
//   /255, (x-0.5)/0.5, HWC -> CHW, float32.
//
// Returns a contiguous NCHW blob of length 3*size*size.
std::vector<float> preprocess_bgr(const uint8_t* bgr, int width, int height,
                                  int row_stride_bytes, int size = 320);

}  // namespace kvc
