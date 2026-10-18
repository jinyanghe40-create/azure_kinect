#include "kvc/preprocess.hpp"

#include <stdexcept>

#if KVC_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace kvc {

std::vector<float> preprocess_bgr(const uint8_t* bgr, int width, int height,
                                  int row_stride_bytes, int size) {
#if KVC_HAVE_OPENCV
    if (!bgr || width <= 0 || height <= 0 || size <= 0) {
        throw std::runtime_error("preprocess_bgr: invalid image");
    }
    cv::Mat src(height, width, CV_8UC3, const_cast<uint8_t*>(bgr),
                static_cast<size_t>(row_stride_bytes));
    cv::Mat rgb;
    cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);

    cv::Mat small;
    // cv2.resize default interpolation on uint8 is INTER_LINEAR; no letterbox.
    cv::resize(rgb, small, cv::Size(size, size), 0.0, 0.0, cv::INTER_LINEAR);

    std::vector<float> out(static_cast<size_t>(3) * size * size);
    const int plane = size * size;
    for (int y = 0; y < size; ++y) {
        const uint8_t* row = small.ptr<uint8_t>(y);
        for (int x = 0; x < size; ++x) {
            const uint8_t* px = row + x * 3;
            for (int c = 0; c < 3; ++c) {
                const float v = static_cast<float>(px[c]) / 255.0f;
                out[static_cast<size_t>(c) * plane + y * size + x] = (v - 0.5f) / 0.5f;
            }
        }
    }
    return out;
#else
    (void)bgr;
    (void)width;
    (void)height;
    (void)row_stride_bytes;
    (void)size;
    throw std::runtime_error("preprocess_bgr: built without OpenCV");
#endif
}

}  // namespace kvc
