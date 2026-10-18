// Built when KVC_WITH_K4A=OFF.  Capture-facing executables refuse to run and
// point at a board build; core libraries and unit tests are unaffected.

#ifndef KVC_HAVE_OPENCV
#define KVC_HAVE_OPENCV 0
#endif

#if KVC_HAVE_OPENCV

#include "kvc/azure_source.hpp"

#include <optional>

namespace kvc {

struct AzureSource::Impl {};

const char* k4a_unavailable_reason() {
    return "this binary was built without the Azure Kinect SDK "
           "(KVC_WITH_K4A=0); rebuild on the Jetson";
}

AzureSource::AzureSource(const SourceConfig&, std::optional<BodyCalib>) {
    throw CaptureError(k4a_unavailable_reason());
}

AzureSource::~AzureSource() = default;

std::optional<FrameData> AzureSource::get(int, int) { return std::nullopt; }

std::optional<Vec3> AzureSource::read_up_depth(std::string* err) {
    if (err) *err = k4a_unavailable_reason();
    return std::nullopt;
}

}  // namespace kvc

#else

// Without OpenCV even the frame container does not exist, so nothing can
// reference AzureSource (run_azure_cpp is skipped in that configuration).
namespace kvc {
const char* k4a_unavailable_reason() {
    return "this binary was built without the Azure Kinect SDK "
           "(KVC_WITH_K4A=0) or OpenCV";
}
}  // namespace kvc

#endif
