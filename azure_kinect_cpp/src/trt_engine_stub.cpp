// Built when KVC_WITH_TRT=OFF (host).  Keeps app_run linkable; the error is
// explicit instead of a link failure.

#include "kvc/trt_engine.hpp"

namespace kvc {

std::unique_ptr<Engine> make_engine(const std::string&) {
    throw InferError(
        "这个二进制没有编入 TensorRT（KVC_WITH_TRT=0）。\n"
        "  检测链路只能在装了 TensorRT 的 Jetson 上跑：\n"
        "    bash ~/work/azure_kinect_cpp/scripts/build_board.sh");
}

}  // namespace kvc
