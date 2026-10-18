#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace kvc {

class InferError : public std::runtime_error {
public:
    explicit InferError(const std::string& what) : std::runtime_error(what) {}
};

// TensorRT SSDlite engine.  Implemented only when KVC_WITH_TRT=1; the host
// build provides a stub that throws a clear error.
class Engine {
public:
    virtual ~Engine() = default;

    // input: 3*320*320 NCHW float32, already (x/255-0.5)/0.5.
    // cls_logits: 3234*5, bbox_reg: 3234*4 (row-major, binding order intact).
    virtual void infer(const std::vector<float>& input,
                       std::vector<float>* cls_logits,
                       std::vector<float>* bbox_reg) = 0;
};

// Throws InferError on missing file / deserialisation / binding mismatch.
std::unique_ptr<Engine> make_engine(const std::string& path);

}  // namespace kvc
