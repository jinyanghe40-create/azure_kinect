// TensorRT backend (Jetson).  Direct counterpart of common/trt_infer.py Engine:
// bindings are addressed by NAME, never by index, and the engine's own I/O
// dtype is reported rather than assumed.

#include "kvc/trt_engine.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

namespace kvc {
namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

int64_t volume(const nvinfer1::Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) throw InferError("engine 是动态 shape，本程序只支持静态引擎");
        v *= d.d[i];
    }
    return v;
}

std::string dims_string(const nvinfer1::Dims& d) {
    std::string s;
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) s += "x";
        s += std::to_string(d.d[i]);
    }
    return s;
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw InferError("找不到 engine: " + path);
    f.seekg(0, std::ios::end);
    const std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n <= 0) throw InferError("engine 文件为空: " + path);
    std::vector<char> buf(static_cast<size_t>(n));
    if (!f.read(buf.data(), n)) throw InferError("读 engine 失败: " + path);
    return buf;
}

void cuda_check(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) {
        throw InferError(std::string(what) + " 失败: " + cudaGetErrorString(rc));
    }
}

class TrtEngine : public Engine {
public:
    explicit TrtEngine(const std::string& path) {
        const std::vector<char> blob = read_file(path);
        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_) throw InferError("createInferRuntime 失败");
        engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
        if (!engine_) throw InferError("engine 反序列化失败: " + path);
        context_ = engine_->createExecutionContext();
        if (!context_) throw InferError("createExecutionContext 失败");

        const int n = engine_->getNbBindings();
        host_in_.clear();
        buffers_.assign(static_cast<size_t>(n), nullptr);
        host_out_.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const char* name = engine_->getBindingName(i);
            const nvinfer1::Dims dims = engine_->getBindingDimensions(i);
            const nvinfer1::DataType dt = engine_->getBindingDataType(i);
            if (dt != nvinfer1::DataType::kFLOAT) {
                throw InferError(std::string("binding ") + name +
                                 " 不是 float32（本程序只支持 float I/O）");
            }
            const int64_t count = volume(dims);
            std::printf("[TRT] %-12s %s  %s\n",
                        engine_->bindingIsInput(i) ? "输入" : "输出", name,
                        dims_string(dims).c_str());
            cuda_check(cudaMalloc(&buffers_[static_cast<size_t>(i)],
                                  static_cast<size_t>(count) * sizeof(float)),
                       "cudaMalloc");
            if (engine_->bindingIsInput(i)) {
                if (input_ >= 0) throw InferError("engine 有多个输入，本程序只支持一个");
                input_ = i;
                host_in_.assign(static_cast<size_t>(count), 0.0f);
            } else {
                host_out_[static_cast<size_t>(i)].assign(
                    static_cast<size_t>(count), 0.0f);
            }
            const std::string sname(name);
            if (sname == "images") images_ = i;
            if (sname == "cls_logits") cls_ = i;
            if (sname == "bbox_reg") reg_ = i;
        }
        if (input_ < 0 || images_ < 0) {
            throw InferError("engine 缺少输入 binding 'images'");
        }
        if (cls_ < 0 || reg_ < 0) {
            throw InferError("engine 缺少输出 binding 'cls_logits' / 'bbox_reg'");
        }
        cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~TrtEngine() override {
        if (stream_) cudaStreamDestroy(stream_);
        for (void* p : buffers_) {
            if (p) cudaFree(p);
        }
        delete context_;
        delete engine_;
        delete runtime_;
    }

    void infer(const std::vector<float>& input, std::vector<float>* cls_logits,
               std::vector<float>* bbox_reg) override {
        if (input.size() != host_in_.size()) {
            throw InferError("输入尺寸不对: 期望 " + std::to_string(host_in_.size()) +
                             "，收到 " + std::to_string(input.size()));
        }
        std::memcpy(host_in_.data(), input.data(),
                    host_in_.size() * sizeof(float));
        cuda_check(cudaMemcpyAsync(buffers_[static_cast<size_t>(input_)],
                                   host_in_.data(),
                                   host_in_.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync(H2D)");
#if NV_TENSORRT_MAJOR >= 10
        for (int i = 0; i < static_cast<int>(buffers_.size()); ++i) {
            context_->setTensorAddress(engine_->getBindingName(i),
                                       buffers_[static_cast<size_t>(i)]);
        }
        const bool ok = context_->enqueueV3(stream_);
#else
        std::vector<void*> bindings = buffers_;
        const bool ok = context_->enqueueV2(bindings.data(), stream_, nullptr);
#endif
        if (!ok) throw InferError("TensorRT enqueue 执行失败");
        cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

        auto copy_out = [&](int idx, std::vector<float>* out) {
            auto& host = host_out_[idx];
            cuda_check(cudaMemcpyAsync(host.data(),
                                       buffers_[static_cast<size_t>(idx)],
                                       host.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_),
                       "cudaMemcpyAsync(D2H)");
            cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
            *out = host;
        };
        copy_out(cls_, cls_logits);
        copy_out(reg_, bbox_reg);
    }

private:
    TrtLogger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> buffers_;
    std::vector<float> host_in_;
    std::vector<std::vector<float>> host_out_;
    int input_ = -1, images_ = -1, cls_ = -1, reg_ = -1;
};

}  // namespace

std::unique_ptr<Engine> make_engine(const std::string& path) {
    return std::unique_ptr<Engine>(new TrtEngine(path));
}

}  // namespace kvc
