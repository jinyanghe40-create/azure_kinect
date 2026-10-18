# Find TensorRT 8.x and the CUDA runtime.
#
# JetPack installs NvInfer.h under /usr/include/aarch64-linux-gnu and the
# libraries under /usr/lib/aarch64-linux-gnu.  No pkg-config / CMake config.

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        ${TENSORRT_ROOT}/include
        /usr/include/aarch64-linux-gnu
        /usr/include/x86_64-linux-gnu
        /usr/local/cuda/include
        /usr/include
)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    HINTS ${TENSORRT_ROOT}/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib
)

find_library(TENSORRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    HINTS ${TENSORRT_ROOT}/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib
)

find_library(CUDA_RUNTIME_LIBRARY
    NAMES cudart
    HINTS /usr/local/cuda/lib64 /usr/local/cuda/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
)
find_path(CUDA_INCLUDE_DIR
    NAMES cuda_runtime_api.h
    HINTS /usr/local/cuda/include /usr/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT DEFAULT_MSG
    TENSORRT_LIBRARY TENSORRT_INCLUDE_DIR CUDA_RUNTIME_LIBRARY CUDA_INCLUDE_DIR)

mark_as_advanced(TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY TENSORRT_PLUGIN_LIBRARY
                 CUDA_RUNTIME_LIBRARY CUDA_INCLUDE_DIR)
