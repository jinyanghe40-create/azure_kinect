cmake_minimum_required(VERSION 3.10)
project(azure_kinect_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(KVC_WITH_K4A   "Build the Azure Kinect capture backend" ON)
option(KVC_WITH_TRT   "Build the TensorRT inference backend"  ON)
option(KVC_BUILD_TESTS "Build unit tests" ON)

# ---------------------------------------------------------------------------
# Required everywhere: Eigen + nlohmann/json
# ---------------------------------------------------------------------------
find_package(Eigen3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Core HINTS /usr/include/eigen3 /usr/local/include/eigen3)
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 not found. Install libeigen3-dev.")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
    if(NOT NLOHMANN_JSON_INCLUDE_DIR)
        message(FATAL_ERROR "nlohmann/json.hpp not found. Install nlohmann-json3-dev.")
    endif()
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NLOHMANN_JSON_INCLUDE_DIR}")
endif()

# serial_out.cpp uses std::thread.  glibc >= 2.34 folds pthread into libc, but
# the Jetson (Ubuntu 18.04, glibc 2.27) still needs -pthread explicitly.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)

# ---------------------------------------------------------------------------
# Optional backends
# ---------------------------------------------------------------------------
if(KVC_WITH_K4A AND NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found -> capture backend disabled")
    set(KVC_WITH_K4A OFF)
endif()

if(KVC_WITH_K4A)
    find_package(K4A QUIET)
    if(K4A_FOUND)
        message(STATUS "K4A: ${K4A_LIBRARY} (${K4A_INCLUDE_DIR})")
    else()
        message(STATUS "K4A not found -> capture backend disabled (core/tests still build)")
        set(KVC_WITH_K4A OFF)
    endif()
endif()

if(KVC_WITH_TRT)
    find_package(TensorRT QUIET)
    if(TensorRT_FOUND)
        message(STATUS "TensorRT: ${TENSORRT_LIBRARY} (${TENSORRT_INCLUDE_DIR})")
    else()
        message(STATUS "TensorRT not found -> TRT backend disabled (decoder tests still build)")
        set(KVC_WITH_TRT OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# kvc_core: math, calibration, json helpers, serial writer
# ---------------------------------------------------------------------------
add_library(kvc_core STATIC
    src/math.cpp
    src/body_calib.cpp
    src/json_util.cpp
    src/serial_out.cpp
    src/match.cpp
    src/cli.cpp
)
target_include_directories(kvc_core PUBLIC include)
target_link_libraries(kvc_core PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json Threads::Threads)

# ---------------------------------------------------------------------------
# kvc_infer: preprocessing + SSD decoder (OpenCV optional), TRT engine optional
# ---------------------------------------------------------------------------
add_library(kvc_infer STATIC
    src/preprocess.cpp
    src/ssd_decoder.cpp
)
target_include_directories(kvc_infer PUBLIC include)
target_link_libraries(kvc_infer PUBLIC kvc_core)
if(OpenCV_FOUND)
    target_link_libraries(kvc_infer PUBLIC ${OpenCV_LIBS})
    target_include_directories(kvc_infer PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(kvc_infer PUBLIC KVC_HAVE_OPENCV=1)
else()
    target_compile_definitions(kvc_infer PUBLIC KVC_HAVE_OPENCV=0)
endif()

if(KVC_WITH_TRT)
    target_sources(kvc_infer PRIVATE src/trt_engine.cpp)
    target_link_libraries(kvc_infer PUBLIC ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY} ${CUDA_RUNTIME_LIBRARY})
    target_include_directories(kvc_infer PUBLIC ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
    target_compile_definitions(kvc_infer PUBLIC KVC_HAVE_TRT=1)
else()
    target_sources(kvc_infer PRIVATE src/trt_engine_stub.cpp)
    target_compile_definitions(kvc_infer PUBLIC KVC_HAVE_TRT=0)
endif()

# ---------------------------------------------------------------------------
# kvc_geom: point-cloud geometry pipeline (pure Eigen)
# ---------------------------------------------------------------------------
add_library(kvc_geom STATIC
    src/pc_pipeline.cpp
    src/geom_ransac.cpp
    src/geom_cluster.cpp
    src/geom_split.cpp
    src/geom_faces.cpp
)
target_include_directories(kvc_geom PUBLIC include)
target_link_libraries(kvc_geom PUBLIC kvc_core)

# ---------------------------------------------------------------------------
# kvc_capture: Azure Kinect backend
# ---------------------------------------------------------------------------
if(KVC_WITH_K4A)
    add_library(kvc_capture STATIC src/azure_source.cpp)
    target_include_directories(kvc_capture PUBLIC include ${K4A_INCLUDE_DIR})
    target_link_libraries(kvc_capture PUBLIC kvc_core ${K4A_LIBRARY})
    # OpenCV is required for MJPG decode and the cv::Mat frame container.
    target_link_libraries(kvc_capture PUBLIC ${OpenCV_LIBS})
    target_include_directories(kvc_capture PUBLIC ${OpenCV_INCLUDE_DIRS})
    # The vendored k4a_export.h was generated on Windows.  Neutralise its
    # __declspec macros on Linux (they are all #ifndef-guarded).
    target_compile_definitions(kvc_capture PUBLIC
        KVC_HAVE_K4A=1 KVC_HAVE_OPENCV=1 K4A_STATIC_DEFINE=1
        K4A_DEPRECATED= K4A_DEPRECATED_EXPORT= K4A_DEPRECATED_NO_EXPORT=)
else()
    add_library(kvc_capture STATIC src/azure_source_stub.cpp)
    target_include_directories(kvc_capture PUBLIC include)
    target_link_libraries(kvc_capture PUBLIC kvc_core)
    if(OpenCV_FOUND)
        # The stubbed header still declares cv::Mat frame fields.
        target_include_directories(kvc_capture PUBLIC ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(kvc_capture PUBLIC ${OpenCV_LIBS})
        target_compile_definitions(kvc_capture PUBLIC
            KVC_HAVE_K4A=0 KVC_HAVE_OPENCV=1)
    else()
        target_compile_definitions(kvc_capture PUBLIC
            KVC_HAVE_K4A=0 KVC_HAVE_OPENCV=0)
    endif()
endif()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(body_calib_cpp tools/main_body_calib.cpp)
target_link_libraries(body_calib_cpp PRIVATE kvc_core)

add_executable(geom_replay tools/geom_replay.cpp)
target_link_libraries(geom_replay PRIVATE kvc_geom kvc_core)

if(OpenCV_FOUND)
    # The app touches cv::Mat frames and writes JPEG snapshots.
    add_executable(run_azure_cpp
        tools/main_run_azure.cpp
        src/app_run.cpp)
    target_link_libraries(run_azure_cpp PRIVATE kvc_core kvc_infer kvc_geom kvc_capture)
else()
    message(STATUS "OpenCV not found -> run_azure_cpp skipped")
endif()

foreach(tool check_body_frame diag_match check_projection check_gravity k4a_probe verify_libs)
    add_executable(${tool}_cpp tools/main_${tool}.cpp)
    target_link_libraries(${tool}_cpp PRIVATE kvc_core kvc_infer kvc_geom kvc_capture)
endforeach()

# verify_libs uses dlopen/dlsym; older glibc needs libdl explicitly.
target_link_libraries(verify_libs_cpp PRIVATE ${CMAKE_DL_LIBS})

# ---------------------------------------------------------------------------
# Tests (no external framework; each test is an executable returning 0/1)
# ---------------------------------------------------------------------------
if(KVC_BUILD_TESTS)
    enable_testing()
    foreach(t test_math test_body_calib test_ssd_decoder test_serial_contract test_pc_pipeline)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/${t}.cpp")
            add_executable(${t} tests/${t}.cpp)
            target_link_libraries(${t} PRIVATE kvc_core kvc_infer kvc_geom)
            target_compile_definitions(${t} PRIVATE
                KVC_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
            add_test(NAME ${t} COMMAND ${t})
        endif()
    endforeach()
endif()
