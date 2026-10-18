# Find the Azure Kinect Sensor SDK (C library + k4a.hpp C++ wrapper).
#
# The SDK is expected via K4A_ROOT (see scripts/build_board.sh), installed
# system-wide, or checked out as a sibling directory.

find_path(K4A_INCLUDE_DIR
    NAMES k4a/k4a.hpp k4a/k4a.h
    HINTS
        ${K4A_ROOT}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../azure_kinect_sdk/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../Azure-Kinect-Sensor-SDK/include
        /usr/include
        /usr/local/include
)

set(_k4a_hints /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/lib)
if(K4A_ROOT)
    list(APPEND _k4a_hints ${K4A_ROOT}/lib ${K4A_ROOT}/lib/aarch64)
endif()

find_library(K4A_LIBRARY
    NAMES k4a libk4a.so.1.4.2
    HINTS ${_k4a_hints}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(K4A DEFAULT_MSG K4A_LIBRARY K4A_INCLUDE_DIR)

mark_as_advanced(K4A_INCLUDE_DIR K4A_LIBRARY)
