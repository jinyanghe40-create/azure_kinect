#!/usr/bin/env bash
# Host (x86) build: core libraries, decoder and unit tests.
# K4A and TensorRT are auto-disabled by CMake when absent, so no SDK is needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build-host}"

cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}" \
    "$@"

cmake --build "$BUILD" -j"$(nproc)"

if [[ "${KVC_SKIP_TESTS:-0}" != "1" ]]; then
    ctest --test-dir "$BUILD" --output-on-failure
fi
