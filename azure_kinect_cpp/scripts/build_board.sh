#!/usr/bin/env bash
# Board (Jetson aarch64) build.  Run this ON the board from ~/work:
#   bash ~/work/azure_kinect_cpp/scripts/build_board.sh
#
# K4A: the bundled SDK is expected at ~/work/azure_kinect_sdk (or pass
# K4A_DIR=...).  TensorRT/CUDA come from JetPack.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"

if [[ -z "${K4A_DIR:-}" ]]; then
    for cand in "$HOME/work/azure_kinect_sdk" "$ROOT/../azure_kinect_sdk" \
                "$HOME/Azure-Kinect-Sensor-SDK"; do
        if [[ -d "$cand" ]]; then K4A_DIR="$cand"; break; fi
    done
fi

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DKVC_WITH_K4A=ON
    -DKVC_WITH_TRT=ON
)
if [[ -n "${K4A_DIR:-}" ]]; then
    CMAKE_ARGS+=(-DK4A_ROOT="$K4A_DIR")
fi

cmake -S "$ROOT" -B "$BUILD" "${CMAKE_ARGS[@]}" "$@"
# ★ 板子上是 CMake 3.10：并行度必须放在 `--` 之后，直接写 -jN 会报
#   "Unknown argument -j2"（`--parallel` 要 3.12+ 才有）。
cmake --build "$BUILD" -- -j"$(nproc)"

# Put the calibration next to the binary: default_calib_path() looks there
# first, then falls back to ../body_calib.json (the repo copy).
CALIB="$ROOT/body_calib.json"
if [[ -f "$CALIB" ]]; then
    cp -f "$CALIB" "$BUILD/body_calib.json"
    echo "calibration -> $BUILD/body_calib.json"
fi

echo
echo "built: $BUILD/run_azure_cpp"
echo "smoke: $BUILD/run_azure_cpp --serial-dry --frames 20 --rotate 0 --no-gravity"
