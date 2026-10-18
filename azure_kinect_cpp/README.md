# azure_kinect_cpp

Azure Kinect 视觉链路的 C++ 实现：**生产运行链路 + 验收/诊断工具**。

```
相机 → 点云 → SSD 检测 → 3D 几何 → 车体系坐标 → 串口 JSONL
```

串口输出为 **rev 2 JSON Lines**，字段定义见仓库根目录
[`README.md`](../README.md#输出协议rev-2)。

## 依赖

| 用途 | 依赖 |
|---|---|
| 全平台 | C++17、CMake ≥ 3.10、Eigen3、nlohmann/json、OpenCV |
| 板上 | Azure Kinect SDK 1.4.2（`~/work/azure_kinect_sdk`）、TensorRT 8.x + CUDA |

电脑上不装 K4A / TRT 也能编：CMake 自动探测失败时生成桩，
`KVC_HAVE_K4A=0` / `KVC_HAVE_TRT=0`，核心库和单测照常运行。
注意桩配置不会编译真实的 `trt_engine.cpp` / `azure_source.cpp`，
这两条路径必须在板上验证。

## 构建与测试

```bash
# 电脑
bash scripts/build_host.sh
# 产物在 build-host/，脚本最后自动跑 ctest
```

```bash
# 板子（在板上执行）
bash ~/work/azure_kinect_cpp/scripts/build_board.sh
# 产物在 build/
```

板上构建脚本会把本目录的 `body_calib.json` 复制到二进制旁边。
`--calib` 的默认查找顺序是 `$K4A_BODY_CALIB` → 二进制目录 → 仓库根目录。

### 生成 TensorRT 引擎

`models/best_trt.engine` 不入库（与机器和 TensorRT 版本绑定），在板上生成：

```bash
cd ~/work
/usr/src/tensorrt/bin/trtexec --onnx=models/best_trt.onnx --fp16 \
  --saveEngine=models/best_trt.engine
```

默认查找路径：`<exe>/models/` → `<exe>/../models/` → `<exe>/../../models/`
→ `$K4A_ENGINE`；也可用 `--engine PATH` 指定。

## 运行

```bash
cd ~/work/azure_kinect_cpp
DISPLAY=:0 ./build/run_azure_cpp --serial-dry --frames 20 \
  --rotate 180 --calib body_calib.json --no-gravity \
  --stride 2 --min-dist 0.3 --max-dist 1.5 \
  --min-cluster-points 200 --only-classified --mm --print
```

- `--serial-dry` 把 JSONL 打到 stdout；`--serial /dev/ttyUSB0` 真发。
- `--json` 输出逐帧调试 JSON（优先于 `--print`）。
- `--profile` 退出时打印各阶段耗时。
- `--help` 查看全部参数。ROS 相关参数（`--ros*`）**未实现**，传入会明确报错。

`DISPLAY` 说明：深度引擎需要 GLX，ssh 里没有 `DISPLAY` 时相机开不起来。
先 `export DISPLAY=:0`，这不是供电问题。

## 验收/诊断工具

| 工具 | 作用 |
|---|---|
| `body_calib_cpp` | `--write/--check/--verify/--pitch-from-target` |
| `check_body_frame_cpp` | 物理验收 C（卷尺标点）/ D（IMU 交叉验证）/ F（3 点卷尺比对） |
| `diag_match_cpp` | 检测框 + 3D 投影叠加，写 `/tmp/diag_match.jpg` |
| `check_projection_cpp` | 面心/体心投影十字，写 `/tmp/proj_check.jpg` |
| `check_gravity_cpp` | 读 IMU，判断哪根轴朝上；不开相机，无 DISPLAY 也能跑 |
| `k4a_probe_cpp` | 设备数 / open / 序列号 / 真开一路流 |
| `verify_libs_cpp` | dlopen 三个库，打印架构/依赖/symbol 问题 |
| `geom_replay` | 点云 PLY + 框 JSON → 几何 JSON，离线对比用 |

用法：

```bash
./build/check_body_frame_cpp --rotate 180
./build/check_body_frame_cpp --only d          # 只跑 IMU 交叉验证
./build/diag_match_cpp --rotate 180
./build/check_projection_cpp --rotate 180
```

`check_body_frame_cpp` 会交互式提示（放方块、按回车）。C 失败时打印
“反号/转置/180° 重复计入”的指纹判读。

## 验证状态

已完成：

- 主机与无 SDK（`KVC_WITH_K4A=OFF KVC_WITH_TRT=OFF`）两种配置都能编译，
  `ctest` 5/5 通过。
- 几何管线 7 个场景与 Python oracle 对比在容差内（`tests/test_pc_pipeline`）。
- `body_calib_cpp` 与 Python 版的 `--check / --verify / --write /
  --pitch-from-target` 输出逐字节一致（仅程序名和文件路径不同）。
- 8 个工具在主机上编译运行，无相机时给出明确报错而不是链接/段错误。

仍需在实际部署环境完成：`run_azure_cpp` 真跑、`diag_match_cpp` /
`check_projection_cpp` 的 JPG、`check_body_frame_cpp` 的 C/D/F，
以及 TensorRT 引擎反序列化与长稳测试。

## 实现说明

1. **`analyze_cluster` 使用 400 次 RANSAC**，数值更稳，单帧几何耗时更高。
2. **随机流固定**：每调用使用固定种子的 PCG32，离线回放可复现。
3. **舍入**：输出构造用 `nearbyint`（默认 `FE_TONEAREST`）实现
   `python_round`，只用于输出，不用于中间量。
4. 生产链路的 `rotate_k` 恒为 0（点云随图像一起转、外参做了共轭），
   `rot_uv` 的其他分支只在离线几何工具里出现。

## 与 Python 参考实现对比

本仓库不包含 Python 实现。若另有一份 Python 版可用来做同场景对比：

```bash
PY=/path/to/run_azure.py bash scripts/compare_python_cpp.sh 20
```

两个程序**先后**跑同一场景（相机同一时刻只能被一个进程占用），
输出在 `/tmp/kvc-compare/{python,cpp}.jsonl`，报告按类别配对后给出
中心距、夹角/倾角、分数差异（容差：中心 ≤20 mm、角度 ≤1°、漏/多 ≤10%）。

## 目录

```
include/kvc/   接口（types/math/calib/json/preprocess/decoder/engine/
               pipeline/serial/source/match/cli/timers）
src/           实现；app_run.cpp 是生产主循环
tools/         8 个可执行工具的 main
tests/         无框架单测（含已生成的 fixtures）
scripts/       build_host.sh / build_board.sh / compare_python_cpp.sh
```
