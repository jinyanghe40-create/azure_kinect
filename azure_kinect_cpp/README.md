# azure_kinect_cpp

Azure Kinect 视觉链路的 C++ 重写：**生产运行链路 + 验收/诊断工具**。
串口输出契约与 Python 版**完全一致**（字段名/单位/语义/顺序都不变），
电控侧一行都不用改。

```
相机 → 点云 → SSD 检测 → 3D 几何 → 车体系坐标 → 串口 JSONL
```

与 Python 版并存：`run_azure_cpp` 可以单独跑；`k4a-vision.service`
**本次没有改动**，切换是后面单独一步（见文末）。

## 依赖

| 用途 | 依赖 |
|---|---|
| 全平台 | C++17、CMake ≥ 3.10、Eigen3、nlohmann/json、OpenCV |
| 板上 | Azure Kinect SDK 1.4.2（`~/work/azure_kinect_sdk`）、TensorRT 8.x + CUDA |

电脑上不装 K4A / TRT 也能编：CMake 自动探测失败时生成桩，
`KVC_HAVE_K4A=0` / `KVC_HAVE_TRT=0`，核心库和单测照常。

## 构建与测试

```bash
# 电脑
bash scripts/build_host.sh
# 产物在 build-host/，脚本最后自动跑 ctest
```

```bash
# 板子（在板上执行）
bash ~/work/azure_kinect_cpp/scripts/build_board.sh
# 产物在 ~/work/azure_kinect_cpp/build/
```

板上构建脚本会把 `../azure_kinect/body_calib.json` 复制到二进制旁边，
`--calib` 的默认查找顺序是 `$K4A_BODY_CALIB` → 二进制目录 → `../azure_kinect/`。

## 运行

参数与 `azure_kinect/run_azure.py` 一一对应：

```bash
./build/run_azure_cpp --serial-dry --frames 20 \
  --rotate 180 --calib ../azure_kinect/body_calib.json --no-gravity \
  --stride 2 --min-dist 0.3 --max-dist 1.5 \
  --min-cluster-points 200 --only-classified --mm --print
```

- `--serial-dry` 把 JSONL 打到 stdout；`--serial /dev/ttyUSB0` 真发。
- `--json` 输出逐帧调试 JSON（优先于 `--print`），字段与 Python 版相同。
- `--profile` 退出时打印各阶段耗时（替代 Python 的 `diag_framebudget*`）。
- `--help` 看全部参数。ROS 相关参数（`--ros*`）**不实现**，传了会明确报错。

`DISPLAY` 说明：深度引擎需要 GLX，ssh 里没有 `DISPLAY` 时相机开不起来。
先 `export DISPLAY=:0`（或通过 `run_with_x.sh` 启动），这不是供电问题。

## 验收/诊断工具

| 工具 | 对应 Python | 作用 |
|---|---|---|
| `body_calib_cpp` | `body_calib.py` | `--write/--check/--verify/--pitch-from-target`，行为一致 |
| `check_body_frame_cpp` | `check_body_frame.py` | 物理验收 C（卷尺标点）/ D（IMU 交叉验证）/ F（3 点卷尺比对） |
| `diag_match_cpp` | `diag_match.py` | 检测框 + 3D 投影叠加，写 `/tmp/diag_match.jpg` |
| `check_projection_cpp` | `check_projection.py` | 面心/体心投影十字，写 `/tmp/proj_check.jpg` |
| `check_gravity_cpp` | `check_gravity.py` | 读 IMU，判断哪根轴朝上；不开相机，无 DISPLAY 也能跑 |
| `k4a_probe_cpp` | `k4a_probe.py` | 设备数 / open / 序列号 / 真开一路流 |
| `verify_libs_cpp` | `verify_libs.py` | dlopen 三个库，打印架构/依赖/symbol 问题 |
| `geom_replay` | `pc_pipeline.py --replay` | 点云 PLY + 框 JSON → 几何 JSON，离线对比用 |

用法：

```bash
./build/check_body_frame_cpp --rotate 180 --calib ../azure_kinect/body_calib.json
./build/check_body_frame_cpp --only d          # 只跑 IMU 交叉验证
./build/diag_match_cpp --rotate 180 --calib ../azure_kinect/body_calib.json
./build/check_projection_cpp --rotate 180 --calib ../azure_kinect/body_calib.json
```

`check_body_frame_cpp` 会交互式提示（放方块、按回车）。C 失败时打印
“反号/转置/180° 重复计入”的指纹判读，和 Python 版一致。

## 验证状态（截至 2026-10-17）

已在电脑（x86）完成：

- 主机与无 SDK（`KVC_WITH_K4A=OFF KVC_WITH_TRT=OFF`）两种配置都能编译，
  `ctest` 5/5 通过。
- 几何管线 7 个场景与 Python oracle 对比在容差内（`tests/test_pc_pipeline`）。
- `body_calib_cpp` 与 Python 版的 `--check / --verify / --write /
  --pitch-from-target` 输出逐字节一致（仅程序名和文件路径不同）。
- 6 个诊断工具在主机上编译运行，无相机时给出明确报错而不是链接/段错误。

待板上验证（M4–M6，本次未做）：相机 + TRT 相关的一切 ——
`run_azure_cpp` 真跑、`diag_match_cpp` / `check_projection_cpp` 的 JPG、
`check_body_frame_cpp` 的 C/D/F、`compare_python_cpp.sh` 报告，
以及是否切换 systemd 服务。

## 与 Python 版对比

```bash
bash scripts/compare_python_cpp.sh 20
```

两个程序**先后**跑同一场景（相机同一时刻只能被一个进程占用），
输出在 `/tmp/kvc-compare/{python,cpp}.jsonl`，报告按类别配对后给出
中心距、夹角/倾角、分数差异，并按计划 M4 的容差给判定
（中心 ≤20 mm、角度 ≤1°、漏/多 ≤10%）。

> 脚本里刻意**不加** `python3 -I`：板上 `-I` 会把 numpy 1.19.5 换成
> 1.13.3 并丢掉 pyserial。

## 已知差异（刻意保留 / 已记录）

1. **`analyze_cluster` 用 400 次 RANSAC**（Python 是 80）。数值更稳，
   但单帧几何耗时更高；这是唯一一处刻意的算法参数偏离。
2. **随机流不同**：C++ 用每调用固定种子的 PCG32，Python 用 numpy PCG64。
   因此对比用容差，不追求逐位一致。
3. **舍入**：Python 是银行家舍入，C++ 用 `nearbyint`（默认
   `FE_TONEAREST`）实现 `python_round`；只用于输出构造，不用于中间量。
   典型半值在单测里锁定。
4. **`--json` 文本格式**：Python `json.dumps(ensure_ascii=False)` 默认在
   `,`/`:` 后带空格；C++ 用 `nlohmann::ordered_json` 紧凑输出后再补空格，
   保证逐字符观感一致（解析结果本来就相同）。
5. **有效点掩码在变换之后计算**：Python 的既有行为（有 `t_out` 时无效点
   可能残留），C++ 原样保留并加注释，避免“顺手修好”改变下游结果。
6. **`check_body_frame.py` 的 C/F 键名 bug**：Python 读
   `o.get("cube_center")` / `o.get("face_center")`，但
   `common/pc_pipeline.py` 实际输出 `cube_center_m` / `face_center_m`，
   两个查找永远拿到 `None`，C/F 在 Python 里会因 0 维数组取下标而崩。
   C++ 版按**本意**取值（优先 `cube_center`，否则 `face_center`），
   所以 C/F 的数值结论无法与 Python 版逐条对照 —— Python 那边先崩了。
   D（IMU 交叉验证）不涉及这两个键，可与 Python 版对照。
7. 生产链路的 `rotate_k` 恒为 0（点云随图像一起转、外参做了共轭），
   `rot_uv` 的其他分支只在离线几何工具里出现。

## 帧预算

`--profile` 打印等帧 / SDK 点云 / 解码 / 检测 / 几何 / 配对各段耗时。
Python 版量到的稳态约 2.98 FPS，其中几何约 147 ms 是大头；C++ 版的
几何同样最重，但 RANSAC 迭代数是 400，比较时注意口径。

## 切换到 C++（本次不执行）

板上验证通过后，把 `k4a-vision.service` 的 `ExecStart` 指向
`/home/<user>/work/azure_kinect_cpp/build/run_azure_cpp`，参数照抄现有的
Python 命令行；回滚就是把 `ExecStart` 改回去并 `systemctl daemon-reload`。
**在并存验证完成前不要改这个 unit 文件。**

## 目录

```
include/kvc/   接口（types/math/calib/json/preprocess/decoder/engine/
               pipeline/serial/source/match/cli/timers）
src/           实现；app_run.cpp 是生产主循环
tools/         8 个可执行工具的 main
tests/         无框架单测（含 Python 生成的 fixtures）
scripts/       build_host.sh / build_board.sh / compare_python_cpp.sh
```
