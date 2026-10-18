# Azure Kinect 视觉链路（C++）

> 在 **Jetson + Azure Kinect DK** 上运行的实时目标检测与 3D 几何位姿估计链路：
> 彩色图检测类别与 2D 框，深度图转点云拟合面心 / 法向 / 边长，
> 统一换算到**车体系**后，通过串口以 **JSON Lines** 发给电控。

![Platform](https://img.shields.io/badge/platform-Jetson%20%7C%20x86-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-8.x-76B900)
![Protocol](https://img.shields.io/badge/serial-JSONL%20rev2-blue)

---

## 它做什么

```
Azure Kinect DK
   │
   ├── 彩色图 ──► SSDlite 检测器（TensorRT）──► 类别 + 2D 框
   │
   └── 深度图 ──► 点云 ──► 距离滤波 ──► RANSAC 去桌面 ──► 聚类
                                ──► 平面拟合 ──► 面心 / 法向 / 边长 / 倾角
                                                   │
                        2D 框与 3D 点云配对 ◄──────┘
                                                   │
                                                   ▼
                    车体系坐标（X 前 / Y 左 / Z 上，原点=车体几何中心）
                                                   │
                                                   ▼
                           /dev/ttyGS0 · 115200 · JSON Lines · rev 2
```

- **检测**：SSDlite（320×320，FP16 TensorRT），4 类 —— `tools / medicine / instrument / food`
- **几何**：纯几何法，不训练。RANSAC 平面剔除 → 体素下采样 → 欧式聚类 → 分簇平面拟合 → 法向统一 → 选识别面
- **输出**：面心 `face`、体心 `cube`、单位法向 `normal`、边长 `edge`、与光轴夹角 `angle`、相对竖直倾角 `lean`、质量等级 `quality`
- **坐标系**：相机倒装（`--rotate 180`），标定一次永久使用；输出统一在车体系
- **附带**：标定、投影诊断、IMU 校验、相机体检等 8 个可执行工具

---

## 目录结构

```
.
├── azure_kinect_cpp/
│   ├── include/kvc/            # 接口（types / math / calib / decoder / engine / pipeline / serial …）
│   ├── src/                    # 实现；app_run.cpp 是生产主循环
│   ├── tools/                  # 8 个可执行工具的 main
│   ├── tests/                  # 无框架单测 + fixtures
│   ├── scripts/                # build_host.sh / build_board.sh / compare_python_cpp.sh
│   ├── body_calib.json         # 本车标定（相机 → 车体）
│   └── README.md               # C++ 模块详细说明
├── models/
│   ├── best_trt.onnx           # TensorRT 引擎的源模型
│   ├── best.onnx / best.pt     # 原始导出与训练权重
│   └── history.json            # 训练记录
└── README.md
```

> `models/*.engine`、`build/`、`build-*/` 不入库，需在目标机上生成。

---

## 环境要求

| 项目 | 要求 |
|---|---|
| 目标平台 | Jetson（Ubuntu / JetPack，ARM64）或 x86 Linux |
| 相机 | Azure Kinect DK（**深度必须接自带电源适配器**） |
| 推理 | TensorRT 8.x + CUDA（随 JetPack） |
| 相机 SDK | Azure Kinect SDK 1.4.2，板上建议放在 `~/work/azure_kinect_sdk` |
| 构建 | CMake ≥ 3.10、C++17、Eigen3、nlohmann/json、OpenCV |

电脑上不装 K4A / TensorRT 也能编译：CMake 自动探测失败时生成桩，
`KVC_HAVE_K4A=0` / `KVC_HAVE_TRT=0`，核心库和单测照常运行。

> **SSH 里相机开不了不是供电问题**：深度引擎需要 GLX，必须有 `DISPLAY`。
> 用 `export DISPLAY=:0` 或 `DISPLAY=:1`。而如果深度相机对应的 USB 设备
> （`045e:097c`）根本不枚举，才去查外接电源。

---

## 构建

```bash
# 电脑（x86）：产物在 build-host/，脚本最后自动跑 ctest
cd azure_kinect_cpp
bash scripts/build_host.sh
```

```bash
# 板子：产物在 azure_kinect_cpp/build/
bash ~/work/azure_kinect_cpp/scripts/build_board.sh
```

板上脚本会在 `~/work/azure_kinect_sdk`、`../azure_kinect_sdk`、
`~/Azure-Kinect-Sensor-SDK` 中自动查找 SDK，也可以显式指定：

```bash
K4A_DIR=/path/to/azure-kinect-sensor-sdk bash scripts/build_board.sh
```

构建完成后，脚本会把 `azure_kinect_cpp/body_calib.json` 复制到二进制旁边。

### 生成 TensorRT 引擎

仓库不包含 `best_trt.engine`（与机器和 TensorRT 版本绑定），在板上生成：

```bash
cd ~/work
/usr/src/tensorrt/bin/trtexec --onnx=models/best_trt.onnx --fp16 \
  --saveEngine=models/best_trt.engine
```

程序默认按以下顺序查找引擎：

```
<可执行文件目录>/models/best_trt.engine
<可执行文件目录>/../models/best_trt.engine
<可执行文件目录>/../../models/best_trt.engine
$K4A_ENGINE
```

也可以用 `--engine PATH` 显式指定。

---

## 标定（一次，需要卷尺）

相机刚性固定在车上，标定一次永久使用。没有标定文件时程序**默认拒绝启动**，
这是刻意的：静默退回相机系会让所有坐标整体偏约 0.6 m 且不报错。

```bash
cd ~/work/azure_kinect_cpp

# 1) 量完车心 / 前向 / 光心位置和俯仰，生成标定
./build/body_calib_cpp --write body_calib.json \
    --pitch 9.7 --forward 250 --left 0 --height 600 --id buggy01-20261015

# 2) 校验正交性 / det / 竖直判据 / 数值区间
./build/body_calib_cpp --check body_calib.json

# 3) 物理验收 C/D/F —— 唯一能抓住 180° 重复计入和矩阵转置写反的三关
DISPLAY=:1 ./build/check_body_frame_cpp --rotate 180
```

`--calib` 的默认查找顺序：`$K4A_BODY_CALIB` → 二进制目录 → 仓库根目录
（`azure_kinect_cpp/body_calib.json`）。

临时排查可用 `--allow-uncalibrated`，此时协议头是 `coord:"camera"`，
**输出不是车体系，不要接电控**。

参数含义与校验规则可直接看 `body_calib_cpp --help`；物理验收的交互式提示
会逐项告诉你该放什么、看什么。

---

## 运行

```bash
cd ~/work/azure_kinect_cpp
DISPLAY=:0 ./build/run_azure_cpp --serial-dry --frames 20 \
  --rotate 180 --calib body_calib.json --no-gravity \
  --stride 2 --min-dist 0.3 --max-dist 1.5 \
  --min-cluster-points 200 --only-classified --mm --print
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--serial-dry` | 把要发给电控的 JSONL 打到 stdout |
| `--serial /dev/ttyGS0 --baud 115200` | 真发串口 |
| `--print` | 终端打印结果 |
| `--json` | 逐帧调试 JSON（优先于 `--print`） |
| `--profile` | 退出时打印各阶段耗时 |
| `--mm` | `obj[]` 内长度字段改用毫米 |
| `--calib PATH` | 指定标定文件 |
| `--allow-uncalibrated` | 无标定也启动（退化模式） |
| `--rotate 0/90/180/270` | 相机安装旋转，**必须与标定文件里的 `image_rotate_deg` 一致** |
| `--help` | 查看全部参数 |

> **`--rotate 180` 不能省。** 本车相机倒装，漏掉它图像倒着（看得见），
> 但点云会**静默地**错掉。它还与标定文件里的 `image_rotate_deg` 互相锁死。
>
> ROS 相关参数（`--ros*`）当前**未实现**，传入会明确报错。

---

## 工具

| 工具 | 作用 |
|---|---|
| `run_azure_cpp` | 生产主循环：取帧 → 检测 → 几何 → 串口 |
| `body_calib_cpp` | `--write / --check / --verify / --pitch-from-target` |
| `check_body_frame_cpp` | 物理验收 C（地面标点）/ D（IMU 交叉验证）/ F（3 点卷尺比对） |
| `diag_match_cpp` | 检测框 + 3D 投影叠加，写 `/tmp/diag_match.jpg` |
| `check_projection_cpp` | 面心 / 体心投影十字，写 `/tmp/proj_check.jpg` |
| `check_gravity_cpp` | 读 IMU 判断轴向；不开相机，无 `DISPLAY` 也能跑 |
| `k4a_probe_cpp` | 设备数 / open / 序列号 / 真开一路流 |
| `verify_libs_cpp` | dlopen 三个库，打印架构 / 依赖 / symbol 问题 |
| `geom_replay` | 点云 PLY + 框 JSON → 几何 JSON，离线对比用 |

```bash
./build/check_body_frame_cpp --rotate 180
./build/check_body_frame_cpp --only d
./build/diag_match_cpp --rotate 180
./build/check_projection_cpp --rotate 180
```

`check_body_frame_cpp` 会交互式提示（放方块、按回车）。C 失败时会打印
“反号 / 转置 / 180° 重复计入”的指纹判读。

---

## 输出协议（rev 2）

一行一个 JSON（JSON Lines），`/dev/ttyGS0`，115200：

```json
{"t":12.345,"frame":123,"seq":45,"n":1,
 "coord":"body","rev":2,"cam":[0.25,0.0,0.6],"cam_unit":"m",
 "calib":"buggy01-20261015","obj_unit":"mm","infer_ms":31.2,
 "obj":[
   {"cls_id":1,"cls":"medicine","score":0.69,
    "face":[1100.0,-300.0,550.0],
    "normal":[0.9857,0.0,-0.1685],
    "cube":[1200.0,-200.0,650.0],
    "edge":335.0,"edge_src":"auto",
    "n_faces":3,"pos_type":"cube","cube_ok":true,
    "range":902.8,"az":19.52,"el":5.97,"nearest":950.0,
    "quality":"GOOD","n_face_pts":1723,
    "angle":48.2,"lean":7.3,"rms_mm":0.9}]}
```

| 字段 | 含义 |
|---|---|
| `coord` | `"body"` = 车体系；`"camera"` = 未标定的退化模式，**别接电控** |
| `rev` | 协议版本，当前为 `2` |
| `cam` / `cam_unit` | 相机光心在车体系的位置，恒为米 |
| `obj_unit` | **只管 `obj[]` 内的长度字段**（`"m"` / `"mm"`） |
| `face` | ★ 面心，平面拟合的直接结果（主数据） |
| `cube` | 体心，由三面推出；看 `cube_ok` |
| `normal` | 面朝向单位向量，无量纲，不随 `obj_unit` 变 |
| `edge` | 实测边长 |
| `range` / `az` / `el` / `nearest` | **相机光学系**下的距离 / 方位 / 俯仰 / 最近点 |
| `angle` | 法向与**相机光轴**夹角（含相机俯仰） |
| `lean` | 法向偏离**车体竖直**的角度：0°=铅垂面，90°=水平面 |
| `quality` | `GOOD` / `FAIR` / `COARSE`，由面点数、距离、拟合残差决定 |

**判读建议**：定位优先用 `face + normal`；`cube` 的误差会被 `edge/2` 放大
（实测抖动 `face ≈ 5 mm`、`cube ≈ 14 mm`）。

---

## 测试

```bash
# 主机编译并运行全部单测（build_host.sh 自动执行 ctest）
bash azure_kinect_cpp/scripts/build_host.sh

# 几何离线回放：PLY + 框 JSON → 几何 JSON
./build-host/geom_replay --ply azure_kinect_cpp/tests/fixtures/geom/synth4_default.ply \
  --boxes azure_kinect_cpp/tests/fixtures/geom/grid_split_on.boxes.json

# 物理验收 C/D/F
./build/check_body_frame_cpp --rotate 180
```

`tests/fixtures/` 中已包含各场景的点云、框和期望结果，测试不依赖相机。
fixtures 的生成脚本（`tests/gen_*.py`）需要外部的 Python 参考实现，
日常编译和测试不需要运行它们。

如需和 Python 参考实现做同场景对比：

```bash
PY=/path/to/run_azure.py bash azure_kinect_cpp/scripts/compare_python_cpp.sh 20
```

两个程序会**先后**运行（相机同一时刻只能被一个进程占用），
输出在 `/tmp/kvc-compare/`，按中心距、角度、分数差异给出判定。

---

## 性能

| 指标 | 数值（Jetson 实测） |
|---|---|
| 端到端稳态帧率 | ≈ 3 FPS |
| 几何阶段 | ≈ 147 ms（最大头） |
| 检测器单独推理 | ≈ 32 FPS |
| 彩色分辨率 | 1440p 是拐点；4K 明显掉帧 |

`--profile` 会打印等帧 / SDK 点云 / 解码 / 检测 / 几何 / 配对各段耗时。

---

## 已知限制

- **检测距离**：现有模型训练集只有 0.4 m / 1.0 m，1 m 以外召回明显下降；
  anchor 下限 64 px。
- **紧贴的方块**：点云上会合并成一簇，已用 2D 框切簇缓解，但会裁掉 2–4 cm 面。
- **模型引擎**：`models/best_trt.engine` 不入库，需在板上由 `best_trt.onnx`
  生成，且必须与板上 TensorRT 版本匹配。

---

## 一句话原则

> 相机只负责给“深度图 + 内参”，中间用**点云**这个标准格式隔开；
> 换相机只换取帧层，几何算法与串口输出契约保持不变。
