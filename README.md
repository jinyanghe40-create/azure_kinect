# Azure Kinect 视觉链路

> 在 **Jetson Nano + Azure Kinect DK** 上跑的目标检测与 3D 几何位姿估计链路：
> 彩色图检测类别与 2D 框，深度图转点云拟合面心 / 法向 / 边长，
> 统一换算到**车体系**后，通过串口以 **JSON Lines** 发给电控。

![Platform](https://img.shields.io/badge/platform-Jetson%20Nano%20%7C%20x86-76B900?logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.6%2B-3776AB?logo=python&logoColor=white)
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
- **双实现**：Python 生产链路 + 行为等价的 C++ 重写（`azure_kinect_cpp/`）

---

## 目录结构

```
.
├── common/                 # 相机无关的算法层（两个相机共用）
│   ├── trt_infer.py        # 主检测器：图像/视频 → 类别 + 2D 框
│   ├── pc_pipeline.py      # 几何管线：点云 → 去桌面 → 聚类 → 拟合面 → 夹角
│   ├── ssd_decoder.py      # SSD 后处理（anchor / 解码 / softmax / NMS），纯 numpy
│   ├── depth_to_cloud.py   # 深度图 + 内参 → 点云（解耦的关键一层）
│   ├── serial_out.py       # 串口输出线程（JSON Lines，不阻塞推理）
│   └── lab1_ransac.py / lab2_pointcloud.py / lab3_angle.py   # 原理解析脚本
│
├── azure_kinect/           # Azure Kinect DK 适配层与部署
│   ├── run_azure.py        # ★ 主入口：取帧 → 检测 → 几何 → 输出
│   ├── azure_capture.py    # k4a ctypes 取帧、坐标变换、车体系断言
│   ├── body_calib.py       # 标定文件生成 / 校验 / 报告
│   ├── check_body_frame.py # 物理验收 C（地面标点）/ D（IMU）/ F（卷尺）
│   ├── k4a_probe.py        # 相机体检
│   ├── install_vision_service.sh   # 安装 systemd 服务 k4a-vision
│   ├── install_x_service.sh        # 安装裸 Xorg 服务 k4a-xorg
│   └── libs/ include/      # Azure Kinect SDK 1.4.2 ARM64 运行库与头文件
│
├── azure_kinect_cpp/       # C++ 重写（生产链路 + 诊断工具）
│   ├── include/kvc/        # 接口定义
│   ├── src/                # 实现，app_run.cpp 是生产主循环
│   ├── tools/              # 8 个可执行工具
│   ├── tests/              # 无框架单测 + Python 生成的 fixtures
│   └── scripts/            # build_host.sh / build_board.sh / compare_python_cpp.sh
│
├── realsense/              # RealSense D435i 适配层（备用相机方案）
├── models/                 # best.pt / best.onnx / best_trt.onnx
├── tools/                  # 精度测试、协议回归、姿态扫描等 PC 侧脚本
├── ros_bridge/             # TCP → ROS 2 桥接 + RViz 配置
├── diag_*.py               # 帧预算 / 分辨率 / 回环 / 匹配等诊断脚本
└── board.sh                # 电脑 ↔ 板子：IP / sshfs 挂载 / 推拉 / 远程执行
```

---

## 环境要求

| 项目 | 要求 |
|---|---|
| 目标平台 | Jetson Nano（Ubuntu 18.04 / JetPack），ARM64 |
| 相机 | Azure Kinect DK（**深度必须接自带电源适配器**） |
| 推理 | TensorRT 8.x + CUDA（随 JetPack） |
| 相机 SDK | Azure Kinect SDK 1.4.2，仓库内已带 ARM64 库（`azure_kinect/libs`） |
| Python | 3.6+，`numpy`、`opencv-python`、`pyserial`（`matplotlib` 仅部分工具需要） |
| 电脑侧（可选） | Ubuntu + ROS 2 Humble（RViz 可视化、精度测试） |
| C++ 版 | CMake ≥ 3.10、Eigen3、nlohmann/json、OpenCV；无 K4A/TRT 时自动生成桩 |

> **SSH 里相机开不了不是供电问题**：深度引擎需要 GLX，必须有 `DISPLAY`。
> 用 `run_with_x.sh` 或先 `export DISPLAY=:0`。
> 而如果深度相机对应的 USB 设备（`045e:097c`）根本不枚举，才去查外接电源。

---

## 快速开始

### 板子上（真跑）

```bash
# 一次性：udev 权限 + X 服务 + 开机自启
sudo bash ~/work/azure_kinect/k4a_udev_setup.sh
sudo bash ~/work/azure_kinect/install_x_service.sh
sudo bash ~/work/azure_kinect/install_vision_service.sh --start

# 日常：一键启动（自动确保 X、日志写到 ~/work/logs/）
bash ~/work/azure_kinect/start_azure.sh
```

手动跑（例如还没标定时）：

```bash
bash ~/work/azure_kinect/start_azure.sh --print --frames 100 --allow-uncalibrated
```

或直接调用入口：

```bash
sudo bash ~/work/azure_kinect/run_with_x.sh \
  python3 ~/work/azure_kinect/run_azure.py \
  --print --frames 30 --stride 2 --rotate 180 --no-gravity
```

常用输出开关：

| 参数 | 作用 |
|---|---|
| `--print` | 终端打印结果 |
| `--json` | 逐帧调试 JSON（优先于 `--print`） |
| `--serial-dry` | 把发给电控的 JSONL 打到 stdout |
| `--serial /dev/ttyGS0` | 真发串口（默认 115200） |
| `--ros HOST:PORT` | 发给电脑上的 ROS 桥接 |
| `--save-dir out` | 存可视化图 |
| `--mm` | `obj[]` 内长度字段改用毫米 |
| `--profile` | 退出时打印各阶段耗时（C++ 版） |

> **`--rotate 180` 不能省。** 本车相机倒装，漏掉它图像倒着（看得见），
> 但点云会**静默地**错掉。它还与标定文件里的 `image_rotate_deg` 互相锁死。

### 标定（一次，需要卷尺）

```bash
# 1) 量完车心 / 前向 / 光心位置和俯仰，生成标定
python3 ~/work/azure_kinect/body_calib.py --write ~/work/azure_kinect/body_calib.json \
        --pitch 9.7 --forward 250 --left 0 --height 600 --id buggy01-20261015

# 2) 校验正交性 / det / 竖直判据 / 数值区间
python3 ~/work/azure_kinect/body_calib.py --check ~/work/azure_kinect/body_calib.json

# 3) 物理验收 C/D/F —— 唯一能抓住 180° 重复计入和矩阵转置写反的三关
DISPLAY=:1 python3 ~/work/azure_kinect/check_body_frame.py --rotate 180
```

**没有标定文件时程序默认拒绝启动（exit 1）**，这是刻意的：
静默退回相机系会让所有坐标整体偏约 0.6 m 且不报错。临时排查用
`--allow-uncalibrated`，此时协议头是 `coord:"camera"`，**不要接电控**。

标定参数的完整含义与校验规则，运行 `python3 ~/work/azure_kinect/body_calib.py --help`
查看；物理验收的交互式提示会逐项告诉你该放什么、看什么。

### 电脑上（可选）

```bash
# 板子 ~/work 挂到本地 ~/board，VS Code 直接改板子上的文件
./board.sh ip
./board.sh mount

# 读串口结果并自动判定
python3 read_serial.py --check

# 离线回归（不用相机、不用板子）
python3 tools/test_body_frame_math.py
python3 tools/test_zero_regression.py
python3 tools/test_output_contract.py

# 精度测试 / 姿态扫描（需要 ROS 2 桥接在跑）
source /opt/ros/humble/setup.bash
python3 tools/test_precision.py --frames 100 --true-edge 0.25
python3 tools/sweep_pose.py --true-edge 0.25
```

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

## Python 版与 C++ 版

`azure_kinect_cpp/` 是生产链路的 C++ 重写，**串口契约与 Python 版完全一致**
（字段名、单位、语义、顺序都不变），电控侧无需改动。

```bash
# 电脑（x86）
bash azure_kinect_cpp/scripts/build_host.sh     # 产物在 build-host/，自动跑 ctest

# 板子
bash ~/work/azure_kinect_cpp/scripts/build_board.sh
~/work/azure_kinect_cpp/build/run_azure_cpp --serial-dry --frames 20 \
  --rotate 180 --calib ~/work/azure_kinect/body_calib.json --no-gravity --print

# Python / C++ 同场景对比（相机同一时刻只能被一个进程占用，会先后跑）
bash azure_kinect_cpp/scripts/compare_python_cpp.sh 20
```

C++ 版同时提供 `body_calib_cpp`、`check_body_frame_cpp`、`diag_match_cpp`、
`check_projection_cpp`、`check_gravity_cpp`、`k4a_probe_cpp`、`verify_libs_cpp`、
`geom_replay` 八个工具，与 Python 版逐项对应。

**验证状态**：主机（含无 K4A / 无 TRT 的桩配置）编译通过，`ctest` 5/5；
几何管线 7 个场景与 Python oracle 在容差内一致；`body_calib_cpp` 与 Python 版
输出逐字节一致。切换 `k4a-vision.service` 到 C++ 之前，仍需完成板上同场景对比验收。

---

## 测试与验收

| 类别 | 命令 | 说明 |
|---|---|---|
| 数学零回归 | `tools/test_body_frame_math.py` | 闭式投影外参与旧实现逐位一致 |
| 几何零回归 | `tools/test_zero_regression.py` | 与改动前快照字符串相等 |
| 输出契约 | `tools/test_output_contract.py` | 串口/ROS 字段与单位检查 |
| 精度 / 重复性 | `tools/test_precision.py` | 面心抖动、尺度、方位角 |
| 姿态鲁棒性 | `tools/sweep_pose.py` | 不同倾角下的检测/拟合表现 |
| C++ 单测 | `ctest`（build_host.sh 自动执行） | SSD 解码、几何、标定、串口契约 |
| 物理验收 C/D/F | `azure_kinect/check_body_frame.py` | 卷尺标点 / IMU 交叉验证 / 3 点比对 |

---

## 性能（Jetson Nano 实测）

| 指标 | 数值 |
|---|---|
| 端到端稳态帧率 | ≈ 2.98 FPS |
| 几何阶段 | ≈ 147 ms（最大头） |
| 检测器单独推理 | ≈ 32 FPS |
| 彩色分辨率 | 1440p 是拐点；4K 掉到 1.71 FPS（解码约 36 ms/MP） |

C++ 版用 `--profile` 打印等帧 / SDK 点云 / 解码 / 检测 / 几何 / 配对各段耗时。

---

## 已知限制

- **检测距离**：现有模型训练集只有 0.4 m / 1.0 m，1 m 以外召回明显下降；
  anchor 下限 64 px。
- **紧贴的方块**：点云上会合并成一簇，已用 2D 框切簇缓解，但会裁掉 2–4 cm 面。
- **模型引擎**：`models/best_trt.engine` 不入库，需在板上由 `best_trt.onnx` 编译
  （`common/trt_infer.py --rebuild`）。

---

## 相关文档

| 文档 | 内容 |
|---|---|
| [`azure_kinect_cpp/README.md`](azure_kinect_cpp/README.md) | C++ 版构建、工具、已知差异 |
| [`ros_bridge/使用说明.md`](ros_bridge/使用说明.md) | TCP → ROS 2 桥接与 RViz |

---

## 一句话原则

> 相机只负责给"深度图 + 内参"，中间用**点云**这个标准格式隔开；
> 换相机只换取帧脚本，算法与输出契约一行不改。
