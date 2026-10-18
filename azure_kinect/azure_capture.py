#!/usr/bin/env python3
"""Azure Kinect DK 取帧 → 点云（对标 realsense/realsense_capture.py）

为什么单独一个文件、而且接口要和 RealSense 那版【一模一样】：
    整条几何管线（pc_pipeline）是**相机无关**的：
        depth_to_points(depth, intr) → 点云 → pc_pipeline.run_pipeline(pts, args)
    这个文件就是那层"适配器"。只要 get() 返回同样的四元组，
    换相机算法一行都不用改。

★ 为什么用 ctypes 而不是 pyk4a：
    板子上只装了 libk4a（C 库），没有 Python 绑定。
    ctypes 直调最省事，也不用再装东西。

★ 和 RealSense 的关键区别：
    ┌──────────────┬──────────────────┬───────────────────────┐
    │              │ RealSense D435i  │ Azure Kinect DK       │
    ├──────────────┼──────────────────┼───────────────────────┤
    │ 深度分辨率   │ 640x480          │ 640x576 (NFOV 不合并) │
    │ 深度单位     │ 米（乘 scale）   │ 毫米（uint16）        │
    │ 深度/彩色对齐│ 需 rs.align 或外参│ 同样用外参自己投      │
    │ 供电         │ USB 就够         │ 必须接自带 12V 电源   │
    │ 无头运行     │ 可以             │ **必须有 X + GLX**    │
    └──────────────┴──────────────────┴───────────────────────┘

用法:
    python3 azure_capture.py --info           # 打印内参/外参
    python3 azure_capture.py --save out.ply   # 抓一帧存点云
"""
import argparse
import ctypes
import math
import os
import sys
import time
from ctypes import POINTER, byref, c_int, c_size_t, c_uint32, c_void_p

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "common"))
for _p in (HERE, COMMON):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from depth_to_cloud import depth_to_points, write_ply   # noqa: E402


# ============================================================ 常量
K4A_WAIT_RESULT_SUCCEEDED = 0
K4A_WAIT_RESULT_TIMEOUT = 1
K4A_WAIT_RESULT_FAILED = 2

K4A_DEPTH_MODE_OFF = 0
K4A_DEPTH_MODE_NFOV_2X2BINNED = 1
K4A_DEPTH_MODE_NFOV_UNBINNED = 2
K4A_DEPTH_MODE_WFOV_2X2BINNED = 3
K4A_DEPTH_MODE_WFOV_UNBINNED = 4
K4A_DEPTH_MODE_PASSIVE_IR = 5

K4A_COLOR_RESOLUTION_OFF = 0
K4A_COLOR_RESOLUTION_720P = 1
K4A_COLOR_RESOLUTION_1080P = 2
K4A_COLOR_RESOLUTION_1440P = 3
K4A_COLOR_RESOLUTION_1536P = 4
K4A_COLOR_RESOLUTION_2160P = 5
K4A_COLOR_RESOLUTION_3072P = 6

K4A_FRAMES_PER_SECOND_5 = 0
K4A_FRAMES_PER_SECOND_15 = 1
K4A_FRAMES_PER_SECOND_30 = 2

K4A_IMAGE_FORMAT_COLOR_MJPG = 0
K4A_IMAGE_FORMAT_COLOR_NV12 = 1
K4A_IMAGE_FORMAT_COLOR_YUY2 = 2
K4A_IMAGE_FORMAT_COLOR_BGRA32 = 3
K4A_IMAGE_FORMAT_DEPTH16 = 4
K4A_IMAGE_FORMAT_IR16 = 5
# ★ 枚举顺序必须照 k4atypes.h 抄：
#   MJPG=0, NV12=1, YUY2=2, BGRA32=3, DEPTH16=4, IR16=5, CUSTOM8=6,
#   CUSTOM16=7, CUSTOM=8
#   之前把 CUSTOM 写成 6（那其实是 CUSTOM8），导致 SDK 点云 buffer 只有一半大，
#   报 "cannot reshape array of size 552960 into shape (576,640,3)"。
K4A_IMAGE_FORMAT_CUSTOM8 = 6
K4A_IMAGE_FORMAT_CUSTOM16 = 7
K4A_IMAGE_FORMAT_CUSTOM = 8

K4A_CALIBRATION_TYPE_DEPTH = 0
K4A_CALIBRATION_TYPE_COLOR = 1
K4A_CALIBRATION_TYPE_GYRO = 2
K4A_CALIBRATION_TYPE_ACCEL = 3
K4A_CALIBRATION_TYPE_NUM = 4

# 深度模式的常规分辨率（用来做默认值/校验）
DEPTH_WH = {
    K4A_DEPTH_MODE_NFOV_2X2BINNED: (320, 288),
    K4A_DEPTH_MODE_NFOV_UNBINNED: (640, 576),
    K4A_DEPTH_MODE_WFOV_2X2BINNED: (512, 512),
    K4A_DEPTH_MODE_WFOV_UNBINNED: (1024, 1024),
}
DEPTH_MODE_BY_NAME = {
    "nfov": K4A_DEPTH_MODE_NFOV_UNBINNED,
    "nfov2x2": K4A_DEPTH_MODE_NFOV_2X2BINNED,
    "wfov": K4A_DEPTH_MODE_WFOV_UNBINNED,
    "wfov2x2": K4A_DEPTH_MODE_WFOV_2X2BINNED,
}

# 彩色档位。以前这里没有名字表、构造函数的默认值又是 720P，
# run_azure.py 干脆没传这个参数 —— 于是**根本没法换**，只能改源码。
# 10-13 实测（板子上，MJPG、p50 帧周期，不含第 1 帧预热）：
#
#   档位     fx      3m 处 20cm    MJPG 解码   帧周期    帧率
#   720p     605.7   43px         47ms        336ms    2.98
#   1080p    908.6   65px         91ms        415ms    2.41
#   1440p   1211.4   87px        149ms        428ms    2.33   ← 拐点
#   2160p   1817.2  130px        313ms        586ms    1.71
#
# 解码耗时严格线性于像素数（≈36ms/MP + 14ms 固定），检测/几何与分辨率无关。
# 1440p 相对 720p 目标线性尺寸近翻倍、帧率只掉 22%；再上 4K 要多付一倍帧时间。
# 另测过 BGRA32（绕开解码）：1080p 能出图，但 USB 要搬 8.3MB/帧，
# 取色 109ms、还把几何拖到 355ms → 1.43 FPS，比 MJPG 更差，别走这条路。
#
# ⚠️ 这段是【板子上】加的，本地树一度没有它 —— 两边分叉过一次。
#    合并时优先保留本地（本地有车体系改造），但要把它原样接回来。
COLOR_RES_BY_NAME = {
    "720p": K4A_COLOR_RESOLUTION_720P,
    "1080p": K4A_COLOR_RESOLUTION_1080P,
    "1440p": K4A_COLOR_RESOLUTION_1440P,
    "1536p": K4A_COLOR_RESOLUTION_1536P,
    "2160p": K4A_COLOR_RESOLUTION_2160P,
    "off": K4A_COLOR_RESOLUTION_OFF,
}


# ============================================================ ctypes 结构体
class K4ACalibExtrinsics(ctypes.Structure):
    """48 字节：3x3 旋转（行主序）+ 平移(mm)"""
    _fields_ = [("rotation", ctypes.c_float * 9),
                ("translation", ctypes.c_float * 3)]


class K4ACalibIntrinsicParams(ctypes.Union):
    """60 字节：param 结构 和 v[15] 数组是同一块内存（union）"""
    class _Param(ctypes.Structure):
        _fields_ = [("cx", ctypes.c_float), ("cy", ctypes.c_float),
                    ("fx", ctypes.c_float), ("fy", ctypes.c_float),
                    ("k1", ctypes.c_float), ("k2", ctypes.c_float),
                    ("k3", ctypes.c_float), ("k4", ctypes.c_float),
                    ("k5", ctypes.c_float), ("k6", ctypes.c_float),
                    ("codx", ctypes.c_float), ("cody", ctypes.c_float),
                    ("p2", ctypes.c_float), ("p1", ctypes.c_float),
                    ("metric_radius", ctypes.c_float)]
    _fields_ = [("param", _Param), ("v", ctypes.c_float * 15)]


class K4ACalibIntrinsics(ctypes.Structure):
    """68 字节：type(4) + parameter_count(4) + parameters(60)"""
    _fields_ = [("type", c_int),
                ("parameter_count", c_uint32),
                ("parameters", K4ACalibIntrinsicParams)]


class K4ACalibCamera(ctypes.Structure):
    """128 字节：extrinsics(48) + intrinsics(68) + w(4) + h(4) + radius(4)"""
    _fields_ = [("extrinsics", K4ACalibExtrinsics),
                ("intrinsics", K4ACalibIntrinsics),
                ("resolution_width", c_int),
                ("resolution_height", c_int),
                ("metric_radius", ctypes.c_float)]


class K4ACalibration(ctypes.Structure):
    """1032 字节：depth(128) + color(128) + extrinsics[4][4](768) + 2 个枚举(8)"""
    _fields_ = [("depth_camera_calibration", K4ACalibCamera),
                ("color_camera_calibration", K4ACalibCamera),
                ("extrinsics",
                 (K4ACalibExtrinsics * K4A_CALIBRATION_TYPE_NUM)
                 * K4A_CALIBRATION_TYPE_NUM),
                ("depth_mode", c_int),
                ("color_resolution", c_int)]


class K4ADeviceConfiguration(ctypes.Structure):
    """官方 4.x 的 _k4a_device_configuration_t（v1.4 是 9 个字段）"""
    _fields_ = [("color_format", c_int),
                ("color_resolution", c_int),
                ("depth_mode", c_int),
                ("camera_fps", c_int),
                ("synchronized_images_only", c_int),
                ("depth_delay_off_color_usec", c_int),
                ("wired_sync_mode", c_int),
                ("subordinate_delay_off_master_usec", c_uint32),
                ("disable_streaming_indicator", c_int)]


class K4AFloat3(ctypes.Union):
    """k4a_float3_t：xyz[3] 和 (x,y,z) 是同一块内存（union）"""
    class _XYZ(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float),
                    ("z", ctypes.c_float)]
    _fields_ = [("xyz", ctypes.c_float * 3), ("v", _XYZ)]


class K4AImuSample(ctypes.Structure):
    """k4a_imu_sample_t = 48 字节（字段顺序照 k4atypes.h 抄）"""
    _fields_ = [("temperature", ctypes.c_float),
                ("acc_sample", K4AFloat3),
                ("acc_timestamp_usec", ctypes.c_uint64),
                ("gyro_sample", K4AFloat3),
                ("gyro_timestamp_usec", ctypes.c_uint64)]


def rotation_between(a, b):
    """把单位向量 a 转到 b 的【最小旋转】矩阵（绕 a×b 轴）。

    ★ 为什么特意用"最小旋转"而不是随便找一个旋转：
      最小旋转的转轴 ⟂ a、⟂ b，所以它是【纯俯仰/横滚】修正，
      【不会把航向(yaw)带跑】。相机在桌面上往哪边摆都不影响结果，
      换个位置重新标也不用重调角度。
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a / max(1e-12, float(np.linalg.norm(a)))
    b = b / max(1e-12, float(np.linalg.norm(b)))
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:                              # a ∥ b
        if c > 0:
            return np.eye(3)                  # 已经对齐，不用转
        # 正好反向 180°：任取一个与 a 垂直的轴
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(float(a[0])) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, tmp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _load_k4a():
    for name in ("libk4a.so.1.4", "libk4a.so.1", "libk4a.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("找不到 libk4a —— 装了 Azure Kinect SDK 吗？")


# ============================================================ 取帧类
class AzureSource:
    """封装 Azure Kinect：拿到的永远是 (depth_m, color_bgr, intr, (pts, valid, xyz))

    ★ 和 RealSenseSource 保持完全一样的接口，所以 pc_pipeline 不用改。
    """

    def __init__(self, depth_mode=K4A_DEPTH_MODE_NFOV_UNBINNED,
                 color_resolution=K4A_COLOR_RESOLUTION_720P,
                 fps=K4A_FRAMES_PER_SECOND_15, color=True,
                 rotate=0, device_index=0, verbose=True,
                 use_sdk_points=True, read_gravity=True,
                 body_calib=None):
        self.k = _load_k4a()
        self._decl()
        self.rotate = rotate % 360
        self.verbose = verbose
        # ★ 默认用 SDK 的点云（含畸变校正）。设 False 则用针孔公式（快但会弯）
        self.use_sdk_points = use_sdk_points
        self.h = c_void_p()
        self.started = False
        self.intr_color = None
        self.extr = None
        # ★ 工厂【原始】外参（Rz 共轭【之前】的 depth→color）。_finish_rotations
        #   的闭式公式要用它 —— 别再拿 self.extr，那已经是共轭过的了。
        self.extr_base = None
        # 投影的像素修正量（绕光轴转 vs rot90 绕图心转，差一个固定平移）
        # 相机标定那段里算，只有 rotate≠0 时才非零
        self.px_fix = (0.0, 0.0)
        self.calib = None
        # ★ 重力对齐：R_grav 把点云从句"相机竖直"转到"物理竖直"
        self.R_grav = None
        # ★★ 车体标定（离线卷尺量出来的常量，见 body_calib.py）。
        #   R_body = R_cam_to_body：【raw SDK 深度系 → 车体系】的物理真相，
        #   ★ 那 180° 倒装【已经写在矩阵里】—— 所以它直接就是 R_out，
        #   运行时【不再乘 Rz】（乘了就是 180° 重复计入，det 仍 +1，完全静默）。
        #   t_out = 深度光心在车体系里的位置（m）。
        #   两个都为空/零时完全走老路（R_grav 或纯 Rz），零回归。
        self.R_body = None
        self.t_out = np.zeros(3)
        self.calib_id = None
        self.R_out = np.eye(3)
        # ★ 名字从 _rot_needed 改过来：车体标定带平移，只判断"转没转"
        #   会把纯平移的情况静默漏掉（R_out 凑巧是单位阵时）。
        self._xform_needed = False
        self.up_depth = None
        self.grav_info = {}
        # ★ gravity_ok=True 才表示点云的 Y 真的等于物理竖直方向。
        #   下游（run_azure.py → pc_pipeline --gravity）靠它决定要不要
        #   输出 lean_deg。读取失败时保持 False，一切退回原样。
        self.gravity_ok = False
        self.read_gravity = read_gravity
        # ★★ 车体标定 dict（body_calib.load_body_calib() 的返回值），含
        #   "R"(raw 深度系→车体系, 含 180° 倒装) / "t"(m) / "id"。
        #   给了它 → 不读 IMU，R_out/t_out 直接来自标定。
        self.body_calib = body_calib

        n = self.k.k4a_device_get_installed_count()
        if n <= 0:
            raise RuntimeError(
                "没认到 Azure Kinect。\n"
                "  ① 插好 USB 3.0 线  ② 必须接自带 12V 电源\n"
                "  ③ sudo bash ~/work/azure_kinect/k4a_udev_setup.sh\n"
                "  ④ 无头环境还要有 X：bash ~/work/azure_kinect/run_with_x.sh")

        rc = self.k.k4a_device_open(device_index, byref(self.h))
        if rc != 0:
            raise RuntimeError(f"k4a_device_open 失败 rc={rc}\n"
                               "  · 权限：跑 99-k4a.rules\n"
                               "  · 占用：sudo lsof /dev/bus/usb/002/*\n"
                               "  · 供电：Kinect 需要 12V 独立电源")

        buf = ctypes.create_string_buffer(64)
        sz = c_size_t(64)
        if self.k.k4a_device_get_serialnum(self.h, buf, byref(sz)) == 0:
            self.serial = buf.value.decode()
        else:
            self.serial = "?"

        cfg = K4ADeviceConfiguration()
        cfg.color_format = (K4A_IMAGE_FORMAT_COLOR_MJPG if color
                            else K4A_IMAGE_FORMAT_COLOR_MJPG)
        cfg.color_resolution = (color_resolution if color
                                else K4A_COLOR_RESOLUTION_OFF)
        cfg.depth_mode = depth_mode
        cfg.camera_fps = fps
        cfg.synchronized_images_only = 1 if color else 0
        cfg.depth_delay_off_color_usec = 0
        cfg.wired_sync_mode = 0
        cfg.subordinate_delay_off_master_usec = 0
        cfg.disable_streaming_indicator = 0
        self.cfg = cfg

        # ★ 先拿标定：深度内参 / 彩色内参 / 深度→彩色外参 都从这里来
        cal = K4ACalibration()
        rc = self.k.k4a_device_get_calibration(self.h, depth_mode,
                                               cfg.color_resolution, byref(cal))
        if rc == 0:
            self.calib = cal
            d = cal.depth_camera_calibration
            c = cal.color_camera_calibration
            dp = d.intrinsics.parameters.param
            cp = c.intrinsics.parameters.param
            self.intr_depth = {"fx": float(dp.fx), "fy": float(dp.fy),
                               "cx": float(dp.cx), "cy": float(dp.cy),
                               "width": int(d.resolution_width),
                               "height": int(d.resolution_height),
                               "model": "brown_conrady"}
            self.intr_color = {"fx": float(cp.fx), "fy": float(cp.fy),
                               "cx": float(cp.cx), "cy": float(cp.cy),
                               "width": int(c.resolution_width),
                               "height": int(c.resolution_height)}
            # 深度 -> 彩色 的刚体变换（平移单位是毫米 → 转米）
            e = cal.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][
                K4A_CALIBRATION_TYPE_COLOR]
            R = np.array(e.rotation, np.float64).reshape(3, 3)
            t = np.array(e.translation, np.float64) / 1000.0
            self.extr = (R, t)
            # ★★ 点云在 get() 里会跟着彩色图一起转 rot90(k)，所以投影用的
            #    外参必须【一起做共轭旋转】，否则 2D/3D 一定错位：
            #        点云 P' = Rz·P  →  外参 R' = Rz·R·Rzᵀ,  t' = Rz·t
            #
            #    ★ 只在这里算【一次】。rotate 是构造时就定死的，不是每帧变的。
            #      以前这句写在 get() 里（self.extr = Rz@R_@Rzᵀ, …），
            #      get() 每帧都调 → 等于每帧再乘一次 Rz。
            #      --rotate 180 时 Rz² = I，外参就在"对/错"之间来回跳：
            #      实测日志是 obj=1,0,1,0,… 严格交替，正好一半的帧配不上。
            self.extr_base = self.extr
            k = (self.rotate // 90) if self.rotate else 0
            if k:
                th = math.radians(k * 90.0)
                c_, s_ = math.cos(th), math.sin(th)
                Rz = np.array([[c_, -s_, 0.0],
                               [s_, c_, 0.0],
                               [0.0, 0.0, 1.0]])
                self.extr = (Rz @ R @ Rz.T, Rz @ t)
                # ★★ 还有最后一处对不齐，光修外参不够：
                #    点云绕【光轴】转 ⟺ 像点绕【主点】(cx,cy) 转，
                #    可 np.rot90 是把彩色图绕【图心】转。
                #    两者只差一个固定平移   px_fix = rot_uv(主点) − 主点：
                #        rotate 180 → ( −0.8, +18.0) px    （勉强能忍）
                #        rotate  90 → (+288.6, −270.6) px  ← 直接全废
                #        rotate 270 → (+270.6, −271.4) px  ← 直接全废
                #    rotate=180 时主点和图心只差 9px，所以一直没被发现；
                #    一旦换成 90/270，整个配对会偏近 300 像素。
                #    投影端要把这个平移补回去 —— 见 run_azure.py 传给
                #    match_classes() 的 px_fix 参数。
                #    ★ 下面这套公式必须和 realsense/run_realsense.py 的
                #      rot_uv() 严格一致，改一个就得改另一个。
                ccx, ccy = self.intr_color["cx"], self.intr_color["cy"]
                cw, ch = self.intr_color["width"], self.intr_color["height"]
                if k == 1:
                    pcx, pcy = ch - 1 - ccy, ccx
                elif k == 2:
                    pcx, pcy = cw - 1 - ccx, ch - 1 - ccy
                else:                                    # k == 3
                    pcx, pcy = ccy, cw - 1 - ccx
                self.px_fix = (pcx - ccx, pcy - ccy)
            self.dist_depth = {"k1": float(dp.k1), "k2": float(dp.k2),
                               "k3": float(dp.k3), "k4": float(dp.k4),
                               "k5": float(dp.k5), "k6": float(dp.k6),
                               "p1": float(dp.p1), "p2": float(dp.p2)}
            if verbose:
                print(f"[Kinect] 序列号 {self.serial}")
                print(f"[Kinect] 深度内参 fx={dp.fx:.1f} fy={dp.fy:.1f} "
                      f"cx={dp.cx:.1f} cy={dp.cy:.1f} "
                      f"{d.resolution_width}x{d.resolution_height}")
                print(f"[Kinect] 彩色内参 fx={cp.fx:.1f} fy={cp.fy:.1f} "
                      f"cx={cp.cx:.1f} cy={cp.cy:.1f} "
                      f"{c.resolution_width}x{c.resolution_height}")
                print(f"[Kinect] 深度→彩色 |t| = "
                      f"{np.linalg.norm(t)*1000:.1f} mm")
        else:
            if verbose:
                print(f"[Kinect] ⚠️ 取标定失败 rc={rc}（外参配对会不准）")

        # 启动相机
        for attempt in range(3):
            rc = self.k.k4a_device_start_cameras(self.h, byref(cfg))
            if rc == 0:
                break
            time.sleep(0.5)
        if rc != 0:
            self.close()
            raise RuntimeError(
                f"k4a_device_start_cameras 失败 rc={rc}\n"
                "  ★ 最常见：没有 X 显示 → 深度引擎报 error 204\n"
                "     解决：bash ~/work/azure_kinect/run_with_x.sh\n"
                "  · 其次：供电不足（Kinect 要 12V 独立电源）")
        self.trans = None
        if self.calib is not None:
            self.trans = self.k.k4a_transformation_create(byref(self.calib))
            if verbose and not self.trans:
                print("[Kinect] ⚠️ k4a_transformation_create 失败，"
                      "点云将退回针孔模型（畸变大时会不准）")
        self.started = True
        time.sleep(0.5)
        for _ in range(3):                      # 丢掉前几帧（自动曝光稳定）
            cap = c_void_p()
            self.k.k4a_device_get_capture(self.h, byref(cap), 2000)
            if cap:
                self.k.k4a_capture_release(cap)

        # ★★ 车体标定优先：给了标定就【不读 IMU】—— 相机→车体是固定刚体
        #   变换，标一次永久用；重力对齐反而会跟着车姿态跑偏（车在坡上时
        #   输出会被整体转走，而车体系要求跟着车动）。
        if self.body_calib is not None:
            self._setup_body_frame()          # 见下（抽出来是为了能单测）
        # ★★ 重力对齐（读一次 IMU）—— 见 _apply_gravity() 的推导。保留作
        #    fallback：没标定文件时靠它至少能拿到"物理竖直"，比相机系强。
        #    ★ 只在【开机时读一次】：相机一旦挪位置就得重启，
        #      所以 _apply_gravity 里会把测到的倾角打出来存档。
        elif read_gravity:
            self._apply_gravity(verbose=verbose)
        self._finish_rotations()
        if verbose:
            if self.calib_id:
                print(f"[Kinect] 车体标定 '{self.calib_id}' 已装载："
                      f"光心在车体系 {np.round(self.t_out, 4).tolist()} m")
            print(f"[Kinect] 点云输出 R_out（车体系 X前/Y左/Z上）"
                  f"{'已启用' if self._xform_needed else '= 单位阵'}"
                  f"{'，含平移' if np.abs(self.t_out).max() > 1e-9 else ''}")

    def _setup_body_frame(self):
        """装载车体标定 → self.R_body / self.t_out / self.calib_id。

        ★ 单独抽成方法（不塞在 __init__ 里）是为了能【不开相机】单测 ——
          见 tools/test_body_frame_math.py 的 A5。这里抛的异常就是
          "拒绝启动"：绝不静默退回相机系，否则电控拿到车体系坐标，
          其中混着相机系，会被当成"车前方 0.3m 有东西"直接撞上去。
        """
        R_raw = np.asarray(self.body_calib["R"], np.float64)
        from body_calib import upright_residual
        # ★ 零成本的强校验：把相机的图像翻滚【还原】之后，"图像里的下"
        #   必须指向物理下方。这一条同时挡住"标定是按 post-Rz 系写的"
        #   （180° 重复计入 → 点云整个翻过去，det 仍是 +1，完全静默）
        #   和"矩阵转置写反"。
        #   ⚠️ 判据必须【带 Rz】：不能直接写 R[2,1] < 0 —— 那隐含假设
        #      rotate=0。这台相机 rotate=180，正确的标定 R[2,1] = +1，
        #      用不带 Rz 的判据会【把对的标定拒掉】。
        #   ⚠️ 光轴那列（第三列）两种约定里一样，只有第二列能区分。
        res = upright_residual(R_raw, self.rotate)
        if res >= 0:
            raise RuntimeError(
                "车体标定没通过『图像竖直』判据：\n"
                f"  (R_cam_to_body @ Rz({self.rotate})ᵀ)[2,1] = {res:+.4f} ≥ 0\n"
                "  含义：把图像翻滚还原后，『图像里的下』没有指向车体下方。\n"
                "  · 若第二列是 +1 附近（而 rotate=180）：矩阵是按 post-Rz 系\n"
                "    写的 → 180° 会被重复计入，点云整个翻过去。\n"
                "    应当存【raw SDK 深度系 → 车体系】的物理真相。\n"
                "  · 若第三列也不像光轴：多半是矩阵转置写反了。\n"
                "  自检： python3 azure_kinect/body_calib.py --check <文件>")
        self.R_body = R_raw
        self.t_out = np.asarray(self.body_calib["t"], np.float64).reshape(3)
        self.calib_id = self.body_calib.get("id")
        # gravity_ok 复用成"输出帧已对齐到某个物理竖直系"的信号 ——
        # 下游 run_azure.py 只拿它决定要不要输出 lean_deg，少改一处。
        self.gravity_ok = True

    def _decl(self):
        """声明函数签名 —— 不声明的话 ctypes 会按 int 传参，指针就废了"""
        k = self.k
        k.k4a_device_get_installed_count.restype = c_uint32
        k.k4a_device_open.restype = c_int
        k.k4a_device_open.argtypes = [c_uint32, POINTER(c_void_p)]
        k.k4a_device_get_serialnum.restype = c_int
        k.k4a_device_get_serialnum.argtypes = [c_void_p, ctypes.c_char_p,
                                               POINTER(c_size_t)]
        k.k4a_device_get_calibration.restype = c_int
        k.k4a_device_get_calibration.argtypes = [c_void_p, c_int, c_int,
                                                 POINTER(K4ACalibration)]
        k.k4a_device_start_cameras.restype = c_int
        k.k4a_device_start_cameras.argtypes = [c_void_p,
                                               POINTER(K4ADeviceConfiguration)]
        k.k4a_device_stop_cameras.restype = None
        k.k4a_device_stop_cameras.argtypes = [c_void_p]
        k.k4a_device_get_capture.restype = c_int
        k.k4a_device_get_capture.argtypes = [c_void_p, POINTER(c_void_p),
                                             c_int]
        k.k4a_capture_release.restype = None
        k.k4a_capture_release.argtypes = [c_void_p]
        k.k4a_device_close.restype = None
        k.k4a_device_close.argtypes = [c_void_p]
        for fn in (k.k4a_capture_get_color_image, k.k4a_capture_get_depth_image):
            fn.restype = c_void_p
            fn.argtypes = [c_void_p]
        for fn in (k.k4a_image_get_width_pixels, k.k4a_image_get_height_pixels,
                   k.k4a_image_get_size):
            fn.restype = c_size_t
            fn.argtypes = [c_void_p]
        k.k4a_image_get_buffer.restype = POINTER(ctypes.c_uint8)
        k.k4a_image_get_buffer.argtypes = [c_void_p]
        k.k4a_image_release.restype = None
        k.k4a_image_release.argtypes = [c_void_p]
        # ★ 畸变校正后的点云：必须用 SDK 自己的变换，不能自己拿针孔公式算
        k.k4a_transformation_create.restype = c_void_p
        k.k4a_transformation_create.argtypes = [POINTER(K4ACalibration)]
        k.k4a_transformation_destroy.restype = None
        k.k4a_transformation_destroy.argtypes = [c_void_p]
        k.k4a_image_create.restype = c_int
        k.k4a_image_create.argtypes = [c_int, c_int, c_int, c_int,
                                       POINTER(c_void_p)]
        k.k4a_transformation_depth_image_to_point_cloud.restype = c_int
        k.k4a_transformation_depth_image_to_point_cloud.argtypes = [
            c_void_p, c_void_p, c_int, c_void_p]
        # ★ IMU（重力对齐要用）—— 参数写错就是段错误，务必声明全
        k.k4a_device_start_imu.restype = c_int
        k.k4a_device_start_imu.argtypes = [c_void_p]
        k.k4a_device_get_imu_sample.restype = c_int
        k.k4a_device_get_imu_sample.argtypes = [c_void_p,
                                                POINTER(K4AImuSample), c_int]
        k.k4a_device_stop_imu.restype = None
        k.k4a_device_stop_imu.argtypes = [c_void_p]

    # ---------------------------------------------------------- SDK 点云
    def _sdk_points(self, dimg, w, h):
        """用 SDK 把深度图变成点云（**含畸变校正**）。

        ★ 为什么不能自己拿 fx/cx 反投影：
          Azure Kinect 的深度畸变很大（实测 k1=3.43）。
          在物体常见的 r≈0.4 处，畸变因子
              1 + k1·r² + k2·r⁴ ≈ 1.53
          也就是**53% 的畸变**。用针孔公式算出来的点云是弯的，
          边长会少 10% 左右（实测 25cm 量成 22.3cm）。
          k4a_transformation_depth_image_to_point_cloud 内部会用
          完整的 Brown-Conrady 模型（含切向畸变），这才是对的。

        返回 (h, w, 3) 的 float32 米制坐标；失败返回 None。
        """
        if not self.trans:
            return None
        stride = w * 3 * 2                       # 每像素 3 个 int16
        xyz_img = c_void_p()
        rc = self.k.k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM, w, h,
                                     stride, byref(xyz_img))
        if rc != 0 or not xyz_img:
            return None
        try:
            rc = self.k.k4a_transformation_depth_image_to_point_cloud(
                self.trans, dimg, K4A_CALIBRATION_TYPE_DEPTH, xyz_img)
            if rc != 0:
                return None
            buf = self.k.k4a_image_get_buffer(xyz_img)
            # int16，单位毫米；invalid 是 K4A_INVALID_POINT 即 (0,0,0)
            # ★ as_array 的 shape 要给【字节数】= h*w*3*2，
            #   view(int16) 会把元素个数减半，之后才是 h*w*3 个坐标分量。
            #   之前误写成 h*w*3，结果 reshape 时少一半，报
            #   "cannot reshape array of size 552960"。
            arr = (np.ctypeslib.as_array(buf, shape=(h * w * 3 * 2,))
                   .view(np.int16).reshape(h, w, 3).astype(np.float32))
            return arr / 1000.0                  # mm -> m
        finally:
            self.k.k4a_image_release(xyz_img)

    # ---------------------------------------------------------- 重力对齐
    def _read_gravity(self, secs=1.5):
        """读 IMU 加速度计，返回 (深度系里的"上"单位向量, info)。

        ★ 原理：加速度计静止时读到的不是"重力方向"，而是
          【支撑力】—— 也就是指向上方。所以静止均值的方向 = 上。
          再乘上 accel 系 → depth 系的外参，就得到"上"在深度系里的方向。

        失败 / 读数可疑时返回 (None, info)，调用方退回原来行为。
        """
        info = {}
        if not self.calib:
            info["err"] = "没有标定 → 没法把 IMU 系转到深度系"
            return None, info
        rc = self.k.k4a_device_start_imu(self.h)
        if rc != 0:
            info["err"] = f"k4a_device_start_imu rc={rc}（IMU 被别的进程占着？）"
            return None, info
        accs, gyros = [], []
        try:
            t0 = time.time()
            while time.time() - t0 < secs:
                s = K4AImuSample()
                if self.k.k4a_device_get_imu_sample(self.h, byref(s),
                                                    200) == 0:
                    accs.append((s.acc_sample.v.x, s.acc_sample.v.y,
                                 s.acc_sample.v.z))
                    gyros.append((s.gyro_sample.v.x, s.gyro_sample.v.y,
                                  s.gyro_sample.v.z))
        except Exception as e:                   # noqa: BLE001
            info["err"] = f"读 IMU 出错：{e}"
        finally:
            try:
                self.k.k4a_device_stop_imu(self.h)
            except Exception:                    # noqa: BLE001
                pass
        if len(accs) < 50:
            info.setdefault("err", f"只拿到 {len(accs)} 个 IMU 样本")
            return None, info

        a = np.asarray(accs, np.float64)
        g = np.asarray(gyros, np.float64) if gyros else np.zeros((1, 3))
        m = a.mean(axis=0)
        mag = float(np.linalg.norm(m))
        info.update(n_samples=len(accs), accel_raw=m, accel_norm=mag,
                    accel_jitter=float(np.max(a.std(axis=0))),
                    gyro_norm=float(np.linalg.norm(g.mean(axis=0))))
        # 静止时模长必须是 1g（9.8 左右）；差太多说明相机在动 / IMU 没在校正
        if not (8.5 <= mag <= 11.0):
            info["err"] = (f"加速度模长 {mag:.2f} m/s² 不像重力（静置应≈9.8）"
                           "→ 相机可能在动")
            return None, info

        e = self.calib.extrinsics[K4A_CALIBRATION_TYPE_ACCEL][
            K4A_CALIBRATION_TYPE_DEPTH]
        R = np.array(e.rotation, np.float64).reshape(3, 3)
        up = R @ (m / mag)
        n = float(np.linalg.norm(up))
        if n < 0.5:
            info["err"] = "accel→depth 外参退化"
            return None, info
        info["R_accel_depth"] = R
        return up / n, info

    def _apply_gravity(self, verbose=True):
        """读一次 IMU → 算出把点云转正的 R_grav，并同步修好投影外参。

        ★★ 两处的接法【不一样】，这是整件事最容易写错的地方：

          · 点云： p_out = R_grav · (Rz · p_sdk)
                  → get() 里用 self.R_out = R_grav @ Rz

          · 外参： 彩色图【没有】跟着 R_grav 转（R_grav 是任意角度，
                  彩色图没法跟着转），所以【不能】像 Rz 那样做共轭。
                  投影要的是"转过之后的点 → 原来那张彩色图的像素"：
                      Xc = R'·p' + t'    且   p' = R_gravᵀ·p''
                      ⇒ Xc = (R'·R_gravᵀ)·p'' + t'
                  → 所以是【右乘 R_gravᵀ】，平移 t 不变。
        """
        up, info = self._read_gravity()
        self.grav_info = info
        if up is None:
            if verbose:
                print(f"[Kinect] ⚠️ 重力对齐【跳过】：{info.get('err')}")
                print("[Kinect]    点云仍是相机系 → Y 含相机俯仰的假分量，"
                      "angle_deg 报的是相机的倾角")
            return False

        up_out = self._Rz(self.rotate) @ up      # "上"在 rot90 后的点云系里
        self.up_depth = up
        # 目标：+Y 朝【下】→ 把 up_out 转到 (0,-1,0)
        self.R_grav = rotation_between(up_out, np.array([0.0, -1.0, 0.0]))
        self.gravity_ok = True
        tilt = math.degrees(math.acos(max(-1.0, min(1.0,
                                                    -float(up_out[1])))))
        info["tilt_deg"] = tilt
        if verbose:
            print(f"[Kinect] 重力对齐：IMU {info['n_samples']} 样本，"
                  f"|g|={info['accel_norm']:.2f} m/s²，抖动 "
                  f"{info['accel_jitter']*100:.1f} cm/s²")
            print(f"[Kinect] 相机相对竖直偏 {tilt:.2f}°"
                  f"{'（已把点云转正，Y=真实高度差）' if tilt > 0.15 else '（基本水平）'}")
            if info["gyro_norm"] > 0.15:
                print(f"[Kinect]   ⚠️ 采样时陀螺仪 {info['gyro_norm']:.2f} "
                      "rad/s ≠ 0 —— 相机在动，这次对齐不可信，"
                      "摆稳后重启")
        return True

    def _Rz(self, rotate):
        """绕光轴转 rotate 度（点云/外参共用的那个 Rz）"""
        k = (rotate // 90) if rotate else 0
        if not k:
            return np.eye(3)
        th = math.radians(k * 90.0)
        c_, s_ = math.cos(th), math.sin(th)
        return np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])

    def _finish_rotations(self):
        """定下点云的输出旋转 R_out 与平移 t_out，并一起接进投影外参。

        ★ 用【闭式】公式。它是原来那两步（__init__ 里的 Rz 共轭 + 下面右乘
          R_gravᵀ）的合并 —— 两个极端下与旧代码逐位相同：
            X=I     , t_out=0 → (Rz·R_dc·Rzᵀ,         Rz·t_dc)  = 旧 @339
            X=R_grav, t_out=0 → (Rz·R_dc·Rzᵀ·R_gravᵀ, Rz·t_dc)  = 旧 @637
          与旧式 (R @ Xᵀ, t) 的唯一差别是【平移不再不变】—— 旧式那只在
          t_out = 0 时成立。

        推导（列向量）：投影是 Xc = R'·p' + t'，其中 p' = Rz·p_sdk 是 post-Rz
          系的点。又 p_body = R_out·p_sdk + t_out ⟹ p_sdk = R_outᵀ(p_body − t_out)。
          代入即得
              Xc = (Rz·R_dc·R_outᵀ)·p_body + Rz·(t_dc − R_dc·R_outᵀ·t_out)
          ⚠️ 点云是【左乘】R_out（见 get()），外参是【右乘】R_outᵀ。这个不一致
             正是 _apply_gravity() 那段推导警告的同一个坑，别搞反。

        ★ 为什么 R_out 在标定路径下【不再乘 Rz】：标定文件存的是
          R_cam_to_body ——【raw SDK 深度系 → 车体系】的物理真相，那 180° 倒装
          已经写在矩阵里了。再乘一次 Rz 就是 180° 重复计入（det 仍是 +1，
          完全静默，点云会翻过来）。判据见 body_calib.py 的 R[2,1] < 0。
        """
        Rz = self._Rz(self.rotate)
        if self.R_body is not None:
            # ★ 标定矩阵 = 物理真相，直接就是 R_out，别碰 Rz
            self.R_out = np.asarray(self.R_body, np.float64)
        else:
            # 老路：R_grav（重力对齐）或什么都不做，再绕光轴转 rotate
            X = self.R_grav if self.R_grav is not None else np.eye(3)
            self.R_out = X @ Rz
        # ★ 平移也算"需要变换" —— 否则 R_out 凑巧是单位阵时平移被静默丢掉
        self._xform_needed = bool(np.abs(self.R_out - np.eye(3)).max() > 1e-9
                                  or np.abs(self.t_out).max() > 1e-9)
        if self.extr_base is None:
            return
        R_dc, t_dc = self.extr_base
        self.extr = (Rz @ R_dc @ self.R_out.T,
                     Rz @ (t_dc - R_dc @ self.R_out.T @ self.t_out))

    # ---------------------------------------------------------- 取一帧
    def get(self, stride=1, timeout_ms=2000):
        """返回 (depth_m, color_bgr, intr, (pts, valid, xyz))

        接口和 RealSenseSource.get() 完全一致。
        stride>1 时【先抽样再反投影】（省掉大量无用计算）。
        返回的 intr 始终是【全分辨率】的（投回彩色图要用它）。
        """
        import cv2
        cap = c_void_p()
        rc = self.k.k4a_device_get_capture(self.h, byref(cap), timeout_ms)
        if rc != K4A_WAIT_RESULT_SUCCEEDED or not cap:
            return None
        try:
            dimg = self.k.k4a_capture_get_depth_image(cap)
            cimg = self.k.k4a_capture_get_color_image(cap)
            if not dimg:
                return None
            w = self.k.k4a_image_get_width_pixels(dimg)
            h = self.k.k4a_image_get_height_pixels(dimg)
            buf = self.k.k4a_image_get_buffer(dimg)
            # 深度是 uint16，单位【毫米】—— 和 RealSense 不同，要 /1000
            depth = (np.ctypeslib.as_array(buf, shape=(h * w * 2,))
                     .view(np.uint16).reshape(h, w).astype(np.float32) / 1000.0)
            # ★ 先算 SDK 的畸变校正点云（必须在 dimg 释放前）
            xyz_sdk = self._sdk_points(dimg, w, h) if self.use_sdk_points \
                else None
            intr = {"fx": self.intr_depth["fx"], "fy": self.intr_depth["fy"],
                    "cx": self.intr_depth["cx"], "cy": self.intr_depth["cy"],
                    "width": int(w), "height": int(h),
                    "model": "brown_conrady"}

            color = None
            if cimg:
                cw = self.k.k4a_image_get_width_pixels(cimg)
                ch = self.k.k4a_image_get_height_pixels(cimg)
                csz = self.k.k4a_image_get_size(cimg)
                cbuf = self.k.k4a_image_get_buffer(cimg)
                jpg = np.ctypeslib.as_array(cbuf, shape=(csz,)).tobytes()
                color = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                     cv2.IMREAD_COLOR)
                self.k.k4a_image_release(cimg)
                self._color_wh = (cw, ch)
            self.k.k4a_image_release(dimg)

            k = (self.rotate // 90) if self.rotate else 0
            if k and color is not None:
                color = np.rot90(color, k=-k).copy()
            # ★★ 点云【也一起转】—— 这样输出的 XYZ 才是世界对齐的。
            #
            #   之前只转彩色图、不转点云（RealSense 那版就是这么做的），后果是：
            #   相机倒装时，深度系的 +Y 其实指向【上】（用加速度计测过），
            #   但报告里仍按"+Y 朝下"解读，于是"目标在上方"却显示 Y 为正，
            #   看起来像符号错了。
            #
            #   数学关系（已推导+实测验证）：
            #     把点云绕光轴转 Rz(k·90°)，再用【原来的内参/外参】投影，
            #     得到的就是【旋转后那张图】里的像素坐标。
            #     → 所以转完点云，投影端【不需要】再做 rot_uv。
            if k:
                depth = np.rot90(depth, k=-k).copy()
            # ★ 点云【也一起转】—— 这样输出的 XYZ 才是"物理竖直"对齐的。
            #
            #   之前只转彩色图、不转点云（RealSense 那版就是这么做的），后果是：
            #   相机倒装时，深度系的 +Y 其实指向【上】（用加速度计测过），
            #   但报告里仍按"+Y 朝下"解读，于是"目标在上方"却显示 Y 为正，
            #   看起来像符号错了。
            #
            #   数学关系（已推导+实测验证）：
            #     把点云绕光轴转 Rz(k·90°)，再用【原来的内参/外参】投影，
            #     得到的就是【旋转后那张图】里的像素坐标。
            #     → 所以转完点云，投影端【不需要】再做 rot_uv。
            #
            #   ★ R_out 里除了 Rz 还含车体标定 R_body（或 fallback 的
            #     R_grav），两个都是【构造时算一次】存进 self.R_out 的，
            #     这里只做矩阵乘法。t_out 同理，也只在 _finish_rotations()
            #     里定一次 —— get() 是每帧调的，现算就是每帧变一次。
            #
            #   ★ 深度图 depth 只跟 rot90（Rz），【不】跟 R_body/R_grav：
            #     深度图是给点云/彩色配准用的原始栅格，
            #     换系只是换个坐标系去【表达】点云。
            #
            #   ★★ 平移必须一起加！只把 t_out 塞进 extr 而点云不加，
            #      点云参考点与投影参考点就差 |t|≈0.6m → 投影偏数百像素
            #      → obj 直接变 0。下面两个分支【逐字一致】，别只改一个。
            if self._xform_needed and xyz_sdk is not None:
                xyz_sdk = xyz_sdk @ self.R_out.T + self.t_out
            # ★ 外参也跟着转过了（R' = Rz·R·Rzᵀ 再做 R'·R_gravᵀ）——
            #    推导见 __init__ / _apply_gravity()。少转一步，Azure 的
            #    t=32.3mm 会造成 2D/3D 差 ~30~60 像素 → 投影落到框外 → obj=0。
            #
            #    ★ 那些旋转在 __init__ 里已经算好并存进 self.extr 了。
            #      【绝对不要在这里再改 self.extr】—— get() 是每帧调的，
            #      在这里改就等于每帧再乘一次，外参会来回跳。
            self.rotate_k = 0        # 点云已转正 → 投影不用再 rot_uv

            # 抽样后再反投影（和 RealSense 版一致）
            if stride and stride > 1:
                depth_s = depth[::stride, ::stride].copy()
                intr_pts = dict(intr)
                intr_pts["fx"] /= stride
                intr_pts["fy"] /= stride
                intr_pts["cx"] /= stride
                intr_pts["cy"] /= stride
                intr_pts["width"] = depth_s.shape[1]
                intr_pts["height"] = depth_s.shape[0]
            else:
                depth_s, intr_pts = depth, intr
            if xyz_sdk is not None:
                # ★ 用 SDK 的点云（已做畸变校正），只做抽样和有效性筛选
                xyz_full = xyz_sdk[::stride, ::stride] if stride > 1 else xyz_sdk
                flat = xyz_full.reshape(-1, 3)
                # SDK 用 (0,0,0) 表示无效点；再按距离范围筛一次
                valid = (np.abs(flat).sum(axis=1) > 1e-6)
                return depth, color, intr, (flat, valid, xyz_full)
            pts, valid, xyz = depth_to_points(depth_s, intr_pts)
            if self._xform_needed:
                # 针孔兜底路径也要转正，否则和 SDK 路径的输出坐标系
                # 不一致（一个世界对齐、一个相机系），下游会时对时错。
                # ★ 与上面 SDK 分支【逐字一致】（含 + t_out）。
                pts = np.asarray(pts, np.float64) @ self.R_out.T + self.t_out
                xyz = np.asarray(xyz, np.float64) @ self.R_out.T + self.t_out
            return depth, color, intr, (pts, valid, xyz)
        finally:
            self.k.k4a_capture_release(cap)

    def to_camera_frame(self):
        """返回一个 `p_out -> p_cam` 的函数：输出帧（车体系）→ 相机光学系。

        ★ 为什么需要它：位置字段（face/cube/normal）转车体系发出去，
          但 `range`/`az`/`el` 是【相对相机光轴】的量 —— 那是用户明确
          拍板保留的语义（球坐标三元组必须自洽，而且 quality_of 的阈值
          是按相机距离标定的）。这几个得在相机系里算。

        p_body = R_out·p_sdk + t_out  ⟹  p_sdk = R_outᵀ(p_body − t_out)
        再绕光轴转 rotate 度，得到【post-Rz 系】= "把相机当正装"的光学系：
            p_cam = Rz · R_outᵀ · (p_body − t_out)

        ⚠️ 用的是 post-Rz 系（③）而不是 raw 系（①）。差别只在 rotate≠0 时：
           ③ = 屏幕上那张【转正后】的图的方向，正好是操作员看到的样子；
           ① 是传感器原始方向，倒装时会和屏幕左右/上下全反。
           本车倒装（rotate=180），两者符号全反 —— 这一点写进规范文档，
           电控按文档的约定解读。
        """
        Rz = self._Rz(self.rotate)
        R_out_T = np.asarray(self.R_out, np.float64).T
        t = np.asarray(self.t_out, np.float64)

        def f(p3):
            if p3 is None:
                return None
            return Rz @ (R_out_T @ (np.asarray(p3, np.float64).reshape(3) - t))

        f.R_cam = Rz @ R_out_T          # 方向用它转（不平移）
        f.t_cam = -Rz @ (R_out_T @ t)
        return f

    def close(self):
        try:
            if self.started:
                try:                             # IMU 没停干净会挡住下次 open
                    self.k.k4a_device_stop_imu(self.h)
                except Exception:                # noqa: BLE001
                    pass
                self.k.k4a_device_stop_cameras(self.h)
                self.started = False
            if getattr(self, "trans", None):
                self.k.k4a_transformation_destroy(self.trans)
                self.trans = None
            if self.h:
                self.k.k4a_device_close(self.h)
                self.h = c_void_p()
        except Exception:
            pass


# ============================================================ 自测
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", action="store_true", help="打印内参/外参")
    ap.add_argument("--save", default=None, help="抓一帧点云存到 .ply")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--depth-mode", default="nfov",
                    choices=list(DEPTH_MODE_BY_NAME))
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--save-color", default=None, help="顺便存彩色图")
    a = ap.parse_args()

    src = AzureSource(depth_mode=DEPTH_MODE_BY_NAME[a.depth_mode],
                      color=not a.no_color)
    try:
        best = None
        for i in range(max(1, a.frames)):
            r = src.get(stride=a.stride)
            if r is None:
                continue
            depth, color, intr, (pts, valid, xyz) = r
            n = int(valid.sum())
            print(f"  第{i}帧: {n} 个有效点  "
                  f"深度范围 {depth.reshape(-1)[valid].min():.3f}~{depth.reshape(-1)[valid].max():.3f} m")
            if best is None or n > best[0]:
                best = (n, pts[valid].copy(), depth, color, intr)
        if best is None:
            print("❌ 一帧都没拿到")
            return 1
        n, p, depth, color, intr = best
        print(f"\n最佳帧: {n} 点")
        if a.save:
            write_ply(a.save, p)
            print(f"✅ 点云已存 {a.save}")
        if a.save_color and color is not None:
            import cv2
            cv2.imwrite(a.save_color, color)
            print(f"✅ 彩色图已存 {a.save_color}")
        if a.info:
            print(f"\n深度内参: {intr}")
            if src.extr is not None:
                R, t = src.extr
                print(f"外参 |t| = {np.linalg.norm(t)*1000:.1f} mm")
                print(f"R = \n{np.round(R, 4)}")
        # 畸变系数也打出来 —— 想上 undistort 时有用
        if src.calib is not None:
            d = src.dist_depth
            print(f"\n深度畸变: k1={d['k1']:.5f} k2={d['k2']:.5f} "
                  f"k3={d['k3']:.5f} p1={d['p1']:.5f} p2={d['p2']:.5f}")
    finally:
        src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
