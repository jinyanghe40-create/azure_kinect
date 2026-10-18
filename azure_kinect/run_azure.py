#!/usr/bin/env python3
"""
============================================================
 Azure Kinect DK —— 全流程入口（对标 realsense/run_realsense.py）
============================================================
 干什么：
    彩色图 → SSDlite 检测器（TensorRT）→ 类别 + 2D 框
    深度图 → 点云 → 去桌面 → 聚类 → 拟合面 → 面心/法向/夹角
    两者配对 → 输出【类别 + 位置 + 朝向 + 尺寸 + 可信度】

 跑法（板子上）:
    # ★ 必须先有 X 显示（Ki 深度引擎要 GLX）
    sudo bash ~/work/azure_kinect/run_with_x.sh \
        python3 ~/work/azure_kinect/run_azure.py --print --frames 30 --stride 2

    --serial-dry     把要发给电控的 JSON 打到屏幕（不接硬件）
    --ros HOST:PORT  发给电脑上的 RViz 桥接

 ★ 关键设计：**相机无关的代码不重复写**
    match_classes / serial_obj / quality_of / project_to_image / rot_uv
    这些和用哪个相机没关系，直接复用 run_realsense 里的 ——
    否则两边会慢慢跑偏（改了一边忘了另一边）。

 这个文件只负责【Kinect 取帧】这一层适配。
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "common"))
REALSENSE = os.path.abspath(os.path.join(HERE, "..", "realsense"))
# ★★ 顺序必须写死：后 insert 的排在越前面，这里要【本目录优先】。
#   别改成 `for p in (...): if p not in sys.path: insert(0, p)` ——
#   `python3 run_azure.py` 时 Python 已经把脚本所在目录放进 sys.path[0]，
#   那个 if 会把 HERE 这一轮跳过，而 COMMON 照插不误，于是 common/ 里
#   同名的 pc_pipeline.py 反把本目录的盖掉。
#   踩过：改 azure_kinect/pc_pipeline.py 加了个功能，跑起来毫无变化，
#   也不报错，查了半天才发现 import 到的是 common/ 那份。
#   （和下面 RR 那段说的"common/ 里一份误传的旧 run_realsense.py"同类。）
for p in (COMMON, REALSENSE, HERE):
    while p in sys.path:
        sys.path.remove(p)
for p in (COMMON, REALSENSE, HERE):
    sys.path.insert(0, p)

import pc_pipeline                                       # noqa: E402
from azure_capture import (AzureSource, DEPTH_MODE_BY_NAME,  # noqa: E402
                           COLOR_RES_BY_NAME)

# ★★ 复用 RealSense 版里相机无关的那部分（配对/串口字段/质量分级/投影）。
#   这里【必须按文件路径显式加载】，不能写 `import run_realsense` ——
#   因为 common/ 也在 sys.path 里，一旦哪里多出一份同名的旧文件，
#   Python 会抓到旧的，而且**不报错**，只是字段全变 None。
#   实测踩过：common/ 里有一份误传的旧 run_realsense.py，
#   结果 face_center/faces/edge_src 全是 None，精度报告"好得离谱"。
import importlib.util                                    # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "_rr_shared", os.path.join(REALSENSE, "run_realsense.py"))
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"找不到 {REALSENSE}/run_realsense.py")
RR = importlib.util.module_from_spec(_spec)
sys.modules["_rr_shared"] = RR
_spec.loader.exec_module(RR)


def parse_args():
    p = argparse.ArgumentParser(description="Azure Kinect 全流程")
    p.add_argument("--engine", default=RR.CFG["engine"])
    p.add_argument("--score", type=float, default=RR.CFG["score_thresh"])
    p.add_argument("--edge", type=float, default=0.25,
                   help="方块边长(m)，只是兜底 —— 默认从拟合面自己量")
    p.add_argument("--no-edge-auto", dest="edge_auto", action="store_false",
                   default=True)
    p.add_argument("--min-dist", type=float, default=0.3)
    p.add_argument("--max-dist", type=float, default=1.5)
    p.add_argument("--plane-thresh", type=float, default=0.003)
    p.add_argument("--table-thresh", type=float, default=0.010)
    p.add_argument("--voxel", type=float, default=0.02)
    p.add_argument("--min-cluster-points", type=int, default=0)
    p.add_argument("--face-select", choices=["most_facing", "largest"],
                   default="most_facing")
    p.add_argument("--depth-noise-coef", type=float, default=0.008)
    p.add_argument("--max-faces", type=int, default=3)
    p.add_argument("--stride", type=int, default=2,
                   help="深度抽样步长。★ 越小越准（面上点更多），但更慢。"
                        "实测 stride5 在 0.8m 处抖 14mm，stride2 只抖 2mm")
    p.add_argument("--depth-mode", default="nfov",
                   choices=list(DEPTH_MODE_BY_NAME))
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    # ★ 彩色分辨率。名单和实测表在 azure_capture.py 的 COLOR_RES_BY_NAME 上面。
    #   默认 720p 不变（帧率优先）。远处小目标要像素时才往上调。
    p.add_argument("--color-res", default="720p",
                   choices=list(COLOR_RES_BY_NAME),
                   help="彩色分辨率。★ 远处小目标要更多像素，但 MJPG 解码费"
                        "严格按像素线性涨（实测每 MP ≈36ms）：3m 处的 20cm "
                        "方块在 720p(fx=606) 只有 40px，1440p(fx=1211) 到 81px "
                        "而帧率只从 2.98 掉到 2.33；2160p(fx=1817) 121px 但只剩 "
                        "1.71 FPS。默认 720p 不变。")
    # ★★ 车体标定：相机→车体 的固定刚体变换（见 body_calib.py）。
    #   给了它 → 点云/输出全在【车体系】，且【不读 IMU】。
    #   默认路径固定，缺文件就拒绝启动（--allow-uncalibrated 才放行）。
    p.add_argument("--calib", default=os.environ.get("K4A_BODY_CALIB"),
                   metavar="PATH",
                   help="车体标定文件（默认 azure_kinect/body_calib.json）。"
                        "★ 没有它输出的就不是车体系坐标，电控拿到的数不能用。")
    p.add_argument("--allow-uncalibrated", dest="allow_uncalibrated",
                   action="store_true",
                   help="没有标定文件也硬跑 —— 会退回相机系/重力对齐，"
                        "输出【不是】车体坐标。只在调试/标定前用，别接电控。")
    p.add_argument("--no-gravity", dest="gravity", action="store_false",
                   default=True,
                   help="关掉重力对齐。默认开机读一次 IMU，把点云转到"
                        "物理竖直系 —— Y 才是真实高度差，angle_deg 才是"
                        "物体真实倾角。IMU 读失败会自动退回相机系。")
    p.add_argument("--only-classified", action="store_true")
    p.add_argument("--print", dest="show_text", action="store_true")
    p.add_argument("--json", dest="json_out", action="store_true")
    p.add_argument("--ros", default=None, metavar="HOST:PORT")
    p.add_argument("--ros-hz", type=float, default=10.0)
    p.add_argument("--ros-max-points", type=int, default=30000)
    p.add_argument("--no-ros-image", dest="ros_image", action="store_false",
                   default=True)
    p.add_argument("--serial", default=None)
    p.add_argument("--serial-dry", dest="serial_dry", action="store_true")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--no-split", dest="split_boxes", action="store_false",
                   default=True,
                   help="关掉【按 2D 检测框切簇】。默认开 —— 两个方块贴"
                        "在一起时体素聚类只会给出一团，切一刀才能一块一个。"
                        "没有检测框/投影失败时自动退回老行为。")
    p.add_argument("--mm", dest="mm", action="store_true",
                   help="输出单位换成【毫米】。默认米。"
                        "★ 覆盖终端 / 串口 / ROS 三类对象输出；"
                        "★ ROS 里的【点云仍然是米】（RViz 的几何按米画），"
                        "只有对象字典变 mm，头里会带 obj_unit 说明。"
                        "只动长度字段 —— normal(单位向量)/az/el(角度)/"
                        "rms_mm(本来就是mm) 都不碰。")
    p.add_argument("--no-dedup", dest="dedup", action="store_false",
                   default=True,
                   help="关掉【吞并抑制】。默认开 —— 检测器有时会给出一个"
                        "把真框整个包住的大虚检框，普通 NMS 按 IoU 压不掉它")
    p.add_argument("--geo-verbose", dest="geo_verbose",
                   action="store_true",
                   help="打出几何管线的每步日志（去桌面 / 体素聚类 /"
                        "按框切簇 / 每个簇的面）。★ 排查【为什么只出 2 个】"
                        "时必须开 —— 默认关是因为每个簇要打三行，"
                        "10Hz 长期跑会刷屏。")
    return p.parse_args()



def _make_uv_fn(src):
    """点云 (N,3) → 彩色图像素 (N,2)。

    ★ 这里的变换链必须和 RR.match_classes 里的一模一样（project → rot_uv
      → +px_fix），否则【切簇用的框】和【最后贴标签用的框】不在同一套
      坐标里，切出来的块会跟着错位。

    ★ 点云已经被 AzureSource 转过（rotate + 重力对齐），self.extr 也跟着
      转过了，所以除了 rot_uv / px_fix 这两个"像素端的修正"，不要再自己
      补任何旋转 —— match_classes 里怎么写的，这里就怎么抄。

    只建一次（内参/外参不随帧变）；检测框每帧换，由 run_pipeline 的
    args.split_boxes 传进去。
    """
    intr = src.intr_color
    extr = getattr(src, "extr", None)
    rot_k = getattr(src, "rotate_k", 0)
    pfx, pfy = getattr(src, "px_fix", (0.0, 0.0))
    W, H = intr["width"], intr["height"]

    def fn(P):
        P = np.asarray(P, np.float64)
        Q = P
        if extr is not None:
            R, t = extr
            Q = P @ np.asarray(R, np.float64).T + np.asarray(t, np.float64)
        Z = Q[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = intr["fx"] * Q[:, 0] / Z + intr["cx"]
            v = intr["fy"] * Q[:, 1] / Z + intr["cy"]
        u, v = RR.rot_uv(u, v, rot_k, W, H)
        bad = (Z <= 1e-6) | ~np.isfinite(u) | ~np.isfinite(v)
        # 无效点推到图像外，自然落不进任何框（别让它们污染 argmin）
        u = np.asarray(np.where(bad, -1e9, u), np.float64) + pfx
        v = np.asarray(np.where(bad, -1e9, v), np.float64) + pfy
        return np.stack([u, v], axis=1)

    return fn


# =====================================================================
# ★★ --mm：把输出单位从米换成毫米。
#    白名单制 —— 输出字典里的字段【不都是长度】，一律乘 1000 会毁数据：
#        normal      单位向量（乘了就废）
#        az / el     角度
#        angle       角度（度）
#        lean_deg    角度（度）
#        rms_mm      本来就是毫米
#        uv          像素
#        score / n_* / pos_type / edge_src / cls_* / faces 的个数
#    ⚠ 新加字段想跟着 --mm 走，必须加进下面某张表；宁可漏一个，
#      也不能让 normal 被乘 1000。
_MM_VEC = ("face", "cube", "xyz", "face_center", "cube_center", "nearest_pt")
_MM_SCAL = ("range", "nearest", "edge")
_MM_NEST = ("faces",)          # [{"normal":..,"center":[x,y,z]}, ...]


def _to_mm(d):
    """把一份输出字典里的长度字段（米）换成毫米。只认白名单里的键。"""
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        elif k in _MM_VEC and not isinstance(v, (str, bytes)):
            out[k] = [round(float(x) * 1000.0, 1) for x in v]
        elif k in _MM_SCAL:
            out[k] = round(float(v) * 1000.0, 1)
        elif k in _MM_NEST and isinstance(v, list):
            lst = []
            for f in v:
                if isinstance(f, dict) and f.get("center") is not None:
                    f = dict(f)
                    f["center"] = [round(float(x) * 1000.0, 1)
                                   for x in f["center"]]
                lst.append(f)
            out[k] = lst
        else:
            out[k] = v              # ← 不在名单里的原样透传
    return out


def _dedup_contained(dets, cover=0.80):
    """丢掉"吞掉了一个更可信的框"的大框（吞并抑制）。

    ★ 为什么需要：这个场景里检测器在 0.20~0.40 所有阈值下都稳定给出
      5 个框 —— 4 个方块 + 1 个虚检。那个虚检的"药品"框横跨两列，
      把真的"药品"框整个包在里面。
      ★ 普通 NMS 压不掉它：NMS 按 IoU 判，IoU = 小框面积/大框面积
        只有 ~0.4，低于 nms_iou=0.55；可交集占了【小框】的 100%。
        按"覆盖了小框多少"判就干净了。
      ★ 高分框先入选；候选框只要有 ≥cover 的面积被某个已入选的框盖住，
        就丢掉。真框 0.73 分先入选，虚检框 0.56 分被它盖住 → 丢虚检。

    副作用要想清楚：两个方块一前一后、前面那个几乎挡住后面那个时，
    后面的框也会被压掉 —— 但那时它本来也只剩一小条可见，压掉是对的。

    实测：5 框 → 4 框，物体数 6 → 4。
    """
    kept = []
    for d in sorted(dets, key=lambda t: -t[5]):
        x0, y0, x1, y1 = d[0], d[1], d[2], d[3]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        swallowed = False
        for k in kept:
            ix0, iy0 = max(x0, k[0]), max(y0, k[1])
            ix1, iy1 = min(x1, k[2]), min(y1, k[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if inter <= 0.0:
                continue
            karea = max(0.0, k[2] - k[0]) * max(0.0, k[3] - k[1])
            if inter / max(1e-6, min(area, karea)) >= cover:
                swallowed = True
                break
        if not swallowed:
            kept.append(d)
    return kept


def main():
    a = parse_args()


    if not os.path.exists(a.engine):
        print(f"[ERROR] 找不到 engine: {a.engine}")
        return 1
    if not os.environ.get("DISPLAY"):
        print("[警告] 没有 DISPLAY —— Azure Kinect 深度引擎需要 X/GLX，"
              "会报 error 204")
        print("       用: sudo bash ~/work/azure_kinect/run_with_x.sh "
              "python3 run_azure.py ...")

    # ★★ 车体标定：读文件、全套校验、失败就【拒绝启动】。
    #   rotate 一起传进去 —— load_body_calib 会断言 --rotate 和文件里的
    #   image_rotate_deg 一致。这是零成本的强校验，顺手挡住
    #   "服务单元漏了 --rotate 180"（曾经的真实 bug：图倒着看得见，
    #   点云错得静默）。
    import body_calib as BC
    calib = BC.load_or_exit(a.calib, rotate=a.rotate,
                            allow_missing=a.allow_uncalibrated)

    if a.mm:
        print("[单位] --mm：对象输出用【毫米】。"
              "（ROS 点云仍是米 —— 只有 objects 字典变 mm，头里带 obj_unit）")
    src = AzureSource(depth_mode=DEPTH_MODE_BY_NAME[a.depth_mode],
                      color_resolution=COLOR_RES_BY_NAME[a.color_res],
                      color=not a.no_color, rotate=a.rotate,
                      read_gravity=a.gravity, body_calib=calib)

    print("[TRT] 加载 engine…")
    eng = RR.Engine(a.engine)
    S = RR.CFG["img_size"]
    ANCH = RR.generate_anchors(S)
    gargs = RR.geometry_args(
        edge=a.edge, min_dist=a.min_dist, max_dist=a.max_dist,
        plane_thresh=a.plane_thresh, table_thresh=a.table_thresh,
        voxel=a.voxel, min_cluster_points=a.min_cluster_points,
        face_select=a.face_select, max_faces=a.max_faces,
        depth_noise_coef=a.depth_noise_coef, edge_auto=a.edge_auto,
        # ★★ 告诉 pc_pipeline "点云在哪个坐标系里"。三个都从 src 的真实
        #   状态取，不猜：
        #     up_axis    —— 只有点云真的对齐到某个物理竖直系才有"上"。
        #                   标定过 → 车体系 Z；老重力对齐 → 相机系 Y。
        #                   都没有 → None（不算 lean_deg，与老行为一致）。
        #     cam_axis   —— 光轴在【输出帧】里的方向 = R_out 的第三列。
        #                   不是 (0,0,1)！车体系下光轴指的是车头方向。
        #     cam_origin —— 光心在【输出帧】里的位置 = t_out。
        #                   ★ 不给它 → orient_toward_camera 会把法向翻反，
        #                     体心沿错方向退 16.7cm，而 cube_ok 仍是 true。
        up_axis=([0.0, 0.0, 1.0] if calib is not None
                 else ([0.0, 1.0, 0.0] if getattr(src, "gravity_ok", False)
                       else None)),
        cam_axis=src.R_out[:, 2].tolist(),
        cam_origin=src.t_out.tolist(),
        # 老开关仍然传（给其它读 args.gravity 的地方用），但 pc_pipeline
        # 现在优先看 up_axis —— 传了 up_axis 时 gravity 会被忽略。
        gravity=bool(getattr(src, "gravity_ok", False)))
    # ★ 按 2D 检测框切簇（默认开）。投影函数只建一次；框每帧换，在主循环里
    #   设 gargs.split_boxes。关掉时这两个属性为 None，run_pipeline 直接跳过。
    gargs.split_uv_fn = _make_uv_fn(src) if a.split_boxes else None
    gargs.split_boxes = None

    ros = None
    if a.ros:
        from ros_sender import RosSender
        rhost, _, rport = a.ros.partition(":")
        ros = RosSender(rhost, int(rport or 9100),
                        max_points=a.ros_max_points, hz=a.ros_hz)
    ser = None
    if a.serial or a.serial_dry:
        from serial_out import SerialOut
        ser = SerialOut(a.serial if not a.serial_dry else None, a.baud, a.hz)

    # ★★ 车体系 → 相机系 的换算函数，给 range/az/el/angle 用。
    #   位置字段（face/cube/normal）按【车体系】直接发 —— 那正是电控要的。
    #   但 (range, az, el) 是球坐标三元组，必须相对【相机光心/光轴】才有意义，
    #   而且 quality_of 的 (nearest ≤ 1.05 / 1.30) 是按相机距离实测标定的
    #   —— 换基准会让 quality 整体掉成 COARSE【而且不报错】。
    #   标定没装载时（--allow-uncalibrated）点云本来就在相机系，传 None。
    #   直接建出来就行 —— 没标定/没对齐时 R_out == Rz、t_out == 0，
    #   它会精确退化成【单位映射】（az/el 与改前逐位相同），不是特殊分支。
    to_cam = src.to_camera_frame()

    # ★★ 顶层头。`coord` + `rev` 必须有 —— 否则电控分不清新旧协议：
    #   同一个 "face":[0.31,0.02,0.62] 在相机系和车体系下是两个完全不同的点。
    #   `cam`/`cam_unit` 给电控相机在车体哪儿，它自己也能算相机距离。
    #   ⚠️ 单位约定：obj_unit 只管 obj[]（--mm 时是毫米）；
    #      顶层几何量（点云、cam）恒为【米】，与 ROS 那路一致。
    if calib is not None:
        top_extra = {"coord": "body", "rev": 2, "cam": np.round(src.t_out, 4).tolist(),
                     "cam_unit": "m", "calib": calib["id"]}
    else:
        top_extra = {"coord": "camera", "rev": 2, "cam": [0.0, 0.0, 0.0],
                     "cam_unit": "m", "calib": None}
    top_extra["obj_unit"] = "mm" if a.mm else "m"

    n = 0
    t_start = time.time()
    t_loop = time.time()
    t_inf = t_geo = 0.0
    try:
        while a.frames == 0 or n < a.frames:
            r = src.get(stride=a.stride)
            if r is None:
                continue
            depth, color, intr, (pts, valid, xyz) = r
            n += 1
            if n == 1:
                # ★ 从第一帧开始计时：总时间里含 TensorRT 引擎加载（几十秒），
                #   拿它算 FPS 会得出 0.6 这种假数字（实测稳态是 2.2）。
                t_loop = time.time()
            if color is None:
                print("[ERROR] 没拿到彩色图（--no-color？）")
                break
            W, H = color.shape[1], color.shape[0]

            # ---- ① 2D 检测 ----
            t0 = time.time()
            out = eng.infer(RR.preprocess(color, S))
            dets = RR.postprocess(eng.get_output(out, "cls_logits")[0],
                                  eng.get_output(out, "bbox_reg")[0],
                                  ANCH, a.score, RR.CFG["nms_iou"],
                                  RR.CFG["max_det"])
            sx, sy = W / float(S), H / float(S)
            dets = [(x0 * sx, y0 * sy, x1 * sx, y1 * sy, c, sc)
                    for (x0, y0, x1, y1, c, sc) in dets]
            # ★ 吞并抑制：普通 NMS 只按 IoU 判，压不掉"包住真框的大虚检框"
            if a.dedup:
                n0 = len(dets)
                dets = _dedup_contained(dets)
                if a.geo_verbose and len(dets) != n0:
                    print(f"[dets] 吞并抑制：{n0} 框 → {len(dets)} 框")
            t_inf = (time.time() - t0) * 1000

            # ---- ② 3D 几何 ----
            t0 = time.time()
            # ★ 把这一帧的检测框交给几何：贴在一起的方块靠它才能分开
            gargs.split_boxes = dets if a.split_boxes else None
            geo = pc_pipeline.run_pipeline(pts[valid], gargs,
                                           verbose=a.geo_verbose)
            t_geo = (time.time() - t0) * 1000

            # ---- ③ 配对 ----
            objs = RR.match_classes(dets, geo["objects"], src.intr_color,
                                    src.extr, getattr(src, "rotate_k", 0),
                                    getattr(src, "px_fix", (0.0, 0.0)))
            if a.only_classified:
                objs = [o for o in objs if o.get("cls")]

            # ---- ④ 输出 ----
            if a.json_out:
                rec = {"frame": n, "t": round(time.time() - t_start, 3),
                       "infer_ms": round(t_inf, 1), "geo_ms": round(t_geo, 1),
                       "w": W, "h": H, "n": len(objs),
                       "obj": [{"cls": o["cls"], "cn": o["cn"],
                                "score": o["score"], "xyz": o["xyz"],
                                "angle": o["angle"],
                                "lean": o.get("lean_deg"),
                                "pos_type": o["pos_type"]} for o in objs]}
                print(RR.json.dumps(rec, ensure_ascii=False), flush=True)
            elif a.show_text:
                # ★ --mm 时按毫米打（0.254 m -> 254 mm），小数位也跟着变：
                #   毫米下再留三位小数就是在显示微米级噪声。
                _k = 1000.0 if a.mm else 1.0
                _u = "mm" if a.mm else "m"
                _f = "{:.0f}" if a.mm else "{:.3f}"

                def _one(o):
                    # ★ 打【面心】不是体心。体心 = 面心 − (edge/2)·Σn，是拿
                    #   实测边长反推的；边长一错体心就跟着错（实测"仪器"那块
                    #   面只量到 20.1cm，体心 Z 就差了 26mm）。面心是平面拟合
                    #   的直接结果，帧间只抖 2mm，标定/给电控都用它。
                    #   ⚠ 换过来之后 Z 会比原来小 ~125mm（= 边长的一半），
                    #     不是算错了，是量的东西变了。
                    _p = o.get("face_center") or o["xyz"]
                    _xyz = ",".join(_f.format(_p[i] * _k) for i in range(3))
                    s = (f"{(o['cn'] or '?')}{o['cls'] or ''} "
                         f"{o['score'] or 0:.2f} "
                         f"面心=({_xyz}){_u} 夹角={o['angle']:.1f}°")
                    # ★ 重力对齐生效时才有；未对齐时 lean 是 None，
                    #   直接不显示（不能拿 None 当 0°）。
                    if o.get("lean_deg") is not None:
                        s += f" 倾角={o['lean_deg']:.1f}°"
                    return s
                body = " | ".join(_one(o) for o in objs) or "-"
                print(f"[{n:05d}] 检测{t_inf:5.1f}ms 几何{t_geo:6.1f}ms "
                      f"obj={len(objs)}  {body}", flush=True)

            if ser is not None:
                ser_objs = [RR.serial_obj(o, to_cam=to_cam) for o in objs]
                if a.mm:
                    ser_objs = [_to_mm(d) for d in ser_objs]
                # ★ 把单位写进头里。--mm 的 help 一直说"头里会带 obj_unit 说明"，
                #   但之前 extra 从来没传，电控【无法判断】收到的是米还是毫米 ——
                #   同一个 123.4 在两种单位下差 1000 倍，这是会撞机械臂的。
                ser.send(n, ser_objs, t_inf, extra=top_extra)

            if ros is not None:
                ros_objs = []
                for o in objs:
                    d = {
                        "cls": o["cls"], "cn": o["cn"], "score": o["score"],
                        "xyz": o["xyz"], "angle": o["angle"],
                        "lean_deg": o.get("lean_deg"),
                        "normal": o.get("normal"),
                        "face_center": o.get("face_center"),
                        "cube_center": o.get("cube_center"),
                        "pos_type": o.get("pos_type"),
                        "faces": o.get("faces"),
                        "nearest": o.get("nearest_m"),
                        "nearest_pt": o.get("nearest_pt"),
                        "n_points": o.get("n_points"),
                        "rms_mm": o.get("rms_mm"),
                        "edge": o.get("edge_used") or a.edge,
                        "edge_src": o.get("edge_src"),
                    }
                    if a.mm:
                        d = _to_mm(d)
                    ros_objs.append(d)
                # ★ 点云 pts 一直是米（RViz 按米画几何），只有 objects
                #   字典跟着 --mm 走 —— 头里带 obj_unit 让电脑侧知道。
                # ★ frame_id：点云已经在【车体系】里了，就不再是
                #   camera_depth_optical_frame —— 报错的话 RViz 里 TF 会
                #   把它摆到相机光心去，看着"点云飘了"。base_link 是车体。
                ros.send(pts[valid], ros_objs, intr, (W, H),
                         frame_id=("base_link" if calib is not None
                                   else "camera_depth_optical_frame"),
                         rot_k=getattr(src, "rotate_k", 0),
                         image=(color if a.ros_image else None),
                         obj_unit=("mm" if a.mm else "m"),
                         extra=top_extra)

            if a.save_dir and n % max(1, a.save_every) == 0:
                import cv2
                if not os.path.isdir(a.save_dir):
                    os.makedirs(a.save_dir, exist_ok=True)
                vis = color.copy()
                for (x0, y0, x1, y1, cid, sc) in dets:
                    col = RR.CFG["colors"][cid % len(RR.CFG["colors"])]
                    cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                                  col, 3)
                cv2.imwrite(os.path.join(a.save_dir, f"f{n:05d}.jpg"), vis)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if ros is not None:
            ros.close()
        if ser is not None:
            ser.close()
    dt = max(1e-6, time.time() - t_loop) if n else 1.0
    print(f"\n[DONE] {n} 帧  {n/dt:.1f} FPS（不含启动）  "
          f"（检测 {t_inf:.1f}ms + 几何 {t_geo:.1f}ms）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
