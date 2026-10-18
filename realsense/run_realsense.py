#!/usr/bin/env python3
"""
============================================================
 RealSense D435i —— 全流程入口
============================================================
 干什么：
    彩色图 → SSDlite 检测器（TensorRT）→ 类别 + 2D 框
    深度图 → 点云 → RANSAC去桌面 → 聚类 → 拟合面 → 法向量
    两者配对 → 输出【类别 + 3D坐标 + 与相机的夹角】

 跑法：
    python3 run_realsense.py                    # 实时（接显示器时开窗口）
    python3 run_realsense.py --print            # 无显示器：每帧打一行
    python3 run_realsense.py --json             # 每帧打一行 JSON
    python3 run_realsense.py --serial /dev/ttyGS0   # 结果发串口
    python3 run_realsense.py --frames 50 --save-dir out   # 存带框的图

 ★ 需要真实相机。没有相机想验证算法，用 realsense_capture.py --fake

 这个文件只负责【RealSense 取帧 + 把两个结果配对】，
 算法部分全在 ../common/ 里（相机无关）。
"""
import argparse
import json
import os
import sys
import time

import math

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "common"))
sys.path.insert(0, COMMON)
sys.path.insert(0, HERE)

import pc_pipeline                                # noqa: E402
from depth_to_cloud import depth_to_points        # noqa: E402
from ssd_decoder import generate_anchors, postprocess   # noqa: E402
from trt_infer import CFG, Engine, preprocess     # noqa: E402


# =====================================================================
def geometry_args(**over):
    """从 pc_pipeline 拿一份默认参数（保证两边默认值永远一致）"""
    bak = sys.argv
    sys.argv = [bak[0]]
    try:
        a = pc_pipeline.parse_args()
    finally:
        sys.argv = bak
    for k, v in over.items():
        setattr(a, k, v)
    a.synthetic = False
    a.replay = None
    a.json = True          # 让 run_pipeline 不要 print
    return a


def project_to_image(pt, intr, extr=None):
    """3D 点（深度相机坐标系，米）→ 彩色图像像素 (u, v)

    extr=(R,t) 是"深度→彩色"的外参。相机没做对齐时，深度和彩色光心不重合
    （D435i 相差约 1.5cm），必须先用外参把点变到彩色坐标系再投影。
    做这一步只要一次 3x3 矩阵乘法（微秒级），比 rs.align() 的 440ms 便宜太多。
    """
    X, Y, Z = float(pt[0]), float(pt[1]), float(pt[2])
    if extr is not None:
        R, t = extr
        X, Y, Z = (R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z + t[0],
                   R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z + t[1],
                   R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z + t[2])
    if Z <= 1e-6:
        return None
    return intr["fx"] * X / Z + intr["cx"], intr["fy"] * Y / Z + intr["cy"]


def rot_uv(u, v, k, w0, h0):
    """把【原始彩色图】的像素坐标换算到【旋转后】的图坐标。

    和 np.rot90(color, k=-k) 完全对应：
        k=1 (顺时针90°) : u'=H-1-v, v'=u
        k=2 (180°)      : u'=W-1-u, v'=H-1-v
        k=3 (顺时针270°): u'=v,     v'=W-1-u
    """
    if k == 1:
        return h0 - 1 - v, u
    if k == 2:
        return w0 - 1 - u, h0 - 1 - v
    if k == 3:
        return v, w0 - 1 - u
    return u, v


def _sph(p3):
    """面心 -> (距离, 方位角°, 俯仰角°)。相机系 +X右/+Y下/+Z前。"""
    # ⚠️ 必须是 `is None`，不能写 `if not p3` —— 传进来的是 numpy 数组时
    #    `not array` 会抛 "truth value of an array ... is ambiguous"。
    #    （以前凑巧没炸是因为上游给的是 list；接上车体系换算后
    #      _cam() 返回的一定是 ndarray，就必炸了。）
    if p3 is None:
        return None, None, None
    x, y, z = float(p3[0]), float(p3[1]), float(p3[2])
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-9:
        return 0.0, 0.0, 0.0
    return (round(r, 4),
            round(math.degrees(math.atan2(x, z)), 2),
            round(math.degrees(math.atan2(-y, math.hypot(x, z))), 2))


def _rng(p3):
    return _sph(p3)[0]


def _az(p3):
    return _sph(p3)[1]


def _el(p3):
    return _sph(p3)[2]


def quality_of(o):
    """给电控的"这帧够不够用"判据 —— 远距离粗定位、近距离精操作。

    依据是实测出来的规律：**横向精度 ≈ (半边长)/√面上点数**，
    也就是"物体在深度图上占的像素越多越准"。所以判据用三个量：
        面上点数 n_face_pts  —— 主因
        距离 nearest         —— 越远点越少、噪声越大
        面拟合残差 rms_mm    —— 面本身干不干净

    分级（阈值是按 40~80cm 五点扫描 + stride 对比实测标定的）：
        GOOD    可以直接拿去做抓取/精细操作
                  实测：面心抖动 ~2mm、角度 ±0.1°、尺度 ±1%
        FAIR    可以引导机械臂靠近、大致对准
                  实测：面心抖动 ~10mm、角度 ±0.3°
        COARSE  只够知道"东西大概在哪个方向"
                  实测：面心抖动 >20mm，别用来定位
    """
    n = o.get("n_face_pts") or 0
    d = o.get("nearest_m")
    rms = o.get("rms_mm")
    if rms is not None and rms > 5.0:
        return "COARSE"                  # 面都拟合不干净，别信
    # ★ 阈值是按"实测抖动"反推的（不是拍的）：
    #     面上 ~1600 点 → 抖动 0.5~2.0mm   （40cm/stride5、77cm/stride2）
    #     面上 ~1000 点 → 抖动 0.8mm       （50cm/stride5）
    #     面上 ~400  点 → 抖动 9~21mm      （80cm/stride5）→ 只够粗定位
    # ★★ 判据尽量"物理"，别用死板的距离界：
    #   真正决定横向精度的是【面上点数】（= 距离 + stride 的综合结果）。
    #   rms 则守住远端 —— 距离一远深度噪声就把面拟合搞脏，那时点数再多也不准。
    #   实测标定：
    #     n≈3600, 0.80m, rms 2.8mm → 抖动 ~2mm      ✅ GOOD
    #     n≈1600, 0.77m, rms 1.2mm → 抖动 ~2mm      ✅ GOOD
    #     n≈ 400, 0.80m, rms 3.0mm → 抖动 9~21mm    ⚠️ 只够粗定位
    #     rms > 5mm                                  ❌ 面都拟合不干净
    if n >= 1200 and d is not None and d <= 1.05:
        return "GOOD"
    if n >= 500 and d is not None and d <= 1.30:
        return "FAIR"
    return "COARSE"


def serial_obj(o, to_cam=None):
    """给电控的字段。★ 核心设计：把"直接测到的"和"推出来的"分开发。

    为什么必须分开（实测数据，相机不动、只统计算法噪声）：
        面心 face_center  抖动 5.0 mm     ← 平面拟合的直接结果，可靠
        体心 cube_center  抖动 14.0 mm    ← 推出来的，误差大 2.8 倍
        法向 → 角度       抖动 0.45°      ← 也很稳

    体心为什么不准： cube_center = face_center − (edge/2)·Σn
        · edge/2 ≈ 167mm，法向的微小抖动被放大 167 倍
        · 实测边长每帧在 32.4~33.8cm 跳，edge/2 跟着跳
        · 面数从 3 变到 2/1 时估法整个换掉（精确解 → 质心猜）→ 会大跳
    所以 n_faces / pos_type 必须一起发，让电控知道体心可不可信：
        n_faces == 3 → 体心是精确解，可以用
        n_faces <  3 → 体心有一半是猜的，建议只用 face + normal

    ★★ to_cam：车体系 → 相机系 的换算函数 `p_body -> p_cam`。
       位置字段（face/cube/normal）直接发，**它们在哪个系就是哪个系**
       （标定过就是车体系，电控要的正是这个）。
       但 range/az/el/angle 是【相对相机光轴】的量 —— 这是用户明确
       拍板保留的语义，所以这几个必须在【相机系】里算。

       ⭐ 为什么 range 跟着留在相机系、不换成"到车心距离"：
          (range, az, el) 是球坐标【三元组】，range 换基准而角度不换
          就重建不出同一个点，自相矛盾。而且 quality_of 的阈值
          (nearest ≤ 1.05 / 1.30) 是按相机距离实测标定的 —— 一换基准
          quality 整体掉成 COARSE（电控读到的意思是"这些数据别用"），
          而且不报错。留在相机系 → 零改动、零重标。

       to_cam=None 时按下不变换（相机系，RealSense 那路和以前完全一样）。
    """
    def _cam(p3):
        """把【位置】换到相机系（只给 range/az/el 用）。"""
        if p3 is None or to_cam is None:
            return p3
        return to_cam(p3)

    def _cam_dir(n3):
        """把【方向】换到相机系 —— ★ 只转，不平移。

        ⚠️ 这里踩过一次：直接对法向调 to_cam 会把平移也减掉，
           法向长度变成 |n − t| ≈ 0.6（t 是光心在车体系的位置），
           再送进 _angle_vs_axis 就完全不是角度了（acos 被 min(1.0,·) 夹住
           → 恒定 0°，而且不报错）。
        换算式：仿射映射 f(p) = R·p + t，方向部分 = f(n) − f(0) = R·n。
           不需要调用方额外提供 R —— 对任何仿射 callable 都精确成立。
        """
        if n3 is None or to_cam is None:
            return n3
        n3 = np.asarray(n3, np.float64)
        zero = np.zeros(3, np.float64)
        return np.asarray(to_cam(n3), np.float64) - np.asarray(to_cam(zero),
                                                              np.float64)

    def _vec3(v):
        """★ 出串口/ROS 的东西一律转成纯 Python —— 见 common/jsonable.py。

        为什么在源头就转，而不是只靠下游兜底：上游 pc_pipeline 给的是
        numpy 数组（face_center / normal / cube_center 都是）。
        `json.dumps(np.ndarray)` 会抛 TypeError，而串口和 ROS 的编码都在
        【后台线程】里做 —— 那个异常没人接，线程直接死，主循环却一切正常。
        源头转掉是最省事、也最不容易再犯的一层。
        """
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        try:
            a = np.asarray(v, np.float64).reshape(-1)
        except Exception:
            return v
        return [float(x) for x in a]

    def _num(v):
        """角度/分数/点数这类标量，转成原生 float/int。"""
        if v is None:
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return float(v)
        return v

    fc = _cam(o.get("face_center"))
    nc = _cam_dir(o.get("normal"))      # 法向是【方向】，只转不平移
    return {
        # ---- 身份 ----
        "cls_id": _num(o.get("cls_id")),    # 发数字，别让电控解析中文
        "cls": o.get("cls"),
        "score": _num(o.get("score")),
        # ---- ① 直接测到的（主数据，用它）----
        "face": _vec3(o.get("face_center")),   # 面心 xyz —— 与点云同一个系
        "normal": _vec3(o.get("normal")),      # 面朝向单位向量 ← 完整的"角度"
        # ---- ② 推出来的（参考，看 n_faces）----
        "cube": _vec3(o.get("cube_center")),   # 体心 xyz —— 与点云同一个系
        # ★ 面心的球坐标：距离 + 方位角 + 俯仰角（"朝向相机哪个方位"）
        #   相机系约定：+X 右 / +Y 下 / +Z 前(光轴)
        #     az = atan2(X, Z)          正 = 在相机右边
        #     el = atan2(-Y, √(X²+Z²))  正 = 在相机上方
        #   ★ 与点云在哪个系【无关】：始终在相机系里算，数值与老协议一致。
        "range": (_rng(fc)),
        "az": (_az(fc)),
        "el": (_el(fc)),
        # ★ 离相机最近的物体点距离 —— 卷尺量的就是这个，直接比
        "nearest": _num(o.get("nearest_m")),
        "edge": _num(o.get("edge_used")),   # 实测边长 (m)，夹爪开口也用它
        "edge_src": o.get("edge_src"),      # ★ auto=可信 / auto-max=勉强 / cli=不可信
        "n_faces": _num(o.get("n_faces")),
        "pos_type": o.get("pos_type"),      # cube / face
        # ★★ 这帧能不能用来做精操作（远=粗定位，近=精操作）
        "quality": quality_of(o),
        "n_face_pts": _num(o.get("n_face_pts")),
        # ★★ 给电控的一个直接可用的判据：cube 这个点能不能信。
        #    必须「3 个面都拟合出来」且「边长是各面一致时量的」，才算可信。
        #    实测：n_faces==3 & edge_src==auto 时体心抖动 ~14mm；
        #          面数掉到 2/1 时会跳几十毫米。
        "cube_ok": bool(o.get("n_faces") == 3
                        and o.get("edge_src") == "auto"),
        # ---- ③ 质量 + 可读性 ----
        # ★ angle = 法向与【相机光轴】的夹角。换车体系后光轴不再是
        #   (0,0,1)，所以必须用转回相机系的法向算，否则数值会变。
        #   （点云里那个 angle_deg 用的是输出帧的光轴，两者含义不同：
        #     这里是"相对相机看过去正不正"，那是"相对输出帧的光轴"。）
        "angle": (_num(o.get("angle")) if to_cam is None
                  else _angle_vs_axis(nc)),
        "rms_mm": _num(o.get("rms_mm")),    # 面心精度的直接证据
        # ★★ 新增：法向偏离【输出帧竖直】的角度 = 物体歪没歪。
        #   0° = 铅垂面（像一堵墙），90° = 水平面（像桌面）。
        #   lean_deg 一直没进过串口（只在 --json/ROS 里），这次补上 ——
        #   换车体系后这才是电控真正想要的量：angle 把相机自己的俯仰
        #   也算进去了，lean 才是"物体相对地面歪了多少"。
        #   ⚠️ 角度字段，【不要】加进 --mm 的长度白名单。
        "lean": _num(o.get("lean_deg")),
    }


def _angle_vs_axis(n_cam):
    """法向（相机系）与相机光轴 (0,0,1) 的夹角，度。"""
    if n_cam is None:
        return None
    nz = abs(float(n_cam[2]))
    return round(math.degrees(math.acos(min(1.0, nz))), 2)


def match_classes(dets, objects, intr, extr=None, rot_k=0, px_fix=(0.0, 0.0)):
    """
    把 2D 检测框和 3D 簇配对。
    做法：把簇的 3D 中心投影回图像，看落在哪个框里。

    dets:    [(x0,y0,x1,y1, cls_id, score)]（彩色图坐标系）
    objects: pc_pipeline 输出的 objects（相机坐标系，米）
    px_fix:  投影的像素修正，来自 AzureSource.px_fix。
             ★ 为什么需要它：点云是绕【光轴】转的（为了让输出的 XYZ 是
             世界对齐的），那等价于像点绕【主点】转；而 rot_uv 对应的是
             np.rot90，是绕【图心】转的。两者差一个固定平移。
             AzureSource 只转彩色图、点云不转（走 rot_uv）→ px_fix 恒为 0。
    返回:    [{cls, score, xyz, angle, ...}]  按分数降序
    """
    out = []
    for o in objects:
        # 优先用立方体中心（3 个面都拟合出来了），否则用主面中心
        p3 = o.get("cube_center_m") or o.get("face_center_m")
        pos = "cube" if o.get("cube_center_m") else "face"
        uv = project_to_image(p3, intr, extr)
        if uv is None:
            continue
        # 投影给的是【原始彩色图】坐标；检测框在【旋转后】的图上 → 换算
        u, v = rot_uv(uv[0], uv[1], rot_k, intr["width"], intr["height"])
        # ★ 绕光轴转 ⟺ 绕主点转，而 rot_uv 是绕图心转 —— 差的这个平移补回来
        u += px_fix[0]
        v += px_fix[1]
        if not (0 <= u < intr["width"] and 0 <= v < intr["height"]):
            pass          # 画出界也没关系，只是不会匹配上框
        best = None
        for (x0, y0, x1, y1, cid, sc) in dets:
            if x0 <= u <= x1 and y0 <= v <= y1:
                if best is None or sc > best[1]:
                    best = (cid, sc)
        out.append({
            "cls_id": None if best is None else int(best[0]),
            "cls": None if best is None else CFG["classes"][best[0]],
            "cn": None if best is None else CFG["class_cn"][CFG["classes"][best[0]]],
            "score": None if best is None else round(float(best[1]), 3),
            "pos_type": pos,
            # ★ 实际用的边长（自动量出来 / 或 --edge 兜底）
            "n_face_pts": o.get("n_face_pts"),
            "nearest_m": o.get("nearest_m"),
            "nearest_pt": o.get("nearest_pt_m"),
            "edge_used": o.get("edge_used_m"),
            "edge_src": o.get("edge_src"),
            "edge_faces": o.get("edge_faces_m"),
            "xyz": [round(float(t), 4) for t in p3],
            # ★ 分开给：xyz 是"画方块用的中心"，face_center 才是"算法真的
            #   测到的那张脸在哪"。RViz 里箭头要从 face_center 长出来，
            #   才看得出它确实垂直插在真实的那张面上。
            "face_center": o.get("face_center_m"),
            "cube_center": o.get("cube_center_m"),
            "angle": o.get("angle_deg"),
            # ★ lean_deg 必须透传 —— 换了车体系后这才是电控真正想要的
            #   "物体歪没歪"（angle 把相机自己的俯仰也算进去了）。
            #   点云没对齐到物理竖直系时它是 None，串口那边就发 null。
            "lean_deg": o.get("lean_deg"),
            "normal": o.get("normal"),
            # ★ 每个面各自的法向+面心（RViz 画"拟合面"和转正方块都用它）
            "faces": [{"normal": f.get("normal"), "center": f.get("center")}
                      for f in o.get("faces", [])],
            "n_points": o.get("n_points"),
            "n_faces": o.get("n_faces"),
            "rms_mm": o.get("face_rms_mm"),
            "uv": [round(u, 1), round(v, 1)],
        })
    out.sort(key=lambda d: -(d["score"] or 0))
    return out


# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="RealSense 全流程")
    p.add_argument("--engine", default=CFG["engine"])
    p.add_argument("--onnx", default=CFG["onnx"])
    p.add_argument("--score", type=float, default=CFG["score_thresh"])
    p.add_argument("--edge", type=float, default=0.25,
                   help="方块边长(m)。★ 只是【兜底值】—— 默认会从拟合出来的"
                        "面自己量（各面长宽一致才认），量不出来才用这个")
    p.add_argument("--no-edge-auto", dest="edge_auto", action="store_false",
                   default=True,
                   help="关掉自动量边长，强制用 --edge")
    p.add_argument("--min-dist", type=float, default=0.35)
    p.add_argument("--max-dist", type=float, default=1.5)
    p.add_argument("--plane-thresh", type=float, default=0.003)
    p.add_argument("--table-thresh", type=float, default=0.010)
    p.add_argument("--voxel", type=float, default=0.02)
    p.add_argument("--min-cluster-points", type=int, default=0,
                   help="小于这么多点的簇直接丢掉（0=按边长自动）。"
                        "场景杂乱时调大能显著提速，例如 800")
    p.add_argument("--rotate", type=int, default=0,
                   choices=[0, 90, 180, 270],
                   help="相机装歪了时用：0/90/180/270 度顺时针。"
                        "图像+深度+点云+内参一起转，2D框和3D坐标不会错位")
    p.add_argument("--only-classified", action="store_true",
                   help="只输出认得出类别的物体。深度相机视野比彩色宽，"
                        "几何会找到一些彩色图里看不到的东西（只能报坐标、没类别），"
                        "加这个开关就把它们滤掉，输出更干净")
    p.add_argument("--align", action="store_true",
                   help="打开深度→彩色对齐（精度略好，但慢 440ms/帧）。"
                        "默认关：用外参在投影时对齐，效果等价、快得多")
    p.add_argument("--flip", action="store_true",
                   help="等价于 --rotate 180（相机倒装）")
    p.add_argument("--stride", type=int, default=3,
                   help="深度图抽样倍数（越大越快，精度略降）。1=不抽，默认3")
    p.add_argument("--depth-size", default="640x480")
    p.add_argument("--color-size", default="1280x720")
    p.add_argument("--print", dest="show_text", action="store_true")
    p.add_argument("--json", dest="json_out", action="store_true")
    p.add_argument("--ros", default=None, metavar="HOST:PORT",
                   help="把点云+检测结果发给电脑上的 ROS 桥接（RViz 可视化），"
                        "例如 --ros 10.42.0.1:9100")
    p.add_argument("--ros-hz", type=float, default=10.0)
    p.add_argument("--ros-max-points", type=int, default=30000)
    p.add_argument("--no-ros-image", dest="ros_image", action="store_false",
                   default=True,
                   help="不给 RViz 发彩色图（默认发，方便把照片和点云并排核对）")
    p.add_argument("--serial", default=None,
                   help="串口设备，如 /dev/ttyGS0。接电控前先用 --serial-dry 看格式")
    p.add_argument("--serial-dry", dest="serial_dry", action="store_true",
                   help="★ 不接硬件，只把要发出去的 JSON 打到屏幕上 —— "
                        "和电控对协议之前先用这个把格式确认好")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if not os.path.exists(a.engine):
        print(f"[ERROR] 找不到 engine: {a.engine}")
        print("        先跑: python3 ../common/trt_infer.py --rebuild")
        return 1
    if not os.environ.get("DISPLAY"):
        a.no_show = True

    # ★★ 顺序很重要：必须【先初始化 CUDA/TensorRT，再打开 RealSense】！
    #    反过来（先开相机再建 TRT）会报：
    #        CUDA initialization failure with error 999
    #        pybind11::init(): factory function returned nullptr
    #    因为 RealSense 管线会先占掉一大块内存/CMA，而 Jetson 上 CUDA 初始化
    #    对可用内存很敏感，拿不到就建不了 context。
    print("[TRT] 加载 engine…")
    eng = Engine(a.engine)
    S = CFG["img_size"]
    ANCH = generate_anchors(S)

    print("[RealSense] 打开相机…")
    from realsense_capture import RealSenseSource
    dw, dh = (int(x) for x in a.depth_size.lower().split("x"))
    cw, ch = (int(x) for x in a.color_size.lower().split("x"))
    try:
        src = RealSenseSource((dw, dh), (cw, ch), flip=a.flip,
                              rotate=a.rotate, align_to_color=a.align)
    except Exception as e:
        print(f"[ERROR] 打不开相机: {type(e).__name__} {e}")
        print("        排查: python3 realsense_capture.py --list")
        return 1
    print(f"[RealSense] ✅ 深度 {dw}x{dh}  彩色 {cw}x{ch}  "
          f"深度尺度 {src.depth_scale}  "
          f"对齐={'开' if src.aligner else '关(用外参，省440ms)'}")
    extr = getattr(src, "extr", None)
    if extr is not None:
        print(f"[RealSense] 深度→彩色外参 |t| = "
              f"{float(np.linalg.norm(extr[1]))*1000:.1f} mm")

    gargs = geometry_args(edge=a.edge, min_dist=a.min_dist, max_dist=a.max_dist,
                          plane_thresh=a.plane_thresh,
                          table_thresh=a.table_thresh, voxel=a.voxel,
                          min_cluster_points=a.min_cluster_points)

    ros = None
    if a.ros:
        from ros_sender import RosSender
        rhost, _, rport = a.ros.partition(":")
        ros = RosSender(rhost, int(rport or 9100),
                        max_points=a.ros_max_points, hz=a.ros_hz)

    ser = None
    if a.serial or a.serial_dry:
        from serial_out import SerialOut
        # 干跑：SerialOut 在 port=None 时只打印不发送，方便核对协议
        ser = SerialOut(a.serial if not a.serial_dry else None,
                        a.baud, a.hz)

    n = 0
    t_start = time.time()
    t_inf = t_geo = 0.0
    try:
        while a.frames == 0 or n < a.frames:
            # ★ 把 stride 传进去：让相机模块"先抽样再反投影"，
            #   避免先算 92 万点、再扔掉重算 3.7 万点。
            r = src.get(a.stride)
            if r is None:
                continue
            depth, color, intr, (pts, valid, xyz) = r

            # ★★ 要区分两份内参，这是之前 2D/3D 配不上的元凶：
            #    intr_img : 【全分辨率】彩色图的内参 —— 把 3D 点投回图像时用
            #    intr_pts : 抽样后的内参 —— 只用于把抽样深度图反投影成点云
            #    之前抽样时把 intr 就地改了，后来又用它把 3D 点投回 1280x720，
            #    坐标被缩小了 stride 倍 → 所有 3D 点挤到画面左上角 → 永远配不上。
            intr_img = intr      # 点云所在坐标系的内参
            intr_color = src.intr_color or intr   # 彩色图内参（投回图像用）
            n += 1

            # 首帧体检：画面全黑 / 无有效深度时给明确提示，别让你瞎猜
            if n == 1:
                valid_ratio = float(valid.sum()) / max(1, depth.size)
                bright = float(color.mean())
                if bright < 15:
                    print(f"[警告] 画面几乎是全黑（平均亮度 {bright:.1f}）→ "
                          f"镜头被挡住了？")
                    print("       ① 相机是不是扣在桌上/被东西盖着？")
                    print("       ② 镜头保护膜撕了吗？")
                    print("       ③ 换到光线好的地方")
                elif valid_ratio < 0.01:
                    print(f"[警告] 有效深度只有 {valid_ratio*100:.1f}% → "
                          f"立体匹配算不出深度")
                    print("       ① 对着纯色/反光/太远的表面时会出现")
                    print("       ② 把相机对着有纹理的物体（比如桌上的方块）")

            # ---- ① 2D 检测 ----
            t0 = time.time()
            out = eng.infer(preprocess(color, S))
            cls_l = eng.get_output(out, "cls_logits")[0]
            bbox_r = eng.get_output(out, "bbox_reg")[0]
            dets = postprocess(cls_l, bbox_r, ANCH, a.score,
                               CFG["nms_iou"], CFG["max_det"])
            H, W = color.shape[:2]
            sx, sy = W / float(S), H / float(S)
            dets = [(x0 * sx, y0 * sy, x1 * sx, y1 * sy, c, sc)
                    for (x0, y0, x1, y1, c, sc) in dets]
            t_inf = (time.time() - t0) * 1000

            # ---- ② 3D 几何 ----
            t0 = time.time()
            geo = pc_pipeline.run_pipeline(pts[valid], gargs, verbose=False)
            t_geo = (time.time() - t0) * 1000

            # ---- ③ 配对 ----
            # ★ 用【彩色图】的内参 + 深度→彩色外参把 3D 点投回彩色图
            objs = match_classes(dets, geo["objects"], intr_color, extr,
                                 getattr(src, "rotate_k", 0),
                                 getattr(src, "px_fix", (0.0, 0.0)))
            if a.only_classified:
                objs = [o for o in objs if o.get("cls")]

            # ---- ④ 输出 ----
            if a.json_out:
                rec = {"frame": n, "t": round(time.time() - t_start, 3),
                       "infer_ms": round(t_inf, 1), "geo_ms": round(t_geo, 1),
                       "w": W, "h": H, "n": len(objs),
                       "obj": [{"cls": o["cls"], "cn": o["cn"], "score": o["score"],
                                "xyz": o["xyz"], "angle": o["angle"],
                                "pos_type": o["pos_type"]} for o in objs]}
                print(json.dumps(rec, ensure_ascii=False), flush=True)
            elif a.show_text:
                body = " | ".join(
                    f"{(o['cn'] or '?')}{o['cls'] or ''} {o['score'] or 0:.2f} "
                    f"XYZ=({o['xyz'][0]:.3f},{o['xyz'][1]:.3f},{o['xyz'][2]:.3f})m "
                    f"夹角={o['angle']:.1f}°" for o in objs) or "-"
                print(f"[{n:05d}] 检测{t_inf:5.1f}ms 几何{t_geo:6.1f}ms "
                      f"obj={len(objs)}  {body}", flush=True)

            if ser is not None:
                # ★ 串口给电控：面心+法向当主数据，体心当参考（见 serial_obj）
                # ★ extra 必须传 —— 不传的话这一路【永远】没有 obj_unit 头，
                #   电控不知道收到的是米还是毫米（read_serial.py 只能显示 "?"）。
                #   这一路点云本来就在相机系，所以 coord="camera"、无标定。
                ser.send(n, [serial_obj(o) for o in objs], t_inf,
                         extra={"coord": "camera", "rev": 2, "calib": None,
                                "obj_unit": "m"})

            if ros is not None:
                # 给 RViz 的物体列表：补上 edge（画立方体用）和法向量
                ros_objs = []
                for o in objs:
                    ros_objs.append({
                        "cls": o["cls"], "cn": o["cn"], "score": o["score"],
                        "xyz": o["xyz"], "angle": o["angle"],
                        "normal": o.get("normal"),
                        # ★ 面心 + 每个面的法向/面心：RViz 用它
                        #   ① 把方块标记"转正"（否则方块永远轴对齐，看着像角度错）
                        #   ② 从真实的那张面上长箭头（看得出箭是垂直插在面上的）
                        #   ③ 把拟合出来的面画成一片薄板贴在真实表面上
                        #   只挑字段传，o["pts"]（几千个点）不会上网。
                        "nearest": o.get("nearest_m"),
                        "nearest_pt": o.get("nearest_pt"),
                        "face_center": o.get("face_center"),
                        "cube_center": o.get("cube_center"),
                        "pos_type": o.get("pos_type"),
                        "faces": o.get("faces"),
                        "n_points": o.get("n_points"),
                        "rms_mm": o.get("rms_mm"),
                        # ★ 用"实际用的边长"，不是命令行那个兜底值
                        "edge": o.get("edge_used") or a.edge,
                        "edge_src": o.get("edge_src"),
                    })
                ros.send(pts[valid], ros_objs, intr, (W, H),
                         rot_k=getattr(src, "rotate_k", 0),
                         # ★ 把真实照片一起发过去：RViz 里并排看，
                         #   "这个面到底朝哪、角度对不对"一眼就能判，
                         #   不用再对着点云脑补。
                         image=(color if a.ros_image else None))

            if a.save_dir and n % max(1, a.save_every) == 0:
                if not os.path.isdir(a.save_dir):
                    os.makedirs(a.save_dir, exist_ok=True)
                vis = color.copy()
                for (x0, y0, x1, y1, cid, sc) in dets:
                    col = CFG["colors"][cid % len(CFG["colors"])]
                    cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), col, 3)
                    cv2.putText(vis, f"{CFG['classes'][cid]} {sc:.2f}",
                                (int(x0), max(20, int(y0) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                for o in objs:
                    u, v = o["uv"]
                    cv2.circle(vis, (int(u), int(v)), 8, (0, 255, 255), -1)
                    # ★ cv2.putText 画不了中文（会变成 ?），所以用英文类名
                    txt = f"{o['cls'] or 'obj'} {o['angle']:.0f}deg"
                    cv2.putText(vis, txt, (int(u) + 10, int(v)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imwrite(os.path.join(a.save_dir, f"f{n:05d}.jpg"), vis)

            if not a.no_show:
                cv2.imshow("RealSense", color)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if ros is not None:
            ros.close()
        if ser is not None:
            ser.close()
        if not a.no_show:
            cv2.destroyAllWindows()

    dt = max(1e-6, time.time() - t_start)
    print(f"\n[DONE] {n} 帧  {n/dt:.1f} FPS  "
          f"（检测 {t_inf:.1f}ms + 几何 {t_geo:.1f}ms）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
