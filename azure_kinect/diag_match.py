#!/usr/bin/env python3
"""一次性诊断：取帧 → 2D 检测 → 3D 几何 → 配对，每一步都打出来。

为什么需要：
    "obj=0" 有三种完全不同的原因，光看最终计数分不出来：
        ① 2D 检测器没认出东西      → dets 为空
        ② 几何没拟合出物体         → geo["objects"] 为空
        ③ 两边都有，但【投影像素没落进检测框】 → 配对失败
    这个脚本把三步的中间结果全打出来，一眼定位是哪一步断了。

用法（板子上，需要有 X）:
    DISPLAY=:1 python3 ~/work/azure_kinect/diag_match.py --rotate 180
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "common"))
REALSENSE = os.path.abspath(os.path.join(HERE, "..", "realsense"))
for p in (COMMON, HERE, REALSENSE):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2                                               # noqa: E402
import pc_pipeline as P                                  # noqa: E402
import importlib.util                                    # noqa: E402
from azure_capture import AzureSource                    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_rr_shared", os.path.join(REALSENSE, "run_realsense.py"))
RR = importlib.util.module_from_spec(_spec)
sys.modules["_rr_shared"] = RR
_spec.loader.exec_module(RR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotate", type=int, default=180, choices=[0, 90, 180, 270])
    # ★★ 必须和生产用【同一份标定】。投影外参 extr 是从 R_out/t_out 反推的，
    #   点云也在同一个 R_out/t_out 里 —— 两边必须同源。这个工具诊断的是
    #   "3D 点能不能落回检测框里"，正是验收 E；漏了平移 t 的话点云和 extr
    #   参考点差 0.6m → 偏几百像素 → obj=0，那是【响亮】的失败，
    #   但只有标定装对了才跑得出来。
    ap.add_argument("--calib", default=os.environ.get("K4A_BODY_CALIB"),
                    metavar="PATH", help="车体标定 json（默认 body_calib.json）")
    ap.add_argument("--allow-uncalibrated", dest="allow_uncalibrated",
                    action="store_true",
                    help="没标定也跑（输出不是车体坐标，仅用于调试）")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min-dist", type=float, default=0.3)
    ap.add_argument("--max-dist", type=float, default=1.5)
    a = ap.parse_args()

    geom = dict(min_dist=a.min_dist, max_dist=a.max_dist, voxel=0.02, edge=0.25,
                plane_thresh=0.003, table_thresh=0.010, max_faces=3,
                min_cluster_points=200, face_select="most_facing",
                dump=None, edge_auto=True, depth_noise_coef=0.008,
                json=True, synthetic=False, replay=None)

    print("[TRT] 加载 engine…", flush=True)
    eng = RR.Engine(RR.CFG["engine"])
    S = RR.CFG["img_size"]
    ANCH = RR.generate_anchors(S)
    print("[TRT] 好了", flush=True)

    import body_calib as BC
    calib = BC.load_or_exit(a.calib, rotate=a.rotate,
                            allow_missing=a.allow_uncalibrated)

    src = AzureSource(verbose=True, rotate=a.rotate, body_calib=calib)
    try:
        r = src.get(stride=a.stride)
        if r is None:
            print("❌ 取帧失败")
            return 1
        depth, color, intr, (pts, valid, xyz) = r
        H, W = color.shape[:2]
        print(f"\n彩色图 {W}x{H}   rotate_k={getattr(src, 'rotate_k', 0)}")

        # ---------- ① 2D 检测 ----------
        out = eng.infer(RR.preprocess(color, S))
        raw = RR.postprocess(eng.get_output(out, "cls_logits")[0],
                             eng.get_output(out, "bbox_reg")[0], ANCH,
                             RR.CFG["score_thresh"], RR.CFG["nms_iou"],
                             RR.CFG["max_det"])
        sx, sy = W / float(S), H / float(S)
        dets = [(x0*sx, y0*sy, x1*sx, y1*sy, c, sc)
                for (x0, y0, x1, y1, c, sc) in raw]
        print(f"\n① 2D 检测: {len(dets)} 个框")
        for (x0, y0, x1, y1, c, sc) in dets:
            print(f"   {RR.CFG['classes'][c]:12s} {sc:.2f}  "
                  f"框=({x0:6.0f},{y0:6.0f})-({x1:6.0f},{y1:6.0f})  "
                  f"中心=({(x0+x1)/2:6.0f},{(y0+y1)/2:6.0f})")

        # ---------- ② 3D 几何 ----------
        sys.argv = ["x"]
        ga = P.parse_args()
        for k, v in geom.items():
            setattr(ga, k, v)
        res = P.run_pipeline(np.asarray(pts)[valid], ga, verbose=False)
        objs = res["objects"]
        print(f"\n② 3D 几何: {len(objs)} 个物体")
        for o in objs:
            print(f"   {o['n_points']}点 {o['n_faces']}面 "
                  f"面心={np.round(o['face_center_m'],3)} "
                  f"体心={np.round(o.get('cube_center_m') or [0,0,0],3)} "
                  f"夹角={o['angle_deg']}°")

        # ---------- ③ 配对 ----------
        print("\n③ 配对（把 3D 点投回彩色图，看落在哪个框里）")
        for o in objs:
            p3 = o.get("cube_center_m") or o.get("face_center_m")
            uv = RR.project_to_image(p3, src.intr_color, src.extr)
            if uv is None:
                print("   投影失败")
                continue
            u, v = RR.rot_uv(uv[0], uv[1], getattr(src, "rotate_k", 0),
                             src.intr_color["width"], src.intr_color["height"])
            pf = getattr(src, "px_fix", (0.0, 0.0))
            u, v = u + pf[0], v + pf[1]
            hit = None
            for (x0, y0, x1, y1, c, sc) in dets:
                if x0 <= u <= x1 and y0 <= v <= y1:
                    if hit is None or sc > hit[1]:
                        hit = (RR.CFG["classes"][c], sc)
            print(f"   3D 点 {np.round(p3,3)} → 像素 ({u:6.1f},{v:6.1f})  "
                  f"{'✅ 落在 ' + hit[0] + f' ({hit[1]:.2f}) 框里' if hit else '❌ 不在任何检测框里'}")

        # 存一张标注图，十字画出来
        vis = color.copy()
        for (x0, y0, x1, y1, c, sc) in dets:
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                          (0, 255, 0), 3)
        for o in objs:
            p3 = o.get("cube_center_m") or o.get("face_center_m")
            uv = RR.project_to_image(p3, src.intr_color, src.extr)
            if uv is None:
                continue
            u, v = RR.rot_uv(uv[0], uv[1], getattr(src, "rotate_k", 0),
                             src.intr_color["width"], src.intr_color["height"])
            pf = getattr(src, "px_fix", (0.0, 0.0))
            u, v = u + pf[0], v + pf[1]
            cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255),
                           cv2.MARKER_CROSS, 40, 4)
        cv2.imwrite("/tmp/diag_match.jpg", vis)
        print("\n✅ 已存 /tmp/diag_match.jpg（绿框=检测框，红叉=3D 投影）")
        print("   判读：红叉应该在绿框里面。不在就是配对断了。")
    finally:
        src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
