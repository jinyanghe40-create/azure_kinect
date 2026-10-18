#!/usr/bin/env python3
"""把几何管线拟合出来的【面】画回真实照片上，用来核查角度对不对。

为什么需要它：
    "角度算错了" 这种反馈，光看数字没法判断 ——
    是算法错了？还是选错了面？还是显示方式有歧义？
    把【面心 + 法向】按外参投回彩色照片，箭头和真实的面一对比，
    三秒钟就能分辨。

用法（PC 上）:
    python3 tools/verify_angle.py --bundle /tmp/bundle --out /tmp/verify
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "common"))
import pc_pipeline                                     # noqa: E402
import cv2                                             # noqa: E402


def rot_uv(u, v, k, w0, h0):
    """原始彩色图坐标 -> 旋转后图坐标（和 np.rot90(color, k=-k) 对应）。"""
    if k == 1:
        return h0 - 1 - v, u
    if k == 2:
        return w0 - 1 - u, h0 - 1 - v
    if k == 3:
        return v, w0 - 1 - u
    return u, v


class Proj:
    """深度系 3D 点 -> 旋转后彩色图像素坐标。"""

    def __init__(self, meta):
        self.R = np.array(meta["extr_R"], float)
        self.t = np.array(meta["extr_t"], float)
        self.ic = meta["intr_color"]
        self.k = int(meta.get("rotate_k", 0))
        self.w0 = int(self.ic["width"])
        self.h0 = int(self.ic["height"])

    def __call__(self, p3):
        p3 = np.asarray(p3, float)
        pc = p3 @ self.R.T + self.t          # 深度系 -> 彩色系
        if pc[..., 2] <= 1e-3:
            return None
        u = self.ic["fx"] * pc[..., 0] / pc[..., 2] + self.ic["cx"]
        v = self.ic["fy"] * pc[..., 1] / pc[..., 2] + self.ic["cy"]
        return rot_uv(u, v, self.k, self.w0, self.h0)

    def many(self, pts):
        pc = np.asarray(pts, float) @ self.R.T + self.t
        ok = pc[:, 2] > 1e-3
        u = np.full(len(pc), -1.0)
        v = np.full(len(pc), -1.0)
        u[ok] = self.ic["fx"] * pc[ok, 0] / pc[ok, 2] + self.ic["cx"]
        v[ok] = self.ic["fy"] * pc[ok, 1] / pc[ok, 2] + self.ic["cy"]
        u2, v2 = rot_uv(u, v, self.k, self.w0, self.h0)
        return np.stack([u2, v2], axis=1), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/tmp/bundle")
    ap.add_argument("--out", default="/tmp/verify")
    ap.add_argument("--min-dist", type=float, default=0.2)
    ap.add_argument("--max-dist", type=float, default=1.2)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--edge", type=float, default=0.25)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--plane-thresh", type=float, default=0.003)
    ap.add_argument("--table-thresh", type=float, default=0.010)
    ap.add_argument("--max-faces", type=int, default=3)
    ap.add_argument("--min-cluster-points", type=int, default=200)
    ap.add_argument("--face-select", default="most_facing")
    a = ap.parse_args()

    pts = np.load(os.path.join(a.bundle, "grab_pts.npy"))
    meta = json.load(open(os.path.join(a.bundle, "grab_meta.json")))
    img = cv2.imread(os.path.join(a.bundle, "grab_color.jpg"))
    print(f"点云 {pts.shape}  彩色图 {img.shape}  rotate_k={meta['rotate_k']}")

    args = argparse.Namespace(
        min_dist=a.min_dist, max_dist=a.max_dist, edge=a.edge, voxel=a.voxel,
        plane_thresh=a.plane_thresh, table_thresh=a.table_thresh,
        max_faces=a.max_faces, min_cluster_points=a.min_cluster_points,
        face_select=a.face_select, dump=None)

    res = pc_pipeline.run_pipeline(pts.copy(), args, verbose=True)
    objs = res["objects"]
    print(f"\n===== 共 {len(objs)} 个物体 =====")
    for i, o in enumerate(objs):
        print(f"[{i}] {o['n_points']} 点  {o['n_faces']} 面  "
              f"选中夹角={o['angle_deg']}°  法向={o['normal']}")
        for j, f in enumerate(o.get("faces", [])):
            nz = abs(f["normal"][2])
            ang = float(np.degrees(np.arccos(min(1.0, nz))))
            mark = "  ← 选中" if abs(f["angle_deg"] - o["angle_deg"]) < 1e-6 \
                else ""
            print(f"      面{j}: 法向={f['normal']}  夹角={ang:5.1f}° "
                  f"{f['n_points']:>5}点 RMS={f.get('rms_mm', 0):.1f}mm{mark}")

    # ---- 画到照片上 ----
    P = Proj(meta)
    vis = img.copy()
    palette = [(60, 60, 240), (60, 200, 60), (240, 160, 40), (200, 60, 200)]

    # 把物体点投影上去，看覆盖范围对不对
    for i, o in enumerate(objs):
        cp = np.asarray(o.get("pts", []))
        if len(cp) == 0:
            continue
        uv, ok = P.many(cp[::3])
        col = palette[i % len(palette)]
        for (u, v), good in zip(uv, ok):
            if good and 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
                cv2.circle(vis, (int(u), int(v)), 1, col, -1)

    for i, o in enumerate(objs):
        for j, f in enumerate(o.get("faces", [])):
            c = np.asarray(f["center"], float)
            n = np.asarray(f["normal"], float)
            n = n / max(1e-9, np.linalg.norm(n))
            uv0 = P(c)
            uv1 = P(c + n * 0.10)
            if uv0 is None or uv1 is None:
                continue
            p0 = (int(uv0[0]), int(uv0[1]))
            p1 = (int(uv1[0]), int(uv1[1]))
            selected = abs(f["angle_deg"] - o["angle_deg"]) < 1e-6
            col = (0, 0, 255) if selected else (255, 0, 255)
            cv2.arrowedLine(vis, p0, p1, col, 4 if selected else 2,
                            cv2.LINE_AA, tipLength=0.25)
            cv2.circle(vis, p0, 7, col, 2)
            ang = np.degrees(np.arccos(min(1.0, abs(f["normal"][2]))))
            cv2.putText(vis, f"f{j} {ang:.1f}deg",
                        (p0[0] + 10, p0[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, col, 2, cv2.LINE_AA)

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "verify_overlay.jpg")
    cv2.imwrite(p, vis)
    print(f"\n✅ 叠加图: {p}  (红箭头=选中面, 品红=其它面)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
