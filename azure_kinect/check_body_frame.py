#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验收 C / D / F：车体坐标系到底对不对（要相机 + 卷尺）。

═══ 这个脚本存在的唯一理由 ═══
  车体系改造里最危险的错误是【180° 重复计入】：把倒装那 180° 算了两遍
  （标定矩阵里已经含了，代码又乘了一次 Rz）。它的特征非常讨厌：
      · det(R) 仍然是 +1，正交性检验全过
      · 投影到彩色图上【看不出问题】（光轴那一列是对的）
      · 图像转不转 180° 只看 rotate，和这个错无关
  也就是说：所有离线自检都会说"没问题"，上车才发现左右前后全反。

  唯一能抓住它的是【物理真值】：
      C  方块放车正前方 1.00m 中线地面 → 体心必须 ≈ (1.00, 0, ·)
         若 X≈−1.00 或 Y 反号 → 就是 180° 重复计入
      D  IMU 拿"上"在深度系的方向 up（离线的、读一次就够，不违反
         "运行时不读 IMU"）→ 断言 R_out·up ≈ (0,0,1)
         180° 翻滚会让它落到 (0,0,−1)；矩阵转置写反会以"2 倍角度"暴露
      F  地板标 3 个已知点 → 绝对误差 |Δ| < 15mm

用法（板子上，需要有 X）：
    export DISPLAY=:1
    python3 ~/work/azure_kinect/check_body_frame.py                 # 全跑
    python3 ~/work/azure_kinect/check_body_frame.py --only imu      # 只跑 D
    python3 ~/work/azure_kinect/check_body_frame.py --tape 1.00 0.0 # C 的人工真值

★ 每一关都会打印"失败时最可能是哪个错"，因为这几个错误的形态很像，
  光看数字分不出来。
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "common"))
REALSENSE = os.path.abspath(os.path.join(HERE, "..", "realsense"))
for _p in (COMMON, HERE, REALSENSE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import body_calib as BC                                   # noqa: E402
import pc_pipeline as P                                   # noqa: E402
from azure_capture import AzureSource                     # noqa: E402

FAILS = []


def ok(tag, cond, detail=""):
    print(f"  {tag:<40} {'✅' if cond else '❌'}  {detail}")
    if not cond:
        FAILS.append(tag)
    return cond


def verdict(name, diag):
    """失败时给出"最可能是什么错"。"""
    print(f"\n  ┌─ {name} 失败时怎么读 ────────────────────────────")
    for line in diag:
        print(f"  │ {line}")
    print("  └" + "─" * 50)


# ==================================================================== C
def check_ground_target(src, a, calib):
    """C：车正前方 L 米、中线上的地面目标 → 体心应该 ≈ (L, 0, ·)。"""
    print(f"\n── C. 地面标点（车正前方 {a.tape_fwd:.2f} m，横向 {a.tape_left:+.2f} m）"
          " ──")
    print("   ★ 把方块放在那儿再按回车。位置用卷尺从【车体几何中心】量。")
    input("   放好了按回车… ")
    r = src.get(stride=a.stride)
    if r is None:
        ok("取帧", False, "取帧失败")
        return
    _d, _c, _i, (pts, valid, _xyz) = r
    gargs = _gargs(a, src, calib)
    res = P.run_pipeline(np.asarray(pts)[valid], gargs, verbose=False)
    if not res["objects"]:
        ok("检出物体", False, "没检出 —— 方块在画面里吗？")
        return
    o = max(res["objects"], key=lambda x: x["n_points"])
    p = np.asarray(o.get("cube_center") or o.get("face_center"), float)

    d_fwd = p[0] - a.tape_fwd
    d_left = p[1] - a.tape_left
    print(f"   实测车体系 (X,Y,Z) = ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) m")
    print(f"   真值               = ({a.tape_fwd:+.4f}, {a.tape_left:+.4f}, Z?) m")
    print(f"   前向误差 {d_fwd*1000:+.1f} mm   横向误差 {d_left*1000:+.1f} mm")

    tol = a.tol_mm / 1000.0
    ok("前向 X 在容差内", abs(d_fwd) < tol, f"|Δ|={abs(d_fwd)*1000:.1f}mm")
    ok("横向 Y 在容差内", abs(d_left) < tol, f"|Δ|={abs(d_left)*1000:.1f}mm")

    # ★★ 180° 重复计入的指纹
    flipped_x = abs(p[0] + a.tape_fwd) < tol and a.tape_fwd > 0.3
    flipped_y = abs(p[1] + a.tape_left) < tol and abs(a.tape_left) > 0.1
    if flipped_x:
        print("   ❗ X 反号且量值对得上 —— 这是 180° 重复计入的指纹")
    if flipped_y:
        print("   ❗ Y 反号且量值对得上 —— 这是 180° 重复计入的指纹")
    verdict("C", [
        "X≈−真值 或 Y≈−真值 且量值吻合 → 标定矩阵里已含 180°，" ,
        "   代码又绕光轴乘了一次 Rz。查 azure_capture._finish_rotations()",
        "   的 `if self.R_body is not None: self.R_out = self.R_body` 分支",
        "   （有标定时【不能】再乘 Rz）。",
        "X/Y 交换 → 标定文件的 R 行列写反了（转置）。",
        "都对但 Z 差一个固定值 → height_mm 量错（原点高度）。",
    ])
    # 尺度线性：再放远一点对比
    if a.also_far:
        print(f"\n   再放一个在 {a.also_far:.2f} m…")
        input("   放好了按回车… ")
        r = src.get(stride=a.stride)
        if r:
            _d, _c, _i, (pts, valid, _xyz) = r
            res2 = P.run_pipeline(np.asarray(pts)[valid],
                                  _gargs(a, src, calib), verbose=False)
            if res2["objects"]:
                o2 = max(res2["objects"], key=lambda x: x["n_points"])
                p2 = np.asarray(o2.get("cube_center") or o2.get("face_center"),
                                float)
                ratio = p2[0] / max(a.also_far, 1e-6)
                print(f"   实测 X={p2[0]:+.4f} m，真值 {a.also_far:.2f} m，"
                      f"比值 {ratio:.4f}（应≈1.000）")
                ok("尺度线性（远点）", abs(ratio - 1.0) < 0.05,
                   f"比值 {ratio:.4f}")


# ==================================================================== D
def check_imu(src, a, calib):
    """D：IMU 绝对交叉验证 —— 性价比最高的一关。"""
    print("\n── D. IMU 绝对交叉验证（车摆水平）────────────────")
    print("   离线的、读一次就够 —— 不违反『运行时不读 IMU』。")
    if not getattr(src, "_read_gravity", None):
        ok("相机支持读 IMU", False, "这个 AzureSource 没有重力读取")
        return
    # ★ _read_gravity 返回 (up, info)。up 是【深度原始系①】里的"上"，
    #   R_out 正是 ①→输出帧 的映射 —— 所以 R_out·up 应该 == 车体系 +Z。
    up, info = src._read_gravity()    # 深度光学系里的"上"
    if up is None:
        ok("读到 IMU", False, f"IMU 读不到：{info.get('err')}")
        return
    up = np.asarray(up, float)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    R_out = np.asarray(src.R_out, float)
    got = R_out @ up                  # 车体系里"上"应该 == +Z
    want = np.array([0.0, 0.0, 1.0])
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(got, want))))))
    print(f"   IMU 的『上』在深度系 = {np.round(up, 5).tolist()}")
    print(f"   R_out·up（车体系）   = {np.round(got, 5).tolist()}")
    print(f"   与 (0,0,1) 的夹角     = {ang:.3f}°")
    ok("残差 < 0.5°", ang < 0.5, f"{ang:.3f}°")
    verdict("D", [
        "落在 (0,0,−1)（夹角 ≈180°）→ 180° 重复计入，或 roll 写成 0。",
        "夹角 ≈ 2 倍相机俯仰（~19°）→ 矩阵转置写反（det 仍为 +1，",
        "   所以只有这一关能抓住）。",
        "方向偏但角度不大 → 卷尺量的 pitch/yaw 有误差，回 body_calib",
        "   用 IMU 反推 pitch 重写标定（--pitch-from-target）。",
    ])


# ==================================================================== F
def check_tape_points(src, a, calib):
    """F：地板标 3 个已知点 → 绝对误差。"""
    print("\n── F. 卷尺绝对比对（地板标 3 个已知点）──────────")
    print("   ★ 在地板上用胶带标 3 个点，量出它们相对车体几何中心的")
    print("     前向/横向坐标，然后逐个把方块放上去测。")
    pts_true = []
    for i in range(a.n_tape_pts):
        raw = input(f"  第{i+1}点 前向(m),横向(m)（回车跳过）: ").strip()
        if not raw:
            break
        try:
            f_, l_ = [float(x) for x in raw.replace("，", ",").split(",")]
        except Exception:
            print("    格式不对，跳过")
            continue
        input(f"    把方块放到第{i+1}点，按回车… ")
        r = src.get(stride=a.stride)
        if r is None:
            continue
        _d, _c, _i, (pts, valid, _xyz) = r
        res = P.run_pipeline(np.asarray(pts)[valid], _gargs(a, src, calib),
                             verbose=False)
        if not res["objects"]:
            print("    ❌ 没检出")
            continue
        o = max(res["objects"], key=lambda x: x["n_points"])
        p = np.asarray(o.get("cube_center") or o.get("face_center"), float)
        e = math.hypot(p[0] - f_, p[1] - l_)
        print(f"    第{i+1}点 实测({p[0]:+.4f},{p[1]:+.4f}) "
              f"真值({f_:+.4f},{l_:+.4f})  |Δ|={e*1000:.1f} mm")
        pts_true.append(e)
    if not pts_true:
        print("   ⏭  一个点都没测，跳过")
        return
    worst = max(pts_true) * 1000
    ok("3 点最大绝对误差 < 15mm", worst < 15.0, f"{worst:.1f} mm")


# ==================================================================== 公共
def _gargs(a, src, calib):
    """和 run_azure 里那份一模一样的坐标系参数（不能各写一套）。"""
    import types
    g = types.SimpleNamespace(
        min_dist=a.min_dist, max_dist=a.max_dist, voxel=a.voxel,
        edge=a.edge, plane_thresh=a.plane_thresh, table_thresh=a.table_thresh,
        max_faces=a.max_faces, min_cluster_points=a.min_cluster_points,
        face_select=a.face_select, dump=None, edge_auto=True,
        depth_noise_coef=0.008, json=True, synthetic=False, replay=None,
        stride=a.stride, split_boxes=False, split_uv_fn=None,
        up_axis=([0.0, 0.0, 1.0] if calib is not None
                 else ([0.0, 1.0, 0.0] if getattr(src, "gravity_ok", False)
                       else None)),
        cam_axis=np.asarray(src.R_out, float)[:, 2].tolist(),
        cam_origin=np.asarray(src.t_out, float).tolist(),
        gravity=bool(getattr(src, "gravity_ok", False)))
    return g


def main():
    ap = argparse.ArgumentParser(description="车体坐标系物理验收 C/D/F")
    ap.add_argument("--rotate", type=int, default=180,
                    choices=[0, 90, 180, 270])
    ap.add_argument("--calib", default=os.environ.get("K4A_BODY_CALIB"))
    ap.add_argument("--allow-uncalibrated", dest="allow_uncalibrated",
                    action="store_true")
    ap.add_argument("--only", default="all",
                    choices=["all", "c", "d", "f"],
                    help="只跑某一关（默认全跑）")
    ap.add_argument("--tape-fwd", dest="tape_fwd", type=float, default=1.00,
                    help="C：方块前向真值（米，卷尺量）")
    ap.add_argument("--tape-left", dest="tape_left", type=float, default=0.00,
                    help="C：方块横向真值（米，左为正）")
    ap.add_argument("--also-far", dest="also_far", type=float, default=0.0,
                    help="C：再测一个更远的距离看尺度线性（0=不测）")
    ap.add_argument("--tol-mm", dest="tol_mm", type=float, default=15.0)
    ap.add_argument("--n-tape-pts", dest="n_tape_pts", type=int, default=3)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min-dist", type=float, default=0.3)
    ap.add_argument("--max-dist", type=float, default=1.5)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--edge", type=float, default=0.25)
    ap.add_argument("--plane-thresh", dest="plane_thresh", type=float,
                    default=0.003)
    ap.add_argument("--table-thresh", dest="table_thresh", type=float,
                    default=0.010)
    ap.add_argument("--max-faces", dest="max_faces", type=int, default=3)
    ap.add_argument("--min-cluster-points", dest="min_cluster_points",
                    type=int, default=200)
    ap.add_argument("--face-select", dest="face_select",
                    default="most_facing")
    a = ap.parse_args()

    print("═══ 车体坐标系物理验收 C/D/F ═══")
    calib = BC.load_or_exit(a.calib, rotate=a.rotate,
                            allow_missing=a.allow_uncalibrated)
    if calib is None:
        print("\n❌ 没有标定文件 —— C/D/F 验的正是标定本身，没它毫无意义。")
        return 1
    print(f"   标定 {calib['id']}  R=\n{np.round(calib['R'], 5)}")
    print(f"   t = {np.round(calib['t'], 4).tolist()} m")

    src = AzureSource(verbose=True, rotate=a.rotate, body_calib=calib)
    try:
        if a.only in ("all", "d"):
            check_imu(src, a, calib)
        if a.only in ("all", "c"):
            check_ground_target(src, a, calib)
        if a.only in ("all", "f"):
            check_tape_points(src, a, calib)
    finally:
        src.close()

    print("\n" + "═" * 52)
    if FAILS:
        print(f"❌ {len(FAILS)} 条失败：")
        for t in FAILS:
            print(f"   · {t}")
        print("\n  ⚠️ 这几条没过【不要上车】—— 它们抓的都是静默错误。")
        return 1
    print("✅ 全绿（C/D/F 都过）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
