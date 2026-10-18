#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""车体标定文件 —— 相机 → 车体 的固定刚体变换。

纯 numpy，**不依赖 Azure Kinect**，所以笔记本上就能生成、检查、看报告。

═══════════════════════════════════════════════════════════════════
★ 这个文件里最要紧的一件事：【矩阵存的是哪个坐标系】
═══════════════════════════════════════════════════════════════════

  ① raw SDK 深度光学系   +X=图像右 / +Y=图像下 / +Z=光轴
  ③ post-Rz 系           ①绕光轴转 image_rotate_deg（本车 = 180°）
  ④ 车体系（输出）        +X前 / +Y左 / +Z上（ROS REP-103）

  `R_cam_to_body` 存的是 **①→④**，也就是【物理真相】，**那 180° 倒装
  已经写在矩阵里了**。

  ⚠️ 存 ① 还是存 ③ 会在真机上造成一个**完全静默**的错误：
     如果误存成 ③（post-Rz），运行时会再乘一次 Rz(180)，
     点云【整个翻过去】—— 而 det 依然是 +1（还是个合法旋转），
     没有任何一处会报错。**只有下面这条判据能挡住它。**

  ★ 判据（`upright_residual` / `--check` 会跑）：

        (R_cam_to_body @ Rz(image_rotate_deg)ᵀ)[2,1] < 0

    含义：把相机的图像翻滚【还原】之后，"图像里的下"必须指向物理下方。
    ⚠️ 注意【不是】`R_cam_to_body[2,1] < 0` —— 那个写法隐含假设
       rotate=0。本车 rotate=180，正确的 R_cam_to_body[2,1] = +1，
       用不带 Rz 的写法会【把对的标定拒掉】。
    ⚠️ 光轴那一列（第三列）在两种约定里【完全一样】，
       所以"指着哪里"这种检查区分不出 ① 和 ③，**只有第二列能区分**。
       这正是当年 `--rotate 180` 那类 bug 最容易被漏掉的原因。

  位置没有帧歧义：`t_cam_to_body_m` = 从【车体几何中心】指向
  【深度传感器正面玻璃中心】的向量（前/左/上，带符号，单位 m）。

═══════════════════════════════════════════════════════════════════
命令行
═══════════════════════════════════════════════════════════════════
  # 卷尺量完，生成标定文件（长度单位全用 mm，跟卷尺一致）
  python3 body_calib.py --write body_calib.json \\
      --pitch 14.5 --yaw 0 --roll 180 \\
      --forward 250 --left 0 --height 600 \\
      --id buggy01-20261015

  # 检查一份标定（正交性 / det / 竖直判据 / 数值区间）
  python3 body_calib.py --check body_calib.json

  # 打印人眼可核对的报告（基向量都摆出来，方向对不对一眼看得出）
  python3 body_calib.py --verify body_calib.json

  # 不知道该填哪个 pitch？把靶板摆在正前方 L 米，量靶心比传感器轴心
  # 低/高多少 Δh（mm，低了给正数），它会帮你算
  python3 body_calib.py --pitch-from-target --L 1000 --drop 171
"""
import argparse
import json
import math
import os
import sys

import numpy as np

SCHEMA = "k4a_body_calib"
VERSION = 1
# 服务单元里写死的默认路径（同目录）
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "body_calib.json")


class BodyCalibError(RuntimeError):
    """标定文件缺失/不合法 —— 调用方应当【拒绝启动】，不要静默退回相机系。"""


# =====================================================================
# 基础旋转（全用【角度】，因为卷尺量出来的就是角度）
# =====================================================================
def Rz_optical(rotate_deg):
    """绕【光轴】转 rotate_deg —— 必须与 azure_capture._Rz() 逐位一致。

    ★ 这是"图像翻滚"那个矩阵。它同时用在点云旋转和彩色图旋转上，
      所以这里的符号约定跟 azure_capture 那边【必须】是一个。
      （azure_capture._Rz 用的是 +X右/+Y下 的光学系，逆时针为正。）
    """
    k = (int(rotate_deg) // 90) if rotate_deg else 0
    if not k:
        return np.eye(3)
    th = math.radians(k * 90.0)
    c_, s_ = math.cos(th), math.sin(th)
    return np.array([[c_, -s_, 0.0],
                     [s_, c_, 0.0],
                     [0.0, 0.0, 1.0]])


def _Rx(d):
    t = math.radians(d)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _Ry(d):
    t = math.radians(d)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _Rz_body(d):
    """绕【车体 Z（上）】转 —— 偏航。别跟 Rz_optical 搞混。"""
    t = math.radians(d)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# 「正装、朝前、无翻滚」的基准姿态：①→④
#   列 = ① 的三个基在车体系里的方向
#     Xc(图像右) → (0,-1,0)  车体【右】   （−Y 就是车右）
#     Yc(图像下) → (0,0,-1)  车体【下】
#     Zc(光轴)   → (1,0,0)   车体【前】
C0_UPRIGHT = np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]])


def build_R(yaw_deg=0.0, pitch_deg=0.0, roll_deg=180.0):
    """组 R_cam_to_body（①→④）。

    R = Rz_body(yaw) · Ry_body(pitch) · C0_upright · Rz_optical(roll)

    ★ 顺序【写死在这里】，生成的报告里会把三个基向量都打出来供肉眼核对 ——
      顺序写反或符号搞错，同一维会翻 2 倍角度，验收 D（IMU）立刻抓出。

    角度符号（**刻意写清楚，别猜**）：
      pitch_deg > 0 → 光轴【向下俯视】（正前方为 0）
      yaw_deg   > 0 → 光轴偏向车体【左侧】（正前方为 0）
      roll_deg      → 绕光轴的图像翻滚。本车倒装 = 180
    """
    return (_Rz_body(yaw_deg) @ _Ry(pitch_deg)
            @ C0_UPRIGHT @ Rz_optical(roll_deg))


def decompose_R(R):
    """`build_R` 的逆 —— 从矩阵反解出 (yaw, pitch, roll)，用于报告/自检。

    推导：R 的第三列 = 光轴在车体系的方向 a。
        a = Rz(yaw)·Ry(pitch)·C0·Rz_optical(roll)·e₃
          = Rz(yaw)·Ry(pitch)·(1,0,0) = (cos p cos y, cos p sin y, −sin p)
        ★ roll 不影响光轴 —— 它是绕光轴转的，所以 pitch/yaw 可以单独解
        → pitch = asin(−a_z),  yaw = atan2(a_y, a_x)

      R 的第二列 = "图像下"方向 d。把 yaw/pitch 去掉后：
        Rz_optical(roll)·e₁ = (−sin r, cos r, 0)
        C0 · 那个          = −sin r·C0[:,0] + cos r·C0[:,1] = (0, sin r, −cos r)
        u = [Rz(yaw)·Ry(pitch)]⁻¹ · d = Ry(−pitch)·Rz(−yaw)·d = (0, sin r, −cos r)
        → roll = atan2(u_y, −u_z)
        ⚠️ 逆序是 Ry(−pitch)·Rz(−yaw)，【不是】Rz(−yaw)·Ry(−pitch) ——
           矩阵乘法不可交换，写反了会差十几度（俯仰+偏航同时非零时）。
    """
    a = np.asarray(R, float)[:, 2]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -a[2]))))
    yaw = math.degrees(math.atan2(a[1], a[0]))
    d = np.asarray(R, float)[:, 1]
    u = _Ry(-pitch) @ _Rz_body(-yaw) @ d
    roll = math.degrees(math.atan2(u[1], -u[2]))
    return yaw, pitch, roll


def upright_residual(R, rotate_deg):
    """★ 那条判据的数值：(R @ Rz(rotate)ᵀ)[2,1]。

    要求 < 0。见模块开头的长注释 —— 这是挡住「180° 重复计入」的唯一防线。

    顺带说明它为什么恰好是第二列的 [2] 号元素：
      (R @ Rzᵀ) 就是"把相机当成正装"的那个姿态；在正装姿态下，
      "图像里的下"必须指向物理下方，也就是该矩阵第二列的 Z 分量为负。
    """
    return float((np.asarray(R, float) @ Rz_optical(rotate_deg).T)[2, 1])


# =====================================================================
# 校验
# =====================================================================
def _reorthonormalize(R):
    """把接近旋转但数值上不够正交的矩阵按 SVD 投影回 SO(3)。"""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:            # 反射 → 强行掰回旋转
        U[:, -1] = -U[:, -1]
        Rn = U @ Vt
    return Rn


def check_calib(d, rotate=None, fix=False):
    """校验一份标定 dict。返回 (R, t, rotate_deg, warnings[])，不合法就抛异常。

    rotate 给了就顺便断言 image_rotate_deg 一致 —— 这是零成本的强校验，
    同时挡住"忘了给服务加 --rotate 180"这个 bug（服务里现在是 rotate=0！）。
    """
    warn = []
    if not isinstance(d, dict):
        raise BodyCalibError("标定文件顶层不是 JSON 对象")
    if d.get("schema") != SCHEMA:
        raise BodyCalibError(
            f"schema 不对：期望 {SCHEMA!r}，实际 {d.get('schema')!r}。"
            "（是不是把 k4a_calibration.py 的输出当车体标定用了？）")

    for k in ("id", "image_rotate_deg", "R_cam_to_body", "t_cam_to_body_m"):
        if k not in d:
            raise BodyCalibError(f"标定缺字段 {k!r}")

    rot = int(d["image_rotate_deg"])
    if rot % 90:
        raise BodyCalibError(f"image_rotate_deg={rot} 不是 90 的整数倍")
    if rotate is not None and int(rotate) % 360 != rot % 360:
        raise BodyCalibError(
            f"命令行 --rotate {rotate} 与标定的 image_rotate_deg {rot} 不一致。\n"
            "  ★ 这是故意设的硬断言：服务单元里曾经漏掉 --rotate 180，\n"
            "    于是彩色图是倒的、点云却按 0° 解释 —— 图倒着看得见，\n"
            "    点云错得静默。两者必须来自同一个数。\n"
            "  要么改标定，要么改命令行，别让它俩各说各话。")

    R = np.asarray(d["R_cam_to_body"], float)
    if R.shape != (3, 3):
        raise BodyCalibError(f"R_cam_to_body 形状 {R.shape}，要 3×3"
                             "（行主序 [[r00,r01,r02],[...],[...]]）")
    t = np.asarray(d["t_cam_to_body_m"], float).reshape(-1)
    if t.shape != (3,):
        raise BodyCalibError(f"t_cam_to_body_m 要 3 个数，实际 {t.shape}")

    # ---- 正交性 / det ----
    det = float(np.linalg.det(R))
    orth = float(np.abs(R @ R.T - np.eye(3)).max())
    if abs(det - 1.0) > 1e-6 or orth > 1e-6:
        if not fix:
            raise BodyCalibError(
                f"R 不是合法旋转：det={det:.9f}（要 +1），"
                f"|RRᵀ−I|max={orth:.3e}（要 <1e-6）。\n"
                "  用 --check --fix 可以让它按 SVD 重新正交化后写回。")
        warn.append(f"R 数值上不是严格旋转（det={det:.9f}, |RRᵀ−I|={orth:.3e}）"
                    "→ 已按 SVD 重新正交化")
        R = _reorthonormalize(R)
        det = float(np.linalg.det(R))

    # ---- ★ 竖直判据（唯一能挡住 180° 重复计入的） ----
    res = upright_residual(R, rot)
    if res > -1e-9:
        raise BodyCalibError(
            f"没通过『图像竖直』判据：(R @ Rz({rot})ᵀ)[2,1] = {res:+.4f} ≥ 0\n"
            "  含义：把图像翻滚还原后，『图像里的下』没有指向车体下方。\n"
            "  · 第二列 ≈ +1：矩阵是按 post-Rz 系（③）写的 → 180° 会被\n"
            "    重复计入，点云整个翻过去，而且 det 仍是 +1，完全静默。\n"
            "    应当存 raw SDK 深度系 → 车体系的物理真相（①→④）。\n"
            "  · 第三列也不像光轴：多半是矩阵转置写反了。\n"
            "  提示： python3 body_calib.py --verify <文件> 会把三个基向量\n"
            "        和推导出的 yaw/pitch/roll 都打出来，一眼看得出哪错了。")
    if res > -0.5:
        warn.append(f"竖直判据只勉强通过（{res:+.3f}，理想接近 −1）—— "
                    "光轴接近水平但『下』方向偏了，检查 roll/矩阵是否写歪")

    # ---- 位置 ----
    tn = float(np.linalg.norm(t))
    if not (0.0 < tn <= 2.0):
        raise BodyCalibError(f"|t| = {tn:.4f} m 不在 (0, 2] 内。"
                             "单位是不是搞错了（mm 当 m 写）？")

    return R, t, rot, warn


# =====================================================================
# 装载 / 写入
# =====================================================================
def load_body_calib(path=None, rotate=None, allow_missing=False):
    """读标定文件 → azure_capture 要的那个 dict。

    返回 {"R": (3,3) float64, "t": (3,) float64, "id": str,
          "image_rotate_deg": int, "path": str, "warnings": [...]}

    ★ 默认【拒绝启动】：文件不在/不合法就抛 BodyCalibError。
      绝不静默退回相机系 —— 电控拿到的是车体系坐标，混着相机系过去
      会被当成"车前方 0.3m 有东西"直接撞上去。
    """
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        if allow_missing:
            return None
        raise BodyCalibError(
            f"没有车体标定文件：{path}\n"
            "  相机→车体必须先标一次（刚性固定，一次永久用）。\n"
            "  生成：  python3 body_calib.py --write %s --pitch .. "
            "--forward .. --left .. --height ..\n"
            "  临时跳过：加 --allow-uncalibrated（会退回相机系/重力对齐，"
            "输出【不是】车体坐标，别接电控）" % path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        raise BodyCalibError(f"标定文件读不了：{path}（{type(e).__name__}: {e}）")

    R, t, rot, warn = check_calib(d, rotate=rotate)
    return {"R": R, "t": t, "id": str(d.get("id", "")),
            "image_rotate_deg": rot, "path": path, "warnings": warn,
            "raw": d}


def load_or_exit(path=None, rotate=None, allow_missing=False, quiet=False):
    """命令行工具的公共入口：载入标定，失败就打印 + SystemExit(1)。

    ★ 为什么要有这个函数：run_azure / check_projection / diag_match /
      check_body_frame 四处都要载标定。各写一遍的结果是【诊断工具和生产
      用了不同的标定】，于是"diag 说投影是对的"和"生产 obj=0"可以同时
      为真 —— 这种不一致排查起来极贵。收成一个函数，谁都不会跑偏。
    """
    try:
        calib = load_body_calib(path, rotate=rotate, allow_missing=allow_missing)
    except BodyCalibError as e:
        print(f"\n❌ 车体标定不可用，拒绝启动：\n{e}\n", file=sys.stderr)
        raise SystemExit(1)
    if not quiet:
        if calib is None:
            print("[!] 没有车体标定：输出【不是】车体坐标。别接电控。",
                  file=sys.stderr)
        else:
            for w in calib.get("warnings", []):
                print(f"[标定] ⚠️  {w}")
            print(f"[标定] {calib['id']}  光心在车体系 "
                  f"{np.round(calib['t'], 4).tolist()} m  "
                  f"（车体系 X前/Y左/Z上，原点=车体几何中心）")
    return calib


def write_body_calib(path, yaw_deg, pitch_deg, roll_deg,
                     forward_mm, left_mm, height_mm,
                     calib_id="", serial="", depth_mode="nfov", color="720p",
                     tape_extra=None):
    """量出来的数 → 标定文件。长度进来是 mm（跟卷尺一致），存下去是 m。"""
    R = build_R(yaw_deg, pitch_deg, roll_deg)
    t = np.array([forward_mm, left_mm, height_mm], float) / 1000.0
    d = {
        "schema": SCHEMA, "version": VERSION,
        "id": calib_id or "unnamed",
        "created": _today(),
        "body_frame": {
            "convention": "ROS REP-103: X前 / Y左 / Z上",
            "origin": "chassis geometric center (含高度)",
            "units": "position in m, angle in deg",
        },
        "camera": {"serial": serial, "depth_mode": depth_mode, "color": color},
        "image_rotate_deg": int(roll_deg) % 360,
        "R_cam_to_body": [[round(float(v), 10) for v in row] for row in R],
        "t_cam_to_body_m": [round(float(v), 6) for v in t],
        "tape": {
            "yaw_deg": float(yaw_deg), "pitch_deg": float(pitch_deg),
            "roll_deg": float(roll_deg),
            "forward_mm": float(forward_mm), "left_mm": float(left_mm),
            "height_mm": float(height_mm),
            "note": "角度用公式 R = Rz_body(yaw)·Ry_body(pitch)·C0·Rz_optical(roll)",
        },
    }
    if tape_extra:
        d["tape"].update(tape_extra)
    # 立刻自检一遍 —— 量错了现在就该炸，别等上车
    Rc, tc, rot, warn = check_calib(d, rotate=int(roll_deg) % 360)
    d["R_cam_to_body"] = [[round(float(v), 10) for v in row] for row in Rc]
    d["t_cam_to_body_m"] = [round(float(v), 6) for v in tc]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return d, warn


def _today():
    import datetime
    return datetime.date.today().isoformat()


# =====================================================================
# 命令行
# =====================================================================
def _report(d):
    """把标定翻译成人能核对的东西 —— 方向对不对一眼看得出。"""
    R, t, rot, warn = check_calib(d)
    yaw, pitch, roll = decompose_R(R)
    print(f"标定 {d.get('id','?')}   文件旋转 image_rotate_deg = {rot}°")
    print(f"  车体系：X前 / Y左 / Z上，原点 = 车体几何中心（带高度）")
    print()
    print("  ★ R_cam_to_body 的三个基（列 = 该轴在车体系里的方向）")
    names = [("① +X  图像『右』", 0), ("① +Y  图像『下』", 1), ("① +Z  光轴", 2)]
    for nm, j in names:
        v = R[:, j]
        print(f"    {nm:<18} → ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})   {_describe(v)}")
    print()
    print(f"  反解回角度： yaw {yaw:+.2f}°   pitch {pitch:+.2f}°   roll {roll:+.2f}°")
    print(f"    pitch > 0 = 光轴【向下俯视】； yaw > 0 = 光轴偏【左】")
    print()
    res = upright_residual(R, rot)
    print(f"  ★ 竖直判据 (R @ Rz({rot})ᵀ)[2,1] = {res:+.4f}   （必须 < 0）"
          f"  {'✅' if res < 0 else '❌'}")
    print(f"    把图像翻滚还原后，『图像里的下』指向 "
          f"{_describe((R @ Rz_optical(rot).T)[:, 1])}")
    print(f"    再乘一次 Rz 会变成 {upright_residual(R @ Rz_optical(rot), rot):+.4f}"
          f" → {'会被判据拒掉 ✅（说明判据真的在起作用）' if upright_residual(R @ Rz_optical(rot), rot) >= 0 else '⚠️ 判据区分不出来'}")
    print()
    print(f"  光心位置（车心指向深度光心）： "
          f"前 {t[0]*1000:+.1f} mm  左 {t[1]*1000:+.1f} mm  高 {t[2]*1000:+.1f} mm"
          f"   |t| = {np.linalg.norm(t):.4f} m")
    print(f"  det(R) = {np.linalg.det(R):.9f}    |RRᵀ−I|max = "
          f"{np.abs(R @ R.T - np.eye(3)).max():.2e}")
    if warn:
        print()
        for w in warn:
            print(f"  ⚠️  {w}")
    return 0


def _describe(v):
    """把一个单位向量说成人话（哪个方向占主导）。"""
    ax = ["前", "左", "上"]
    i = int(np.argmax(np.abs(v)))
    sgn = "+" if v[i] > 0 else "−"
    mag = abs(v[i])
    if mag > 0.999:
        return f"几乎正{sgn}{ax[i]}"
    if mag > 0.1:
        others = ", ".join(f"{ax[k]}{v[k]:+.2f}" for k in range(3) if k != i
                           and abs(v[k]) > 0.05)
        return f"主要在{sgn}{ax[i]}（{mag:.2f}）{('，还有 ' + others) if others else ''}"
    return "（没有主导方向 —— 这个数可能有问题）"


def main():
    ap = argparse.ArgumentParser(
        description="车体标定文件（相机 → 车体 的固定刚体变换）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("═══\n")[-1] if "═══" in __doc__ else None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="PATH", help="生成标定文件")
    g.add_argument("--check", metavar="PATH", help="校验标定文件")
    g.add_argument("--verify", metavar="PATH", help="打印可核对的报告")
    g.add_argument("--pitch-from-target", action="store_true",
                   help="按靶板测量算 pitch（配合 --L/--drop）")

    ap.add_argument("--L", type=float, default=None,
                    help="靶板到传感器的水平距离 mm（--pitch-from-target 用）")
    ap.add_argument("--drop", type=float, default=None,
                    help="靶心比传感器轴心【低】多少 mm（高了给负数）")

    ap.add_argument("--yaw", type=float, default=0.0, help="偏航角°（>0 偏左）")
    ap.add_argument("--pitch", type=float, default=None, help="俯仰角°（>0 下俯）")
    ap.add_argument("--roll", type=float, default=180.0,
                    help="绕光轴的图像翻滚°（倒装=180）")
    ap.add_argument("--forward", type=float, default=None, help="光心在车心【前】方 mm")
    ap.add_argument("--left", type=float, default=None, help="光心在车心【左】方 mm")
    ap.add_argument("--height", type=float, default=None, help="光心比车心【高】 mm")
    ap.add_argument("--id", default="", help="标定 id（写进串口头部，便于追溯）")
    ap.add_argument("--serial", default="", help="相机序列号（留档）")
    ap.add_argument("--rotate", type=int, default=None,
                    help="断言 image_rotate_deg 等于这个值（服务单元里那个 --rotate）")
    ap.add_argument("--fix", action="store_true",
                    help="--check 时允许 SVD 重新正交化")
    a = ap.parse_args()

    try:
        if a.pitch_from_target:
            if a.L is None or a.drop is None:
                ap.error("--pitch-from-target 需要同时给 --L 和 --drop")
            p = math.degrees(math.atan2(float(a.drop), float(a.L)))
            print(f"靶板正前方 L={a.L:.1f} mm，靶心比传感器轴心低 {a.drop:.1f} mm")
            print(f"  → pitch = atan2({a.drop:.1f}, {a.L:.1f}) = {p:+.3f}°"
                  f"（正 = 光轴下俯）")
            print(f"  生成时用： --pitch {p:.3f}")
            return 0

        if a.verify:
            with open(a.verify, encoding="utf-8") as f:
                d = json.load(f)
            return _report(d)

        if a.check:
            with open(a.check, encoding="utf-8") as f:
                d = json.load(f)
            R, t, rot, warn = check_calib(d, rotate=a.rotate, fix=a.fix)
            print(f"✅ {a.check} 合法")
            print(f"   image_rotate_deg = {rot}°   det(R) = {np.linalg.det(R):.9f}")
            print(f"   竖直判据 = {upright_residual(R, rot):+.4f}（<0 ✔）")
            print(f"   |t| = {np.linalg.norm(t):.4f} m")
            for w in warn:
                print(f"   ⚠️  {w}")
            if a.fix:
                d["R_cam_to_body"] = [[round(float(v), 10) for v in r] for r in R]
                d["t_cam_to_body_m"] = [round(float(v), 6) for v in t]
                with open(a.check, "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                print(f"   ↻ 已把正交化后的 R 写回 {a.check}")
            return 0

        # --write
        missing = [n for n, v in (("--pitch", a.pitch), ("--forward", a.forward),
                                  ("--left", a.left), ("--height", a.height))
                   if v is None]
        if missing:
            ap.error("--write 还需要：" + " ".join(missing)
                     + "（长度都用 mm，跟卷尺一致）")
        d, warn = write_body_calib(
            a.write, a.yaw, a.pitch, a.roll, a.forward, a.left, a.height,
            calib_id=a.id, serial=a.serial)
        print(f"✅ 已写 {a.write}")
        for w in warn:
            print(f"   ⚠️  {w}")
        print()
        return _report(d)

    except BodyCalibError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
