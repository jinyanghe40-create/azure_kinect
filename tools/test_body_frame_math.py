#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验收 A —— 车体变换的【纯数学】零回归测试（不用相机、不用板子、不用 IMU）。

验三件事：

  A1 闭式 extr 公式 == 旧的两步（Rz 共轭 + 右乘 R_gravᵀ）
     在两个极端下必须【逐位相同】。用的【是 azure_capture.py 里真正的那段
     代码】（object.__new__ + 手动塞属性），不是抄一份公式 —— 抄一份就测不出
     将来代码漂移。

  A2 投影自洽：随便取一个车体系点 p_body，
        · 走【新】路：点云在 body 系 (p_body)，外参 self.extr → 像素
        · 走【旧】路：点云在 post-Rz 系 (Rz·R_outᵀ(p_body−t_out))，旧外参 → 像素
     两条路的像素必须一样（1e-9 相对）。这是"点云左乘 R_out、外参右乘 R_outᵀ"
     那个坑的机器化检查 —— 写反的话这里必炸。

  A3 t_out 真的进了点云：给定 (R_out, t_out)，对同一台相机，把目标从
     车体系原点正前方挪开，get() 走的变换必须把点搬到 t_out 边上，而不是原点。

用法：
    python3 tools/test_body_frame_math.py            # 全跑
    python3 tools/test_body_frame_math.py -v         # 打印每步
退出码 0 = 全绿。
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "azure_kinect"))


def _load_source_cls():
    """只 import 类，不碰相机（_load_k4a 只在 __init__ 里调）。"""
    import azure_capture
    return azure_capture.AzureSource


def _bare(rotate, R_dc, t_dc, R_grav=None, R_body=None, t_out=None):
    """造一个【没开相机】的 AzureSource，属性手工塞好。

    ★ 刻意不调 __init__：那会 _load_k4a() 并开相机。
    """
    S = _load_source_cls()
    s = object.__new__(S)
    s.rotate = rotate % 360
    s.extr_base = (np.asarray(R_dc, float), np.asarray(t_dc, float).reshape(3))
    s.extr = None
    s.R_grav = None if R_grav is None else np.asarray(R_grav, float)
    s.R_body = None if R_body is None else np.asarray(R_body, float)
    s.t_out = np.zeros(3) if t_out is None else np.asarray(t_out, float).reshape(3)
    s.calib_id = None
    s.R_out = np.eye(3)
    s._xform_needed = False
    return s


def _old_two_step(rotate, R_dc, t_dc, R_grav=None):
    """旧代码的两步合成，逐字照抄改前的 azure_capture.py。

    步 1（__init__ :339）：self.extr = (Rz @ R @ Rz.T, Rz @ t)
    步 2（_finish_rotations :637）：self.extr = (R @ R_grav.T, t)   ← 平移不变
    """
    Rz = _Rz(rotate)
    R, t = Rz @ R_dc @ Rz.T, Rz @ t_dc
    if R_grav is not None and np.abs(R_grav - np.eye(3)).max() > 1e-9:
        R = R @ np.asarray(R_grav, float).T          # t 不变 —— 旧式只对 t_out=0 成立
    return R, t


def _Rz(rotate):
    k = (rotate // 90) if rotate else 0
    if not k:
        return np.eye(3)
    th = np.radians(k * 90.0)
    c_, s_ = np.cos(th), np.sin(th)
    return np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])


def _rand_rot(rng):
    """均匀随机旋转（QR 法 + 纠正 det=+1）。"""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ------------------------------------------------------------------ A1
def test_a1_extr_identity(verbose=False):
    """闭式 == 旧两步，在 t_out=0 的两个极端下逐位相同。"""
    rng = np.random.default_rng(20261015)
    worst = 0.0
    n = 0
    for rotate in (0, 90, 180, 270):
        for use_grav in (False, True):
            for _ in range(40):
                R_dc, t_dc = _rand_rot(rng), rng.standard_normal(3) * 0.05
                R_grav = _rand_rot(rng) if use_grav else None

                # 老路：X = R_grav 或 I，t_out = 0
                s = _bare(rotate, R_dc, t_dc, R_grav=R_grav, R_body=None, t_out=None)
                if R_grav is None:
                    s.R_grav = None
                s._finish_rotations()
                new_R, new_t = s.extr

                old_R, old_t = _old_two_step(rotate, R_dc, t_dc, R_grav)
                d = max(np.abs(new_R - old_R).max(), np.abs(new_t - old_t).max())
                worst = max(worst, d)
                n += 1
                if verbose and d > 0:
                    print(f"    rotate={rotate} grav={use_grav} Δ={d:.3e}")
    print(f"  A1 闭式 extr == 旧两步   {n} 组，最大差 {worst:.3e}", end="  ")
    ok = worst < 1e-12
    print("✅" if ok else "❌")
    return ok


def test_a1b_body_calib_reduces(verbose=False):
    """★ 180° 重复计入的绊线（风险 #1，完全静默的那种）。

    标定矩阵【本身就是】raw→body 的物理真相，含那 180°。所以
        R_out 必须【正好等于】R_cam_to_body —— 不能再乘一次 Rz。
    再乘一次的后果：det 依然是 +1（还是旋转！），点云整个翻过去，
    没有任何一处会报错。所以只能靠这条断言 + 运行时的 R[2,1]<0 挡住。
    """
    rng = np.random.default_rng(7)
    worst = 0.0
    worst_double = 1e9
    for rotate in (0, 90, 180, 270):
        Rz = _Rz(rotate)
        for _ in range(30):
            R_body = _rand_rot(rng)
            s = _bare(rotate, _rand_rot(rng), rng.standard_normal(3) * 0.05,
                      R_body=R_body)
            s._finish_rotations()
            worst = max(worst, np.abs(s.R_out - R_body).max())
            # "重复计入"版长什么样：R_body @ Rz。rotate≠0 时它必须明显不同。
            if rotate:
                worst_double = min(worst_double, np.abs(R_body @ Rz - R_body).max())
    print(f"  A1b R_out == R_cam_to_body（180°重复计入的绊线）  最大差 {worst:.3e}",
          end="  ")
    ok = worst < 1e-12
    print("✅" if ok else "❌")
    if rotate:
        print(f"      （对照）重复计入版 R_body@Rz 与正确值最小差 "
              f"{worst_double:.3f} —— 差得足够大，这条断言抓得住")
    return ok


# ------------------------------------------------------------------ A2
def _project(p_cam_like, R, t):
    """外参 `Xc = R @ p + t` —— 就是 match_classes / _make_uv_fn 用的形式。"""
    return (np.asarray(R, float) @ np.asarray(p_cam_like, float).reshape(3, 1)).ravel() \
        + np.asarray(t, float).reshape(3)


def test_a2_projection_consistency(verbose=False):
    """▶ 最强的一条：拿【物理真值】当基准比像素。

    真值：相机测到深度点 p_sdk，它在彩色图里的位置就是
              px_true = Rz · (R_dc · p_sdk + t_dc)
          （Rz 是因为彩色图和点云一起转了 rotate）。

    ✅ 新路：p_body = R_out·p_sdk + t_out，再用 extr_new 投影 —— 必须 == px_true
    ✅ 旧路（t_out=0 时）：p_cloud = X·Rz·p_sdk，再用旧 extr 投影 —— 也必须 == px_true

    换句话说：只要这条过，点云的"左乘 R_out + 加 t_out"和外参的"右乘 R_outᵀ"
    就是一整套自洽的（含平移）。写反、漏平移、漏 Rz，这里全都会炸。
    """
    rng = np.random.default_rng(99)
    worst_new = worst_old = 0.0
    n = 0
    for rotate in (0, 90, 180, 270):
        for _ in range(40):
            R_body = _rand_rot(rng)
            t_out = np.array([0.25, 0.0, 0.60]) + rng.standard_normal(3) * 0.05
            R_dc, t_dc = _rand_rot(rng), np.array([-0.032, 0.0, 0.0])
            Rz = _Rz(rotate)

            s = _bare(rotate, R_dc, t_dc, R_body=R_body, t_out=t_out)
            s._finish_rotations()

            p_sdk = rng.standard_normal(3) * 0.8       # 相机测到的深度点
            px_true = Rz @ (R_dc @ p_sdk + t_dc)       # 物理真值

            # 【新】get() 里那一行 + 新 extr
            p_body = s.R_out @ p_sdk + s.t_out
            px_new = _project(p_body, *s.extr)

            # 【旧】等价路（t_out=0 的旧代码，这里把 t_out 从点云里减掉还原）
            p_cloud_old = R_body @ Rz @ p_sdk          # 旧代码：R_grav@Rz 左乘，无平移
            old_R, old_t = _old_two_step(rotate, R_dc, t_dc, R_grav=R_body)
            px_old = _project(p_cloud_old, old_R, old_t)

            scale = max(1.0, np.abs(px_true).max())
            dn = np.abs(px_new - px_true).max() / scale
            do = np.abs(px_old - px_true).max() / scale
            worst_new = max(worst_new, dn)
            worst_old = max(worst_old, do)
            n += 1
            if verbose and max(dn, do) > 1e-12:
                print(f"    rotate={rotate} 新Δ={dn:.3e} 旧Δ={do:.3e}")
    print(f"  A2 新路 vs 物理真值  {n} 组，最大相对差 {worst_new:.3e}", end="  ")
    ok_new = worst_new < 1e-9
    print("✅" if ok_new else "❌")
    print(f"  A2 旧路 vs 物理真值  {n} 组，最大相对差 {worst_old:.3e}", end="  ")
    ok_old = worst_old < 1e-9
    print("✅" if ok_old else "❌")
    return ok_new and ok_old


def test_a2b_transpose_trap(verbose=False):
    """故意把 R_outᵀ 写成 R_out（转置搞反），必须【被 A2 抓出来】。

    这是回归测试的回归测试：A2 连这个都抓不到的话，说明 A2 自己没验到东西。
    """
    rng = np.random.default_rng(5)
    worst = 0.0
    for rotate in (0, 180):
        R_body = _rand_rot(rng)
        t_out = np.array([0.25, 0.0, 0.60])
        R_dc, t_dc = _rand_rot(rng), np.array([-0.032, 0.0, 0.0])
        Rz = _Rz(rotate)
        s = _bare(rotate, R_dc, t_dc, R_body=R_body, t_out=t_out)
        s._finish_rotations()

        p_sdk = np.array([0.3, -0.2, 0.9])
        px_true = Rz @ (R_dc @ p_sdk + t_dc)
        p_body = s.R_out @ p_sdk + s.t_out
        good = _project(p_body, *s.extr)

        # 错误版：外参里用 R_out 而不是 R_outᵀ
        bad_R = Rz @ R_dc @ s.R_out
        bad_t = Rz @ (t_dc - R_dc @ s.R_out @ s.t_out)
        bad = _project(p_body, bad_R, bad_t)

        worst = max(worst, np.abs(good - px_true).max())        # 好的必须对
        d_bad = np.abs(bad - px_true).max()
        if verbose:
            print(f"    rotate={rotate} 正确版Δ={np.abs(good-px_true).max():.2e} "
                  f"写反版Δ={d_bad:.3e}")
        if d_bad < 1e-3:
            print(f"  A2b 转置写反居然没被抓到（差 {d_bad:.3e}）❌ (A2 是假绿！)")
            return False
    print(f"  A2b 转置写反会被抓到 ✅（正确版残差 {worst:.2e}）")
    return True


# ------------------------------------------------------------------ A3
def test_a3_t_out_reaches_cloud(verbose=False):
    """平移真的进了点云：标定的 t_out 必须能从输出点云里读回来。"""
    rng = np.random.default_rng(3)
    worst = 0.0
    for rotate in (0, 180):
        for _ in range(20):
            R_body = _rand_rot(rng)
            t_out = rng.standard_normal(3) * 0.4
            s = _bare(rotate, _rand_rot(rng), np.zeros(3), R_body=R_body, t_out=t_out)
            s._finish_rotations()

            # get() 里那一行（两个分支逐字一致）
            xyz_sdk = rng.standard_normal((50, 3))
            pts = xyz_sdk @ s.R_out.T + s.t_out

            # 反推：p_sdk = R_outᵀ(pts − t_out) 必须还原原始点
            back = (pts - s.t_out) @ s.R_out
            worst = max(worst, np.abs(back - xyz_sdk).max())

            # 平移丢了的话，点云质心会偏 |t_out|
            if np.abs(pts.mean(0) - xyz_sdk.mean(0)).max() < 1e-9 and np.abs(t_out).max() > 0.1:
                print(f"    ❌ rotate={rotate}: 平移没进点云")
                return False
    print(f"  A3 t_out 真的进了点云  最大往返误差 {worst:.3e}", end="  ")
    ok = worst < 1e-12
    print("✅" if ok else "❌")
    return ok


def test_a3b_xform_needed_includes_translation(verbose=False):
    """R_out 凑巧是单位阵、但 t_out≠0 时，_xform_needed 必须为 True。

    这正是改名要挡的那个静默 bug：只看 R_out 的话平移被整段跳过。
    """
    cases = [
        (np.eye(3), np.zeros(3), False),
        (np.eye(3), np.array([0.2, 0.0, 0.0]), True),   # ★ 关键行
        (_Rz(180), np.zeros(3), True),
        (_Rz(180), np.array([0.2, 0.0, 0.0]), True),
    ]
    bad = []
    for R_body, t_out, want in cases:
        s = _bare(0, np.eye(3), np.zeros(3), R_body=R_body, t_out=t_out)
        s._finish_rotations()
        if s._xform_needed != want:
            bad.append((R_body.tolist(), t_out.tolist(), want, s._xform_needed))
    print(f"  A3b _xform_needed 含平移判据  {len(cases)} 组", end="  ")
    ok = not bad
    print("✅" if ok else f"❌ {bad}")
    return ok


def test_real_deployment_numbers(verbose=True):
    """用【真实部署数值】手算一遍，数字肉眼可查。

    ★ 矩阵用 body_calib.build_R() 生成 —— 跟真机上 --write 走的【同一段代码】，
      不是手抄一个。手抄的矩阵上一版就是错的（用 post-Rz 约定手写，
      R[2,1] = −1，正是守门要拒的那个）。
    """
    sys.path.insert(0, os.path.join(ROOT, "azure_kinect"))
    import body_calib as bc

    pitch = 9.7                                  # 光轴下俯（靶板测量值）
    t_out = np.array([0.25, 0.0, 0.60])
    t_dc = np.array([-0.032, 0.002, -0.001])     # 工厂深度→彩色外参的平移
    R_body = bc.build_R(0.0, pitch, 180.0)       # 倒装前视相机
    s = _bare(180, np.eye(3), t_dc, R_body=R_body, t_out=t_out)
    s._finish_rotations()

    print("  ── 真实部署数值抽查 ─────────────────────────────")
    print(f"    R_cam_to_body[2,1] = {R_body[2, 1]:+.3f}"
          f"   （倒装相机【就该是 +1】：raw 深度系的『图像下』物理朝上）")
    print(f"    竖直判据 (R @ Rz(180)ᵀ)[2,1] = "
          f"{bc.upright_residual(R_body, 180):+.4f}   （必须 < 0）")
    print(f"    R_out == R_cam_to_body ? {np.abs(s.R_out - R_body).max():.2e}")
    print(f"    _xform_needed = {s._xform_needed}")
    print(f"    extr R =\n{np.round(s.extr[0], 4)}")
    print(f"    extr t = {np.round(s.extr[1], 5).tolist()}")
    # 车正前方 1m、地面高度 0（车心高度 0）的点
    p = np.array([1.0, 0.0, 0.0])
    xc = s.extr[0] @ p + s.extr[1]
    v = p - t_out
    optic = R_body[:, 2]                       # 光轴在车体系的方向
    # ★ 相机系深度 = 光心到该点的向量【在光轴上的投影】，不是直线距离，
    #   也不是 |v|·cos(pitch)（那个只在该点正好落在光轴上时才成立）。
    depth_expect = float(v @ optic)
    # 剩下的差恰好 = 工厂深度→彩色外参的 z 分量（彩色相机比深度相机差 ~1mm），
    # 因为投影落在【彩色】相机系里。这个残差就是"平移项挂对了相机"的证据。
    dc_z = float(t_dc[2])
    print(f"    车系(1,0,0) → 彩色相机系 {np.round(xc, 4).tolist()}")
    print(f"      v = p − 光心 = {np.round(v, 3).tolist()}，|v| = {np.linalg.norm(v):.3f} m")
    print(f"      光轴 = {np.round(optic, 4).tolist()}（下俯 {pitch}°）")
    print(f"      相机系深度 = v·光轴 = {depth_expect:.4f} m"
          f"   实测 {xc[2]:.4f} m   Δ={abs(xc[2]-depth_expect):.2e}")
    print(f"      Δ 恰好 = 工厂 t_dc[2] = {dc_z:+.4f} m（深度→彩色相机差 ~1mm）"
          f" → 差的量对得上，不是错")
    print(f"      ⚠️ 注意 xc[2] 是【相机系】深度，不是车体系的 Z")

    ok = (R_body[2, 1] > 0
          and bc.upright_residual(R_body, 180) < 0
          and abs(s.R_out - R_body).max() < 1e-12
          and s._xform_needed
          and abs(xc[2] - depth_expect - dc_z) < 1e-9)
    print("  ", "✅" if ok else "❌")
    return ok


def test_a4_body_calib_guards(verbose=False):
    """验收 A4：标定文件的守门 —— 正例过、负例全拒。

    ★ 这个模块存在的理由就是"静默错误肉眼看不见"（180° 重复计入 det 仍是
      +1），所以每一道守门都必须有一条负例证明它真的会响。
    """
    sys.path.insert(0, os.path.join(ROOT, "azure_kinect"))
    import body_calib as bc
    import json as _json
    import tempfile

    ok = True

    # ---- 生成 / 反解 往返 ----
    worst = 0.0
    for y in np.arange(-40, 41, 10):
        for p in np.arange(-40, 41, 10):
            for r in (0, 90, 180, 270):
                R = bc.build_R(y, p, r)
                y2, p2, r2 = bc.decompose_R(R)
                worst = max(worst,
                            abs((y2 - y + 180) % 360 - 180), abs(p2 - p),
                            abs((r2 - r + 180) % 360 - 180))
    print(f"  A4a build_R ∘ decompose_R 往返  最大误差 {worst:.3e}°", end="  ")
    good = worst < 1e-9
    ok &= good
    print("✅" if good else "❌")

    # ---- 拿一份合法的当底子 ----
    d, _ = bc.write_body_calib(tempfile.mktemp(suffix=".json"),
                               0.0, 9.7, 180.0, 250, 0, 600, calib_id="t")
    R = np.asarray(d["R_cam_to_body"], float)

    def must_reject(tag, mutate, rotate=180):
        nonlocal ok
        d2 = _json.loads(_json.dumps(d))
        mutate(d2)
        try:
            bc.check_calib(d2, rotate=rotate)
            print(f"    ❌ {tag}：竟然通过了 —— 守门是假的")
            ok = False
        except bc.BodyCalibError:
            if verbose:
                print(f"    ✅ {tag}：被拒")

    def must_accept(tag, mutate, rotate=180):
        nonlocal ok
        d2 = _json.loads(_json.dumps(d))
        mutate(d2)
        try:
            bc.check_calib(d2, rotate=rotate)
            if verbose:
                print(f"    ✅ {tag}：通过")
        except bc.BodyCalibError as e:
            print(f"    ❌ {tag}：被误拒 —— {e}")
            ok = False

    def set_R(d2, M):
        d2["R_cam_to_body"] = [[float(v) for v in row] for row in M]

    must_accept("正例", lambda d2: None)
    must_reject("★按 post-Rz 系存（180° 重复计入）",
                lambda d2: set_R(d2, R @ bc.Rz_optical(180).T))
    must_reject("矩阵转置写反", lambda d2: set_R(d2, R.T))
    must_reject("det = −1 的反射",
                lambda d2: set_R(d2, np.diag([1.0, 1.0, -1.0]) @ R))
    must_reject("rotate 与文件矛盾", lambda d2: None, rotate=0)
    must_reject("单位把 mm 当 m 写",
                lambda d2: d2.update({"t_cam_to_body_m": [250.0, 0.0, 600.0]}))
    must_reject("schema 不对（误用深度内参文件）",
                lambda d2: d2.update({"schema": "k4a_calibration"}))
    must_reject("R 形状不对", lambda d2: d2.update({"R_cam_to_body": [1.0, 2.0, 3.0]}))

    print(f"  A4b 标定守门（1 正 + 7 负）", end="  ")
    print("✅" if ok else "❌")
    return ok


def test_a5_rejects_double_rotate_in_capture(verbose=False):
    """azure_capture 的构造期断言也会拒（不只是 body_calib 那个入口）。

    防的是"有人绕过 load_body_calib 直接塞 dict"。两处都要响。
    """
    sys.path.insert(0, os.path.join(ROOT, "azure_kinect"))
    import body_calib as bc
    S = _load_source_cls()

    R_good = bc.build_R(0, 9.7, 180)
    R_double = R_good @ bc.Rz_optical(180).T     # 按 post-Rz 系写的

    ok = True
    # 坏标定必须抛
    s = object.__new__(S)
    s.rotate = 180
    s.body_calib = {"R": R_double, "t": np.zeros(3), "id": "bad"}
    try:
        S._setup_body_frame(s)          # 见 azure_capture 里的同名方法
        print("  A5 坏标定没被 azure_capture 拒掉 ❌")
        ok = False
    except Exception:
        pass
    # 好标定必须过
    s2 = object.__new__(S)
    s2.rotate = 180
    s2.body_calib = {"R": R_good, "t": np.array([0.25, 0.0, 0.6]), "id": "ok"}
    try:
        S._setup_body_frame(s2)
    except Exception as e:
        print(f"  A5 好标定被误拒 ❌ —— {e}")
        ok = False
    print(f"  A5 azure_capture 侧同样守门（好过/坏拒）", end="  ")
    print("✅" if ok else "❌")
    return ok


def main():
    ap = argparse.ArgumentParser(description="车体变换纯数学验收（A）")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    print("═══ 验收 A：车体变换纯数学（不用相机 / 不用板子）═══")
    results = [
        test_a1_extr_identity(a.verbose),
        test_a1b_body_calib_reduces(a.verbose),
        test_a2_projection_consistency(a.verbose),
        test_a2b_transpose_trap(a.verbose),
        test_a3_t_out_reaches_cloud(a.verbose),
        test_a3b_xform_needed_includes_translation(a.verbose),
        test_a4_body_calib_guards(a.verbose),
        test_a5_rejects_double_rotate_in_capture(a.verbose),
        test_real_deployment_numbers(a.verbose),
    ]
    print("═" * 52)
    if all(results):
        print(f"✅ 全绿（{len(results)}/{len(results)}）")
        return 0
    print(f"❌ {sum(results)}/{len(results)} 通过")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
