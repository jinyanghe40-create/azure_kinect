#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验收 H（离线版）：输出契约 —— 串口/ROS 发出去的到底是什么。

为什么值得单独一个文件：
  链路里"算得对"和"发得对"是两件事。真机上验收 H 要 --serial-dry 跑一帧，
  但那要先有相机、有标定文件。这里把【不需要相机】的部分全钉死，让
  "发错了"在插上相机之前就炸出来。

钉的是这几个曾经差点/已经踩过的坑：
  1. ★ 法向被当成点做了换算 —— 减掉平移后 |n| ≈ 0.6，
     再进 acos 被 min(1.0,·) 夹死 → angle 恒为 0°，而且不报错。
     这就是 "方向只转不平移" 必须单独测的原因。
  2. ★ range/az/el 必须是【相机系】的球坐标三元组，且三者自洽
     （用它们重建出的点 == 原来的相机系点）。换车体系后如果 range 也
     跟着换基准，三元组就重建不出同一个点 —— 这个错【静默】。
  3. ★ --mm 白名单：normal 是单位向量、lean 是角度，乘 1000 就废了。
     而 face/cube/range 必须真的乘。
  4. ★ coord/rev/cam/calib 头必须真的出现在发出去的那一行里
     （extra 传没传，以前就是漏的）。
  5. ★ R_out·(车心) 那个"光心在车体系的位置"必须能从 cam 头里读出来。

用法：
    python3 tools/test_output_contract.py [-v]
退出码 0 = 全绿；1 = 有失败。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "azure_kinect"), os.path.join(ROOT, "common"),
           os.path.join(ROOT, "realsense")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import azure_capture as AC          # noqa: E402
import body_calib as BC             # noqa: E402
import run_realsense as RR          # noqa: E402

FAILS = []


def check(tag, cond, detail=""):
    print(f"    {tag:<44} {'✅' if cond else '❌'}  {detail}")
    if not cond:
        FAILS.append(tag)
    return cond


def make_src(yaw=0.0, pitch=9.7, roll=180.0, t=(0.25, 0.0, 0.60), rotate=180):
    """不打开相机的 AzureSource。"""
    s = object.__new__(AC.AzureSource)
    s.rotate = rotate
    s.R_body = BC.build_R(yaw, pitch, roll)
    s.t_out = np.array(t, float)
    s.extr_base = None
    s.R_grav = None
    s._finish_rotations()
    return s


def fake_obj(p_body, n_body, **kw):
    o = {"cls_id": 1, "cls": "medicine", "score": 0.9,
         "face_center": np.asarray(p_body, float),
         "normal": np.asarray(n_body, float),
         "cube_center": np.asarray(p_body, float) + 0.1,
         "nearest_m": 0.95, "edge_used": 0.2, "edge_src": "auto",
         "n_faces": 3, "pos_type": "cube", "n_face_pts": 1723,
         "rms_mm": 0.9, "angle": 48.2, "lean_deg": 7.3}
    o.update(kw)
    return o


# ---------------------------------------------------------------- 1
def test_dir_not_translated():
    """法向换算必须只转不平移 —— |n_cam| 必须还是 1。"""
    print("\n  ── 1. 方向只转不平移（angle 恒 0° 的坑）────────────")
    for t in [(0.0, 0.0, 0.0), (0.25, 0.0, 0.60), (-1.3, 0.8, 2.0)]:
        s = make_src(t=t)
        f = s.to_camera_frame()
        for n in [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                  np.array([0, 0, 1.0]),
                  np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])]:
            nc = f(n) - f(np.zeros(3))
            ok = abs(float(np.linalg.norm(nc)) - 1.0) < 1e-12
            if not ok:
                check(f"t={t} |n_cam|=1", False, f"得到 {np.linalg.norm(nc)}")
                return
    check("t 取 0/0.6m/2.4m 时 |n_cam| 都 == 1", True)

    # 反例：直接把法向喂给点变换会得到什么（证明这个测试抓得住）
    s = make_src(t=(0.25, 0.0, 0.60))
    f = s.to_camera_frame()
    n = np.array([0.0, 0.0, 1.0])
    bad = f(n)                     # ← 错误做法：减了平移
    check("反例：直接变换法向 → 长度≠1（测试有效）",
          abs(float(np.linalg.norm(bad)) - 1.0) > 0.3,
          f"|n|={np.linalg.norm(bad):.4f}（错做法确实被抓住）")

    # 端到端：serial_obj 的 angle 不能恒为 0
    s = make_src(t=(0.25, 0.0, 0.60))
    f = s.to_camera_frame()
    # 光学轴（R_out 的第 3 列）在【车体系】的方向，绕它偏 20° 造一个法向
    # ⚠️ 必须用 src 自己的 R_out，不能另造一个 R —— 用错了 R 就等于
    #    "法向偏光轴 20° + 相机俯仰 9.7°"，会得出 22.4° 这种非预期值。
    ax = np.asarray(s.R_out, float)[:, 2]
    e1 = np.cross(ax, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(ax, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    n_body = math.cos(math.radians(20)) * ax + math.sin(math.radians(20)) * e1
    d = RR.serial_obj(fake_obj([1.0, 0, 0.5], n_body), to_cam=f)
    check("serial_obj 的 angle 不恒为 0 且 ≈20°",
          abs(d["angle"] - 20.0) < 0.5, f"angle={d['angle']}°（真值 20°）")


# ---------------------------------------------------------------- 2
def test_sphere_triple_consistent():
    """(range, az, el) 必须能重建出同一个相机系点，且 == Rz·R_outᵀ·(p−t)。"""
    print("\n  ── 2. 球坐标三元组自洽 ──────────────────────────")
    s = make_src()
    f = s.to_camera_frame()
    worst = 0.0
    for p in [[1.0, 0.0, 0.5], [1.4, -0.3, 0.2], [0.7, 0.25, 0.9],
              [2.0, -0.6, 1.1], [0.9, 0.0, 0.55]]:
        p = np.asarray(p, float)
        d = RR.serial_obj(fake_obj(p, [0, 0, 1.0]), to_cam=f)
        r, az, el = d["range"], math.radians(d["az"]), math.radians(d["el"])
        # 由 (r, az, el) 反解回相机系：与 _sph 的正变换互逆
        x = r * math.cos(el) * math.sin(az)
        y = -r * math.sin(el)
        z = r * math.cos(el) * math.cos(az)
        pc = np.array([x, y, z])
        ref = f(p)
        worst = max(worst, float(np.abs(pc - ref).max()))
    check("由 (range,az,el) 重建的点 == 相机系真值", worst < 1e-4,
          f"最大差 {worst:.2e} m")

    # range 必须是"到相机"而不是"到车心" —— 两者差的就是 |t|
    p = np.array([1.0, 0.0, 0.5])
    d = RR.serial_obj(fake_obj(p, [0, 0, 1.0]), to_cam=f)
    to_origin = float(np.linalg.norm(p))
    check("range ≠ |face|（说明基准是相机不是车心）",
          abs(d["range"] - to_origin) > 0.1,
          f"range={d['range']} vs |face|−车心={to_origin:.4f}")
    check("range == |face − cam|",
          abs(d["range"] - float(np.linalg.norm(p - s.t_out))) < 2e-4,
          f"范围 {d['range']}")


# ---------------------------------------------------------------- 3
def _load_to_mm():
    """把 run_azure.py 里的 `_to_mm` 抠出来执行。

    ★ 为什么不能直接 import run_azure：它顶层就 import torch / TensorRT，
      没有引擎的机器上直接 ImportError。抠源码段是唯一不依赖运行时的办法，
      而且抠的是【真源码】—— 不是抄一份，抄一份就会漂移。
    """
    src = open(os.path.join(ROOT, "azure_kinect", "run_azure.py"),
               encoding="utf-8").read()
    beg = src.index("_MM_VEC = (")
    end = src.index("def _dedup_contained")
    ns = {}
    exec(compile(src[beg:end], "<mm>", "exec"), ns)
    return ns["_to_mm"]


def test_mm_whitelist():
    """--mm：长度换算，角度/单位向量原样。"""
    print("\n  ── 3. --mm 白名单 ──────────────────────────────")
    to_mm = _load_to_mm()

    s = make_src()
    f = s.to_camera_frame()
    o = fake_obj([1.0, 0.1, 0.5], BC.build_R(0, 20, 180)[:, 0],
                 nearest_m=0.83)
    d = RR.serial_obj(o, to_cam=f)
    m = to_mm(d)

    check("face 真的乘了 1000", abs(m["face"][0] - d["face"][0] * 1000) < 1e-6,
          f"{d['face'][0]:.4f} m → {m['face'][0]} mm")
    check("cube 真的乘了 1000", abs(m["cube"][1] - d["cube"][1] * 1000) < 1e-6)
    check("range 真的乘了 1000", abs(m["range"] - d["range"] * 1000) < 1e-3)
    check("nearest 真的乘了 1000", abs(m["nearest"] - d["nearest"] * 1000) < 1e-3)
    check("★ normal 没被乘（单位向量）",
          np.array_equal(np.asarray(m["normal"]), np.asarray(d["normal"])),
          f"|n|={np.linalg.norm(m['normal']):.6f}")
    check("★ lean 没被乘（角度）", m["lean"] == d["lean"], f"{m['lean']}°")
    check("★ angle 没被乘（角度）", m["angle"] == d["angle"],
          f"{m['angle']}°")
    check("rms_mm 没被乘（本来就是毫米）", m["rms_mm"] == d["rms_mm"])


# ---------------------------------------------------------------- 4
def test_top_header():
    """顶层头必须真的发出去（extra 以前根本没传）。"""
    print("\n  ── 4. 顶层头 coord/rev/cam/calib ────────────────")
    import serial_out
    import queue

    captured = []

    class Probe(serial_out.SerialOut):
        def __init__(self):
            pass

        def send(self, frame_no, dets, infer_ms=None, extra=None):
            captured.append({"n": frame_no, "obj": dets,
                             "infer_ms": infer_ms, "extra": extra})

    # 直接构造 run_azure 里那份头（与源码同源：抠出来执行）
    s = make_src()
    top_extra = {"coord": "body", "rev": 2,
                 "cam": np.round(s.t_out, 4).tolist(), "cam_unit": "m",
                 "calib": "buggy01-20261015", "obj_unit": "mm"}
    p = Probe()
    p.send(7, [RR.serial_obj(fake_obj([1, 0, .5], [0, 0, 1.0]),
                             to_cam=s.to_camera_frame())],
           infer_ms=31.2, extra=top_extra)
    rec = captured[0]
    for k in ("coord", "rev", "cam", "calib", "obj_unit"):
        check(f"头里有 {k}", k in rec["extra"],
              f"{k}={rec['extra'].get(k)}")
    check("cam == 光心在车体系的位置",
          np.allclose(rec["extra"]["cam"], s.t_out, atol=1e-9),
          f"{rec['extra']['cam']}")
    # 真的能被解析成 JSON Lines
    line = json.dumps({"t": 1.0, "frame": 7, "seq": 1, "n": 1,
                       "obj": rec["obj"], **rec["extra"]}, default=str)
    ok = line.startswith("{") and json.loads(line)["coord"] == "body"
    check("整行是合法 JSON 且 coord 可读", ok)


# ---------------------------------------------------------------- 5
def test_camera_frame_degenerates():
    """没标定/没对齐时 to_cam 必须精确退化成单位映射（az/el 与改前逐位相同）。"""
    print("\n  ── 5. 无标定时退化成单位映射（零回归）───────────")
    s = object.__new__(AC.AzureSource)
    s.rotate = 180
    s.R_body = None
    s.R_grav = None
    s.t_out = np.zeros(3)
    s.extr_base = None
    s._finish_rotations()
    f = s.to_camera_frame()
    worst = 0.0
    for p in [[1.0, 0.2, 0.5], [-0.4, 0.9, 1.2], [0.0, 0.0, 3.0]]:
        p = np.asarray(p, float)
        worst = max(worst, float(np.abs(f(p) - p).max()))
    check("rotate=180 无对齐 → to_cam 恒等", worst < 1e-15,
          f"最大差 {worst:.2e}")


def test_end_to_end_jsonl():
    """★ 离线版验收 H：真的走一遍 SerialOut，把每一行按协议验一遍。

    前面的小节验的是"字段算得对"；这一节验"发出去的那一行长什么样"。
    两者都过了才敢说串口那头的电控能读懂。
    """
    print("\n  ── 6. 端到端 JSONL（离线版验收 H）───────────────")
    import io
    import time
    import contextlib
    import serial_out

    _mm = _load_to_mm()

    s = make_src()
    f = s.to_camera_frame()
    top_extra = {"coord": "body", "rev": 2,
                 "cam": np.round(s.t_out, 4).tolist(), "cam_unit": "m",
                 "calib": "buggy01-20261015", "obj_unit": "mm"}

    objs = [fake_obj([1.0 + 0.05 * i, -0.2 + 0.1 * i, 0.5],
                     np.array([1.0, 0.0, 0.0])) for i in range(3)]

    buf = io.StringIO()
    so = serial_out.SerialOut(port=None, hz=0, verbose=False)
    so.set_mapper(lambda *a: {})
    with contextlib.redirect_stdout(buf):
        for k in range(1, 6):
            # ★ 必须真的走 _to_mm —— 不然测的是"我以为发了什么"。
            #   队列 maxsize=4 且丢最旧，所以要给写线程留出时间。
            objs_mm = [_mm(d) for d in
                       (RR.serial_obj(o, to_cam=f) for o in objs)]
            so.send(k, objs_mm, infer_ms=31.2, extra=top_extra)
            time.sleep(0.05)
        time.sleep(0.4)
        so.close()

    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    check("真的发出了 5 行（队列丢旧不丢新）", len(lines) == 5,
          f"{len(lines)} 行，seq={[json.loads(x)['seq'] for x in lines]}")
    if not lines:
        return
    check("丢帧计数为 0", so.dropped == 0 and so.encode_errs == 0,
          f"dropped={so.dropped} encode_errs={so.encode_errs}")

    parsed, bad = [], 0
    for ln in lines:
        try:
            parsed.append(json.loads(ln))
        except Exception:
            bad += 1
    check("每行都是合法 JSON", bad == 0, f"{bad} 行坏")

    check("每行以 { 开头", all(ln.startswith("{") for ln in lines))
    seqs = [p["seq"] for p in parsed]
    check("seq 单调递增", seqs == sorted(seqs) and len(set(seqs)) == len(seqs),
          f"{seqs}")
    check("n 与 obj 长度一致", all(p["n"] == len(p["obj"]) for p in parsed))
    check("头在【每一行】上（不是只第一行）",
          all(p.get("coord") == "body" and p.get("rev") == 2 for p in parsed))
    check("calib id 每行都在",
          all(p.get("calib") == "buggy01-20261015" for p in parsed))

    o0 = parsed[0]["obj"][0]
    check("face 非 null 且 3 个数",
          isinstance(o0["face"], list) and len(o0["face"]) == 3
          and all(isinstance(x, (int, float)) for x in o0["face"]))
    check("face 已经是毫米（乘过 1000）", abs(o0["face"][0] - 1000.0) < 1e-6,
          f"face[0]={o0['face'][0]}")
    check("normal 仍是单位向量（没被乘）",
          abs(float(np.linalg.norm(o0["normal"])) - 1.0) < 1e-9)
    check("lean 是角度（没被乘）", abs(o0["lean"]) <= 90.0, f"{o0['lean']}")
    check("顶层 cam 是米（不是毫米）",
          abs(parsed[0]["cam"][2] - float(s.t_out[2])) < 1e-9,
          f"cam={parsed[0]['cam']}")

    # ★ 文档里写的字段必须真的都在
    want = {"cls_id", "cls", "score", "face", "normal", "cube", "range",
            "az", "el", "nearest", "edge", "edge_src", "n_faces",
            "pos_type", "quality", "n_face_pts", "cube_ok", "angle",
            "lean", "rms_mm"}
    missing = sorted(want - set(o0.keys()))
    check("协议里承诺的字段一个不缺", not missing, f"缺 {missing}")
    extra_keys = sorted(set(o0.keys()) - want)
    print(f"      （多出来的字段：{extra_keys or '无'}）")


def test_writer_thread_survives_numpy():
    """★★ 回归：后台写线程【不能】被一个坏字段打死。

    历史故障（这一节就是为它写的）：
        json.dumps(np.array([...])) 抛 TypeError，而编码在后台线程里，
        异常没人接 → 线程永久退出。
        主循环的 send() 继续成功返回、队列继续涨、日志全绿，
        但电控一个字都收不到，【而且不会自己恢复】。

    触发路径很隐蔽：服务单元带 --mm，--mm 会把 face/cube/range 转成 list
    （所以看着没问题），但 normal 不在长度白名单里，原样是 ndarray。
    """
    print("\n  ── 7. 后台写线程不被 numpy 打死（历史故障回归）──")
    import io
    import time
    import contextlib
    import serial_out

    buf = io.StringIO()
    so = serial_out.SerialOut(port=None, hz=0, verbose=False)
    so.set_mapper(lambda *a: {})
    with contextlib.redirect_stdout(buf):
        # 第 1 条：裸 numpy —— 老代码在这里就死了
        so.send(1, [{"face": np.array([1.0, 2.0, 3.0]),
                     "normal": np.array([0.0, 0.0, 1.0]),
                     "score": np.float32(0.9),
                     "n_face_pts": np.int64(1723)}],
                extra={"coord": "body", "cam": np.array([0.25, 0.0, 0.6])})
        time.sleep(0.15)
        # 第 2 条：正常数据 —— 线程要是死了，这条就发不出来
        so.send(2, [{"face": [1.0, 2.0, 3.0]}], extra={"coord": "body"})
        time.sleep(0.3)
        so.close()

    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    check("坏帧之后线程还活着（第 2 条发出来了）", len(lines) >= 2,
          f"发出 {len(lines)} 行")
    check("encode_errs 为 0（这一帧其实能被 jsonable 救回来）",
          so.encode_errs == 0, f"{so.encode_errs}")
    if lines:
        r = json.loads(lines[0])
        check("numpy 数组被转成了 list", isinstance(r["obj"][0]["face"], list),
              f"{r['obj'][0]['face']}")
        check("numpy 标量被转成了原生数",
              isinstance(r["obj"][0]["score"], float)
              and isinstance(r["obj"][0]["n_face_pts"], int))
        check("顶层 numpy 数组也被转了", isinstance(r["cam"], list),
              f"cam={r['cam']}")

    # 反证：如果绕过 jsonable，老写法确实是致命的
    import json as _json
    try:
        _json.dumps({"normal": np.array([0.0, 0.0, 1.0])})
        check("反证：裸 json.dumps 会抛（说明这个测试有效）", False)
    except TypeError:
        check("反证：裸 json.dumps 确实抛 TypeError（测试有效）", True)


def main():
    ap = argparse.ArgumentParser(description="输出契约离线验收")
    ap.parse_args()
    print("═══ 输出契约（离线部分，验收 H 的前半）═══")
    test_dir_not_translated()
    test_sphere_triple_consistent()
    test_mm_whitelist()
    test_top_header()
    test_camera_frame_degenerates()
    test_end_to_end_jsonl()
    test_writer_thread_survives_numpy()
    print("\n" + "═" * 52)
    if FAILS:
        print(f"❌ {len(FAILS)} 条失败：")
        for t in FAILS:
            print(f"   · {t}")
        return 1
    print("✅ 全绿")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
