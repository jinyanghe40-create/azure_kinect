#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零回归：把坐标系参数化之后，【缺省行为】必须和改动前逐位相同。

为什么值得单独一个文件：
  pc_pipeline 被三条路共用（RealSense 相机系 / 重力对齐竖直系 / 车体系）。
  把它参数化的时候，任何一个"顺手改"都可能悄悄改掉另外两条路的输出 ——
  而这在真机上表现为"精度好像差了点"，几乎不可能归因。

做法：`_history/` 里有一份改动前的 pc_pipeline 快照。把它和现在这份
  喂【同一份合成点云 + 同一组参数】，输出必须逐位相同。
  ★ 不是"看起来很接近"，是 json.dumps(sort_keys=True) 字符串相等。

  `_history` 快照不在时跳过（并说清楚跳过了），不会误报绿。

用法：
    python3 tools/test_zero_regression.py [-v]
退出码 0 = 全绿 / 跳过；1 = 有差异。
"""
import argparse
import importlib.machinery
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASELINE = os.path.join(ROOT, "_history",
                        "pc_pipeline.py.bak-before-splitdiag")
CURRENT = os.path.join(ROOT, "common", "pc_pipeline.py")


def _load(name, path):
    """从任意文件名加载 python —— 基线文件叫 *.py.bak-xxx，扩展名不是 .py，
    所以不能用 spec_from_file_location（它对未知扩展名直接返回 None）。"""
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    loader.exec_module(m)
    return m


def _args_for(mod, over):
    """拿默认参数再覆盖几项 —— 保证两边默认值来自各自的 parse_args()。"""
    bak = sys.argv
    sys.argv = [bak[0]]
    try:
        a = mod.parse_args()
    finally:
        sys.argv = bak
    a.synthetic = False
    a.replay = None
    a.json = True
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _norm(o):
    """转成可比字符串：去掉计时和逐簇点（内存字段），ndarray 转 list。"""
    o = dict(o)
    o.pop("timing_ms", None)
    objs = []
    for x in o.get("objects", []):
        x = dict(x)
        x.pop("pts", None)
        for k, v in list(x.items()):
            if isinstance(v, np.ndarray):
                x[k] = v.tolist()
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                x[k] = [{kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                         for kk, vv in d.items()} for d in v]
        objs.append(x)
    o["objects"] = objs
    return json.dumps(o, sort_keys=True, default=str)


CASES = [
    ("缺省（RealSense / 老行为）",        {}, False),
    ("老 --gravity",                      {"gravity": True}, False),
    ("老 gravity + 紧平面阈值",            {"gravity": True, "plane_thresh": 0.003}, False),
    ("老 + stride 4",                     {"stride": 4}, False),
    ("老 + max-dist 0.65",                {"max_dist": 0.65}, False),
    ("老 + voxel 0.01 / min_pts 40",      {"voxel": 0.01, "min_cluster_points": 40}, False),
    ("老 + face-select=largest",          {"face_select": "largest", "gravity": True}, False),
]


def main():
    ap = argparse.ArgumentParser(description="pc_pipeline 参数化零回归")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    print("═══ 零回归：pc_pipeline 坐标系参数化 ═══")
    if not os.path.exists(BASELINE):
        print(f"  ⏭  跳过：找不到改动前快照 {BASELINE}")
        print("     （_history 被清理过。要么恢复快照，要么承认这条没验。）")
        return 0

    old = _load("pc_baseline", BASELINE)
    new = _load("pc_current", CURRENT)
    print(f"  基线 {os.path.relpath(BASELINE, ROOT)}")
    print(f"  当前 {os.path.relpath(CURRENT, ROOT)}")

    ok = True
    for tag, over, _ in CASES:
        pts, _t = old.make_synthetic_scene(n_cubes=4, edge=0.05, seed=0,
                                           noise_sigma=0.002)
        A = _norm(old.run_pipeline(pts.copy(), _args_for(old, over), verbose=False))
        B = _norm(new.run_pipeline(pts.copy(), _args_for(new, over), verbose=False))
        same = A == B
        ok &= same
        print(f"    {tag:<28} {'✅ 逐位相同' if same else '❌ 有差异'}")
        if not same and a.verbose:
            print("      旧:", A[:300])
            print("      新:", B[:300])

    # ★ 别名等价：老 --gravity 必须 ≡ 新 --up-axis 0,1,0
    pts, _t = old.make_synthetic_scene(n_cubes=4, edge=0.05, seed=0,
                                       noise_sigma=0.002)
    A = _norm(old.run_pipeline(pts.copy(), _args_for(old, {"gravity": True}),
                               verbose=False))
    B = _norm(new.run_pipeline(pts.copy(), _args_for(new, {"up_axis": "0,1,0"}),
                               verbose=False))
    same = A == B
    ok &= same
    print(f"    {'老 --gravity ≡ 新 --up-axis':<28} "
          f"{'✅ 逐位相同' if same else '❌ 有差异'}")

    # ★ 新的三个参数在车体系取值下必须真的改变行为（否则等于没接上）
    pts, _t = old.make_synthetic_scene(n_cubes=4, edge=0.05, seed=0,
                                       noise_sigma=0.002)
    Base = _norm(new.run_pipeline(pts.copy(), _args_for(new, {}), verbose=False))
    Cam = _norm(new.run_pipeline(
        pts.copy(), _args_for(new, {"up_axis": "0,0,1", "cam_axis": "0,0,1",
                                    "cam_origin": "0.25,0,0.6"}), verbose=False))
    differ = Base != Cam
    ok &= differ
    print(f"    {'车体系参数真的改变行为':<28} "
          f"{'✅ 有差异（说明接上了）' if differ else '❌ 毫无变化（参数没接上！）'}")

    print("═" * 50)
    print("✅ 零回归通过" if ok else "❌ 有回归")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
