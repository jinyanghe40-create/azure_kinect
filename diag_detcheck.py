#!/usr/bin/env python3
"""诊断：把保存下来的图喂给生产用的同一个检测器，看图/方向对检测的影响。

用法（板子上）:
  cd ~/work/azure_kinect && DISPLAY=:1 python3 ../diag_detcheck.py 图1 图2 ...
每张图跑 0° / 180° 两个方向，打印检测框。代码路径逐字照抄 run_azure.py。
"""
import importlib.util
import os
import sys

import cv2

HERE = "/home/nano/work/azure_kinect"
COMMON = "/home/nano/work/common"
RS = "/home/nano/work/realsense"
for p in (COMMON, RS, HERE):
    while p in sys.path:
        sys.path.remove(p)
for p in (COMMON, RS, HERE):
    sys.path.insert(0, p)

spec = importlib.util.spec_from_file_location(
    "_rr_shared", os.path.join(RS, "run_realsense.py"))
RR = importlib.util.module_from_spec(spec)
sys.modules["_rr_shared"] = RR
spec.loader.exec_module(RR)

eng = RR.Engine(RR.CFG["engine"])
S = RR.CFG["img_size"]
ANCH = RR.generate_anchors(S)
SCORE = RR.CFG["score_thresh"]
names = RR.CFG.get("classes") or RR.CFG.get("class_names") or []
print(f"[cfg] img_size={S} score_thresh={SCORE} classes={names}")

for path in sys.argv[1:]:
    img0 = cv2.imread(path)
    if img0 is None:
        print(f"!! 读不到 {path}")
        continue
    for tag, img in (("0deg", img0), ("180deg", cv2.rotate(img0, cv2.ROTATE_180))):
        out = eng.infer(RR.preprocess(img, S))
        dets = RR.postprocess(eng.get_output(out, "cls_logits")[0],
                              eng.get_output(out, "bbox_reg")[0],
                              ANCH, SCORE, RR.CFG["nms_iou"],
                              RR.CFG["max_det"])
        H, W = img.shape[:2]
        sx, sy = W / float(S), H / float(S)
        txt = []
        for (x0, y0, x1, y1, c, sc) in dets:
            nm = names[c] if c < len(names) else str(c)
            txt.append(f"{nm}:{sc:.2f}[{int(x0*sx)},{int(y0*sy)},{int(x1*sx)},{int(y1*sy)}]")
        print(f"{os.path.basename(path):16s} {tag:6s} n={len(dets):2d}  " +
              "  ".join(txt))
