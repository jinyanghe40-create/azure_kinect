#!/usr/bin/env python3
"""Generate Python oracle fixtures for the C++ unit tests.

Run from anywhere:
    python3 azure_kinect_cpp/tests/gen_fixtures.py
Outputs go to azure_kinect_cpp/tests/fixtures/ (binary little-endian float32
plus small JSON headers).  The C++ tests skip quietly when these are absent.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "fixtures")
sys.path.insert(0, os.path.join(ROOT, "common"))

import ssd_decoder  # noqa: E402


def write_f32(name, arr):
    path = os.path.join(OUT, name)
    np.ascontiguousarray(arr, dtype="<f4").tofile(path)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)

    anchors = ssd_decoder.generate_anchors(320).astype(np.float32)
    assert anchors.shape == (3234, 4), anchors.shape
    write_f32("anchors.f32", anchors)
    with open(os.path.join(OUT, "anchors.json"), "w") as f:
        json.dump({"shape": list(anchors.shape)}, f)

    meta = {"seeds": []}
    n = anchors.shape[0]
    for seed in (101, 202, 303):
        rng = np.random.default_rng(seed)
        cls_logits = rng.normal(0.0, 3.0, (n, 5)).astype(np.float32)
        bbox_reg = rng.normal(0.0, 1.5, (n, 4)).astype(np.float32)
        dets = ssd_decoder.postprocess(cls_logits, bbox_reg, anchors)
        write_f32(f"cls_{seed}.f32", cls_logits)
        write_f32(f"reg_{seed}.f32", bbox_reg)
        with open(os.path.join(OUT, f"dets_{seed}.json"), "w") as f:
            json.dump(
                [
                    {
                        "x0": d[0], "y0": d[1], "x1": d[2], "y1": d[3],
                        "cls": int(d[4]), "score": d[5],
                    }
                    for d in dets
                ],
                f,
                indent=1,
            )
        meta["seeds"].append(int(seed))
        print(f"seed {seed}: {len(dets)} detections")

    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("fixtures ->", OUT)


if __name__ == "__main__":
    main()
