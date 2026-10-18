#!/usr/bin/env python3
"""换分辨率之前，先把同一套管线在那一档真跑一遍（不猜、不外推）。

只改 AzureSource 的默认 color_resolution（run_azure.py 没传这个参数，
吃的是类默认 720P），**不动任何源码**。

用法（板子上）:
    cd ~/work/azure_kinect && ./run_with_x.sh python3 - 2160p < ~/work/diag_res.py
    （可选档位: 720p / 1080p / 1440p / 1536p / 2160p）
"""
import statistics as st
import sys
import time

import azure_capture as ac                                    # noqa: E402

MODES = {
    "720p": ac.K4A_COLOR_RESOLUTION_720P,
    "1080p": ac.K4A_COLOR_RESOLUTION_1080P,
    "1440p": ac.K4A_COLOR_RESOLUTION_1440P,
    "1536p": ac.K4A_COLOR_RESOLUTION_1536P,
    "2160p": ac.K4A_COLOR_RESOLUTION_2160P,
}
want = (sys.argv[1] if len(sys.argv) > 1 else "2160p").lower()
if want not in MODES:
    sys.exit(f"档位只能是 {list(MODES)}，给的是 {want!r}")
RES = MODES[want]

_orig_init = ac.AzureSource.__init__


def _init(self, *a, **kw):
    kw["color_resolution"] = RES          # ★ 强制这一档
    _orig_init(self, *a, **kw)
    ic = self.intr_color                        # 是个 dict，不是对象
    print(f"[分辨率] 请求 {want}  →  实到彩色图 "
          f"{ic.get('width')}x{ic.get('height')} fx={ic.get('fx'):.1f}")


ac.AzureSource.__init__ = _init


def _wrap(fn, sink):
    def wrapper(*a, **kw):
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            sink.append(time.perf_counter() - t)

    return wrapper


GET, GEO, INFER, DECODE, PCLOUD = [], [], [], [], []
GET_IN = []


def _wrap_get(fn):
    def wrapper(*a, **kw):
        GET_IN.append(time.perf_counter())
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            GET.append(time.perf_counter() - t)

    return wrapper


import cv2                                                    # noqa: E402
import pc_pipeline                                            # noqa: E402

ac.AzureSource.get = _wrap_get(ac.AzureSource.get)
ac.AzureSource._sdk_points = _wrap(ac.AzureSource._sdk_points, PCLOUD)
cv2.imdecode = _wrap(cv2.imdecode, DECODE)
pc_pipeline.run_pipeline = _wrap(pc_pipeline.run_pipeline, GEO)

import run_azure as ra                                        # noqa: E402

ra.RR.Engine.infer = _wrap(ra.RR.Engine.infer, INFER)

N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
sys.argv = ["run_azure.py", "--serial-dry",
            "--stride", "2", "--min-dist", "0.3", "--max-dist", "1.5",
            "--min-cluster-points", "200", "--only-classified",
            "--frames", str(N)]

try:
    ra.main()
except SystemExit:
    pass


def stats(v, label, unit="ms"):
    if not v:
        print(f"{label:<22}  （没采到）")
        return
    x = sorted(v)
    print(f"{label:<22}{st.mean(x):8.1f}{st.median(x):8.1f}"
          f"{x[int(len(x) * 0.9)]:8.1f}{x[-1]:8.1f}")


period = [(GET_IN[i + 1] - GET_IN[i]) * 1e3 for i in range(len(GET_IN) - 1)]
drop = 1                                   # 第 1 帧含 TRT 预热
body = slice(drop, len(period))

print("\n" + "=" * 62)
print(f"档位 {want}   ——  第 1 帧（TRT 预热）已剔除，样本 {len(period) - drop} 帧")
print("=" * 62)
print(f"{'段':<22}{'平均':>8}{'p50':>8}{'p90':>8}{'最大':>8}")
print("-" * 62)
stats(period[body], "帧周期 period")
stats([v * 1e3 for v in GET[drop + 1:]], "① get() 全程")
stats([v * 1e3 for v in DECODE[drop:]], "  └ MJPG 解码")
stats([v * 1e3 for v in PCLOUD[drop:]], "  └ SDK 点云")
stats([v * 1e3 for v in INFER[drop:]], "② TRT infer")
stats([v * 1e3 for v in GEO[drop:]], "③ 几何 geo")
if len(period) > drop:
    p50 = st.median(period[body])
    g = st.median([v * 1e3 for v in GET[drop + 1:]]) if len(GET) > drop + 1 else 0
    print("-" * 62)
    print(f"p50 帧率 = {1000 / p50:.2f} FPS   （1/{p50:.0f}ms）")
    print(f"其中 get() 占 {g / p50 * 100:.0f}%")
print("=" * 62)
