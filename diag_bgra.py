#!/usr/bin/env python3
"""绕开 MJPG 解码：让相机直接吐裸 BGRA32，看能不能白拿高清。

动机（10-13 实测）：
    720p p50 336ms / 2.98FPS，其中 MJPG 解码 47ms；
    2160p p50 586ms / 1.71FPS，其中 MJPG 解码 313ms（占整帧 53%）。
解码是唯一被分辨率打中的环节 —— Azure 除了 MJPG 还支持 BGRA32（裸格式，
压根没有 JPEG 可解）。若 1080p/1440p 能用 BGRA32，就等于
【更高分辨率 + 解码费归零】。

不动源码的两处：
  1) cfg.color_format 读的是模块常量 K4A_IMAGE_FORMAT_COLOR_MJPG（源码里
     两个分支写的是同一个值，本来就是写死的）→ 直接改常量。
  2) get() 里对彩色缓冲无条件调 cv2.imdecode → 换成认裸 BGRA 的版本。

用法（板子上）:
    cd ~/work/azure_kinect && ./run_with_x.sh python3 - 1080p 15 < ~/work/diag_bgra.py
"""
import statistics as st
import sys
import time

import numpy as np

import azure_capture as ac                                    # noqa: E402

MODES = {
    "720p": ac.K4A_COLOR_RESOLUTION_720P,
    "1080p": ac.K4A_COLOR_RESOLUTION_1080P,
    "1440p": ac.K4A_COLOR_RESOLUTION_1440P,
    "1536p": ac.K4A_COLOR_RESOLUTION_1536P,
    "2160p": ac.K4A_COLOR_RESOLUTION_2160P,
}
want = (sys.argv[1] if len(sys.argv) > 1 else "1080p").lower()
if want not in MODES:
    sys.exit(f"档位只能是 {list(MODES)}，给的是 {want!r}")
RES = MODES[want]

# ---- ① 把"MJPG"常量顶成 BGRA32（cfg.color_format 会读它）----
ac.K4A_IMAGE_FORMAT_COLOR_MJPG = ac.K4A_IMAGE_FORMAT_COLOR_BGRA32

_orig_init = ac.AzureSource.__init__


def _init(self, *a, **kw):
    kw["color_resolution"] = RES
    _orig_init(self, *a, **kw)
    ic = self.intr_color
    print(f"[分辨率] 请求 {want}  →  实到 {ic.get('width')}x{ic.get('height')} "
          f"fx={ic.get('fx'):.1f}")


ac.AzureSource.__init__ = _init

# ---- ② 接管取色：裸 BGRA 直接 reshape，不做解码 ----
import cv2                                                    # noqa: E402

_real_imdecode = cv2.imdecode
# Azure 的全部彩色档位（宽, 高）
RES_LIST = [(1280, 720), (1920, 1080), (2048, 1536),
            (2560, 1440), (3840, 2160), (4096, 3072)]


def imdecode_bgra_aware(buf, flags):
    if buf.size >= 2 and buf[0] == 0xFF and buf[1] == 0xD8:   # 还是 JPEG
        return _real_imdecode(buf, flags)
    for (w, h) in RES_LIST:                                   # 无行填充
        if buf.size == w * h * 4:
            return buf.reshape(h, w, 4)[:, :, :3].copy()
    for (w, h) in RES_LIST:                                   # 带行填充
        if buf.size % h == 0 and (buf.size // h) % 4 == 0 \
                and buf.size // h >= w * 4:
            stride = buf.size // h
            return buf.reshape(h, stride // 4, 4)[:, :w, :3].copy()
    raise RuntimeError(f"彩色缓冲 {buf.size} 字节对不上任何档位")


cv2.imdecode = imdecode_bgra_aware


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


import pc_pipeline                                            # noqa: E402

ac.AzureSource.get = _wrap_get(ac.AzureSource.get)
ac.AzureSource._sdk_points = _wrap(ac.AzureSource._sdk_points, PCLOUD)
# ✎ 注意：这里 cv2.imdecode 已被换成裸 BGRA 版，量到的就是"取色"而不是"解码"
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


def stats(v, label):
    if not v:
        print(f"{label:<22}  （没采到）")
        return
    x = sorted(v)
    print(f"{label:<22}{st.mean(x):8.1f}{st.median(x):8.1f}"
          f"{x[int(len(x) * 0.9)]:8.1f}{x[-1]:8.1f}")


period = [(GET_IN[i + 1] - GET_IN[i]) * 1e3 for i in range(len(GET_IN) - 1)]
drop = 1
print("\n" + "=" * 62)
print(f"BGRA32 档位 {want} —— 第 1 帧（TRT 预热）已剔除，"
      f"样本 {max(0, len(period) - drop)} 帧")
print("=" * 62)
print(f"{'段':<22}{'平均':>8}{'p50':>8}{'p90':>8}{'最大':>8}")
print("-" * 62)
stats(period[drop:], "帧周期 period")
stats([v * 1e3 for v in GET[drop + 1:]], "① get() 全程")
stats([v * 1e3 for v in DECODE[drop:]], "  └ 取色(原解码)")
stats([v * 1e3 for v in PCLOUD[drop:]], "  └ SDK 点云")
stats([v * 1e3 for v in INFER[drop:]], "② TRT infer")
stats([v * 1e3 for v in GEO[drop:]], "③ 几何 geo")
if len(period) > drop:
    p50 = st.median(period[drop:])
    print("-" * 62)
    print(f"p50 帧率 = {1000 / p50:.2f} FPS   （1/{p50:.0f}ms）")
print("=" * 62)
