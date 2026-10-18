#!/usr/bin/env python3
"""帧预算 v2：逐帧记 get / infer / geo，把第 1 帧的 TensorRT 预热单独摘掉。

为什么要 v2：run_azure.py 的 [DONE] 行有两个坑，
  1) 它打印的"检测 Xms + 几何 Yms"是**最后一帧**的值（t_inf/t_geo 每帧被覆盖）；
  2) 它的 FPS 分母 t_loop 在第 1 帧 get() 之后才起表，而第 1 帧
     infer_ms=2567（TRT 预热）——于是 FPS 被系统性压低。
所以 v2 自己按 get() 的**入口时刻**定义帧周期，并丢掉第 1 帧。

用法（板子上）:
    cd ~/work/azure_kinect && ./run_with_x.sh python3 - < ~/work/diag_framebudget2.py
"""
import statistics as st
import sys
import time

import azure_capture as ac                                    # noqa: E402
import pc_pipeline                                            # noqa: E402


def _wrap(fn, sink):
    """返回普通函数（要走描述符协议绑 self），把耗时 append 进 sink。"""

    def wrapper(*a, **kw):
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            sink.append(time.perf_counter() - t)

    return wrapper


GET, GEO, INFER = [], [], []
GET_IN = []          # get() 入口时刻，用来算真实的帧周期


def _wrap_get(fn):
    def wrapper(*a, **kw):
        GET_IN.append(time.perf_counter())
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            GET.append(time.perf_counter() - t)

    return wrapper


ac.AzureSource.get = _wrap_get(ac.AzureSource.get)
pc_pipeline.run_pipeline = _wrap(pc_pipeline.run_pipeline, GEO)

import run_azure as ra                                        # noqa: E402

ra.RR.Engine.infer = _wrap(ra.RR.Engine.infer, INFER)

sys.argv = ["run_azure.py", "--serial-dry",
            "--stride", "2", "--min-dist", "0.3", "--max-dist", "1.5",
            "--min-cluster-points", "200", "--only-classified",
            "--frames", "30"]

try:
    ra.main()
except SystemExit:
    pass

print("\n" + "=" * 68)
print("逐帧账（第 1 帧 = TRT 预热，已剔除）")
print("=" * 68)

# 周期按 get() 入口算：period[i] = get_in[i+1] - get_in[i]
period = [GET_IN[i + 1] - GET_IN[i] for i in range(len(GET_IN) - 1)]
# 每段都对齐到"这一帧"，即从第 2 帧起（下标 1）
rows = []
for i, p in enumerate(period):          # i=0 对应第 2 帧
    k = i + 1
    if k >= len(GET):
        break
    g, f, e = GET[k] * 1e3, GEO[k] * 1e3, INFER[k] * 1e3
    rows.append((p * 1e3, g, f, e, (p - GET[k] - GEO[k] - INFER[k]) * 1e3))

cols = [("帧周期 period", 0), ("① get()", 1), ("② 检测 infer", 3),
        ("③ 几何 geo", 2), ("④ 残差(其余)", 4)]
warm = "（第 1 帧预热 %.0fms，未计入）" % (INFER[0] * 1e3) if INFER else ""
print(f"样本 {len(rows)} 帧  {warm}\n")
print(f"{'段':<18}{'平均':>9}{'p50':>9}{'p90':>9}{'最大':>9}{'占比':>8}")
print("-" * 68)
period_avg = st.mean(r[0] for r in rows)
for name, idx in cols:
    v = sorted(r[idx] for r in rows)
    avg = st.mean(v)
    print(f"{name:<18}{avg:9.1f}{st.median(v):9.1f}"
          f"{v[int(len(v) * 0.9)]:9.1f}{v[-1]:9.1f}{avg / period_avg * 100:7.1f}%")
print("-" * 68)
print(f"{'合计':<18}{sum(st.mean(r[i] for r in rows) for _, i in cols):9.1f}")
print(f"\n稳态帧率 = 1/{period_avg / 1000:.3f}s = {1000 / period_avg:.2f} FPS"
      f"   （[DONE] 行报的 1.9 是把 %.1fs 预热算进了分母）" % (INFER[0] if INFER else 0))
print("=" * 68)
