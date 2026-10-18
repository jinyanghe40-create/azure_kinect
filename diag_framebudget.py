#!/usr/bin/env python3
"""给 run_azure.py 那 379ms 的"黑账"打表。

run_azure.py 只计了"检测"和"几何"。这个探针不改源码，靠猴子补丁把
剩下几段单独计时：等帧 / SDK 点云 / MJPG 解码 / 配对 / get() 总计。

用法（在板子上，用 run_with_x 保证 DISPLAY）:
    ~/work/azure_kinect/run_with_x.sh python3 - < diag_framebudget.py
"""
import sys
import time

import cv2

T = {}


def tick(name, dt):
    e = T.setdefault(name, [0.0, 0])
    e[0] += dt
    e[1] += 1


def _Wrap(name, fn):
    """包一层计时，保留原函数的一切行为（含异常）。

    必须返回【普通函数】而不是可调用对象 —— 赋给类属性时要走描述符协议
    才能绑上 self，否则 src.get(...) 会报 missing 'self'。
    """

    def wrapper(*a, **kw):
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            tick(name, time.perf_counter() - t)

    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


import azure_capture as ac                                    # noqa: E402

_orig_init = ac.AzureSource.__init__


def _init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    # ① SDK 取帧（阻塞等待，含相机帧间隔）
    self.k.k4a_device_get_capture = _Wrap("① 等帧 k4a_get_capture",
                                           self.k.k4a_device_get_capture)
    # ③ SDK 畸变校正点云
    if self.use_sdk_points:
        self._sdk_points = _Wrap("③ SDK 点云 _sdk_points", self._sdk_points)


ac.AzureSource.__init__ = _init
# ⑤ get() 全程（含上面各段 + 深度转换 + 旋转 + 抽样）
ac.AzureSource.get = _Wrap("⑤ get() 总计", ac.AzureSource.get)

# ④ 彩色 MJPG 解码（get() 里是 `import cv2` 后调，改模块属性即可生效）
cv2.imdecode = _Wrap("④ MJPG 解码 imdecode", cv2.imdecode)

import run_azure as ra                                        # noqa: E402

# ⑥ 2D 框 ↔ 3D 面 配对（run_azure.py:355，没计时）
ra.RR.match_classes = _Wrap("⑥ 配对 match_classes", ra.RR.match_classes)

sys.argv = ["run_azure.py", "--serial-dry", "--mm", "--stride", "2",
            "--min-dist", "0.3", "--max-dist", "1.5",
            "--min-cluster-points", "200", "--only-classified",
            "--frames", "20"]

try:
    ra.main()
except SystemExit:
    pass

print("\n" + "=" * 62)
print("分段计时 —— run_azure.py 自己【没有】计入 FPS 的那几段")
print("=" * 62)
for k in sorted(T):
    tot, cnt = T[k]
    print(f"{k:26s} 总 {tot * 1000:9.1f}ms  ×{cnt:3d}  平均 {tot / cnt * 1000:7.1f}ms")
print("=" * 62)
