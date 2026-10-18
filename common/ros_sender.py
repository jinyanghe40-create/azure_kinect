#!/usr/bin/env python3
"""
板子侧：把点云 + 检测结果通过 TCP 发给电脑（给 ROS/RViz 显示用）

板子是 Ubuntu 18.04，装不了 ROS 2 Humble（要 22.04），而且 Nano 跑 RViz 会卡死。
所以分工是：**板子只发数据，电脑负责可视化**。

传输格式（自定义的简单协议，避免在板子上装 ROS）：
    ┌────────────┬──────────────┬────────────────┬──────────────┐
    │ 'K4PC' 4B  │ 头长度 u32   │ 头 (JSON UTF-8) │ 点数据 float32│
    └────────────┴──────────────┴────────────────┴──────────────┘
    头里: {"v":1, "t":时间, "n":点数, "frame_id":..., "w":.., "h":..,
           "intr":{fx,fy,cx,cy}, "objects":[{cls,score,box,xyz,angle,...}]}
    点数据: n × 3 个 float32（X,Y,Z 米）

用法：
    from ros_sender import RosSender
    s = RosSender("10.42.0.1", 9100)          # 电脑的 IP
    s.send(pts, objects, intr, (1280, 720))
    s.close()

设计上和 serial_out.py 一样：**后台线程 + 队列 + 丢最旧**，
绝不阻塞推理主循环。
"""
import json
import os
import queue
import socket
import struct
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import jsonable                                          # noqa: E402

MAGIC = b"K4PC"


class RosSender:
    def __init__(self, host, port=9100, max_points=30000, hz=10.0,
                 verbose=True):
        self.host = host
        self.port = port
        self.max_points = max_points
        self.min_dt = 1.0 / hz if hz > 0 else 0.0
        self.verbose = verbose

        self.q = queue.Queue(maxsize=2)
        self.sent = 0
        self.dropped = 0
        self.encode_errs = 0                 # 编码失败丢掉的帧数（见 _worker）
        self.bytes_sent = 0
        self._last_t = 0.0
        self._stop = threading.Event()
        self._sock = None
        self._th = threading.Thread(target=self._worker, daemon=True)
        self._th.start()
        if verbose:
            print(f"[ROS桥] 目标 {host}:{port}  最多发 {max_points} 点  "
                  f"限速 {hz}Hz")

    # ---------------- 主循环调用（绝不阻塞）----------------
    def send(self, pts, objects=None, intr=None, color_size=None,
             frame_id="camera_depth_optical_frame", rot_k=0, image=None,
             jpg_quality=70, obj_unit="m", extra=None):
        # ★ obj_unit: objects 字典里长度字段的单位（"m" / "mm"）。
        #   点云 pts 恒为米（RViz 按米画），所以两者可能不一致 ——
        #   写进头里让电脑侧一眼看得出，别靠猜。
        """
        pts:     (N,3) float32，相机坐标系，米
        objects: run_realsense.py 里 match_classes() 的输出（可以为空）
        """
        now = time.time()
        if now - self._last_t < self.min_dt:
            return
        self._last_t = now

        pts = np.asarray(pts, dtype=np.float32)
        n_total = len(pts)
        # 抽稀到 max_points（等间隔取，保留空间分布）
        if n_total > self.max_points:
            idx = np.linspace(0, n_total - 1, self.max_points).astype(np.int64)
            pts = pts[idx]

        head = {
            "v": 1,
            "t": round(now, 4),
            "frame_id": frame_id,
            # ★ 彩色图转了几个 90°（0~3）。电脑侧桥接靠它把 RViz 的
            #   "显示系"也转正 —— 相机装倒时，点云在原始深度系里是倒的，
            #   不告诉 RViz 的话，法向箭头在屏幕上会指反（算法没错，
            #   是显示没转正），很容易被误判成"角度算错了"。
            "rot_k": int(rot_k) % 4,
            # ★ 彩色图的 JPEG 字节数（0 = 这帧没带图）。
            #   带上真实照片，RViz 里就能把"照片"和"点云+拟合面"并排看，
            #   角度对不对一眼就分晓 —— 不用再靠脑补。
            "jpg_bytes": 0,
            "n": int(len(pts)),
            "n_raw": int(n_total),
            "w": int(color_size[0]) if color_size else 0,
            "h": int(color_size[1]) if color_size else 0,
            "intr": intr or {},
            "objects": objects or [],
            # ★ objects 的单位。点云恒为米，所以这里可能和点云不同。
            "obj_unit": obj_unit,
        }
        # ★★ 坐标系标识等元数据（coord / rev / cam / calib）。
        #   与串口那路【共用同一个 dict】（run_azure.py 的 top_extra），
        #   避免两条链路各写一份、改了这条忘了那条 —— 电控和 RViz 看到的
        #   坐标系说明必须一字不差。
        if extra:
            head.update(extra)
        # ★ 把彩色图压成 JPEG 一起发（几十 KB/帧，网线毫无压力）
        jpg = b""
        if image is not None:
            try:
                import cv2
                ok, buf = cv2.imencode(
                    ".jpg", image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
                if ok:
                    jpg = buf.tobytes()
                    head["jpg_bytes"] = len(jpg)
            except Exception:
                jpg = b""

        try:
            self.q.put_nowait((head, pts, jpg))
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
                self.q.put_nowait((head, pts, jpg))
            except queue.Empty:
                pass

    # ---------------- 后台线程 ----------------
    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((self.host, self.port))
        s.settimeout(None)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self.verbose:
            print(f"[ROS桥] ✅ 已连上 {self.host}:{self.port}")
        return s

    def _worker(self):
        while not self._stop.is_set():
            try:
                head, pts, jpg = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            # ★★ 和串口那条一样的坑：head 里也可能有 numpy 标量/数组
            #   （cam、objects 里的 normal…），json.dumps 会抛 TypeError，
            #   而这个异常在后台线程里没人接 → 桥【永久】不再发数据。
            #   以前裸着就是因为没人想到 head 里会有 numpy。
            try:
                hb = jsonable.dumps(head, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")
            except Exception as e:
                self.encode_errs += 1
                if self.encode_errs <= 3:
                    print(f"[ROS桥] ❌ 这一帧头编码失败（第 {self.encode_errs} 次）"
                          f"：{type(e).__name__}: {e}", file=sys.stderr)
                continue
            # ★ 注意：jpg 的长度已经在【头】里的 "jpg_bytes" 了，
            #   这里【不能】再写一个 4 字节的长度字段 —— 接收端不会读它，
            #   结果就是整个流错位 4 字节（表现为"JPEG 头不是 ff d8"、
            #   下一帧 magic 读到 JPEG 尾巴）。之前就是这么错的。
            pkt = (MAGIC + struct.pack("<I", len(hb)) + hb
                   + pts.astype("<f4").tobytes()
                   + jpg)

            for _ in range(2):
                try:
                    if self._sock is None:
                        self._sock = self._connect()
                    self._sock.sendall(pkt)
                    self.sent += 1
                    self.bytes_sent += len(pkt)
                    break
                except Exception as e:
                    if self.verbose:
                        print(f"[ROS桥] ⚠️ 发送失败({type(e).__name__})，重连…")
                    self._close()
                    time.sleep(0.5)

    def _close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def close(self):
        self._stop.set()
        self._th.join(timeout=2.0)
        self._close()
        if self.verbose:
            print(f"[ROS桥] 共发送 {self.sent} 帧 "
                  f"({self.bytes_sent/1e6:.1f} MB)，丢弃 {self.dropped} 帧")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 自测：往本机发合成点云（电脑上先开 cloud_bridge.py 再跑这个）
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    s = RosSender(host, 9100, max_points=5000, hz=5)
    print("发 5 帧合成点云（一个倾斜面 + 一个方块）…")
    rng = np.random.RandomState(0)
    for i in range(5):
        n = 6000
        x = rng.uniform(-0.3, 0.3, n)
        y = rng.uniform(-0.2, 0.2, n)
        z = 1.0 + 0.3 * x + rng.normal(0, 0.003, n)
        pts = np.stack([x, y, z], 1).astype(np.float32)
        objs = [{"cls": "tools", "cn": "工具", "score": 0.93,
                 "xyz": [0.0, 0.0, 1.0], "angle": 16.7,
                 "n_points": 4000, "rms_mm": 2.9,
                 "normal": [-0.287, 0.0, -0.958]}]
        s.send(pts, objs, {"fx": 918.4, "fy": 919.2, "cx": 637.6,
                           "cy": 351.1}, (1280, 720))
        time.sleep(0.25)
    time.sleep(0.5)
    s.close()
