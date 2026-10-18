#!/usr/bin/env python3
"""
串口输出模块 —— 把检测结果按 JSON Lines 协议发给另一台设备
（单片机 / STM32 / PLC / 机械臂 / 上位机）

设计要点（都是踩过坑才这么写的）：
  1. **后台线程 + 队列**：串口写可能卡住（对端不读就阻塞），
     绝不能让推理主循环等它 —— 所以写操作全丢给后台线程。
  2. **自动重连**：USB 串口被拔掉/板子重启，串口对象会失效。
     后台线程负责重试，主循环只管往队列里丢数据。
  3. **限速**：推理 30FPS，但下游单片机通常 10~20Hz 就够，
     发太快会把对端缓冲冲爆。默认上限 20Hz。
  4. **新数据优先**：队列满了丢最旧的（实时性 > 完整性）。

协议：一行一个 JSON（JSON Lines），便于对端按行解析
    {"t":12.345,"frame":123,"seq":45,"n":1,"obj":[
        {"cls_id":1,"cls":"medicine","score":0.69,
         "face":[X,Y,Z],          # ★ 面心（直接测到的，可靠，抖动~5mm）
         "normal":[nx,ny,nz],     # ★ 面朝向单位向量 —— 完整的"角度"，别只发 angle
         "cube":[X,Y,Z],          # 体心（推出来的，抖动~14mm，看 cube_ok）
         "edge":0.335,"edge_src":"auto",
         "n_faces":3,"pos_type":"cube","cube_ok":true,
         "angle":48.2,"rms_mm":0.9}]}

  ★★ 给电控的判读规则（实测数据支撑）：
     主数据用 face + normal —— 它们是平面拟合的直接结果。
     体心 cube 是推出来的：cube = face − (edge/2)·Σn，误差被 edge/2≈167mm 放大。
     实测抖动：face 5.0mm / cube 14.0mm（2.8 倍）/ normal→角度 0.45°。

     cube_ok == true   → n_faces==3 且边长是各面一致时量的，体心可用
     cube_ok == false  → 面数不足或边长不一致，体心是估的，建议只用 face+normal

     edge_src: auto     各面量出来一致 → 可信
               auto-max 各面差太多，取最大面（遮挡只会让面变小）→ 勉强
               cli      一个面都没量到，用了命令行兜底 → 不可信

  xyz/angle 在没有深度相机时为 null（先把协议定下来，深度好了直接填）

用法（独立测试，不接硬件）：
    python3 serial_out.py --dry-run          # 打到 stdout 看格式
    python3 serial_out.py --port /dev/ttyGS0 # 真发
"""
import argparse
import json
import queue
import threading
import time


class SerialOut:
    def __init__(self, port=None, baud=115200, hz=20.0, dry_run=False,
                 verbose=True):
        self.port = port
        self.baud = baud
        self.min_dt = 1.0 / hz if hz > 0 else 0.0
        self.dry_run = dry_run or (port is None)
        self.verbose = verbose

        self.q = queue.Queue(maxsize=4)      # 小队列 → 保证发的是最新数据
        self.seq = 0
        self.sent = 0
        self.dropped = 0
        self._last_t = 0.0
        self._stop = threading.Event()
        self._ser = None
        self._lock = threading.Lock()

        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    # ---------------- 供主循环调用（绝不阻塞）----------------
    def send(self, frame_no, dets, infer_ms=None, extra=None):
        """dets: [(x0,y0,x1,y1, cls_id, score), ...]（原图坐标系）
        cls_names/中文名由调用方通过 extra 或 set_classes 提供"""
        now = time.time()
        if now - self._last_t < self.min_dt:
            return                            # 限速
        self._last_t = now
        self.seq += 1
        rec = {"t": round(now, 3), "frame": int(frame_no),
               "seq": self.seq, "n": len(dets),
               "obj": [self._obj(*d) if not isinstance(d, dict) else d
                       for d in dets]}
        if infer_ms is not None:
            rec["infer_ms"] = round(float(infer_ms), 1)
        if extra:
            rec.update(extra)
        try:
            self.q.put_nowait(rec)
        except queue.Full:                    # 丢最旧的，保实时
            try:
                self.q.get_nowait()
                self.dropped += 1
                self.q.put_nowait(rec)
            except queue.Empty:
                pass

    def set_mapper(self, fn):
        """fn(x0,y0,x1,y1,cls_id,score) -> dict"""
        self._obj = fn

    def _obj(self, x0, y0, x1, y1, cls_id, score):
        return {"cls_id": int(cls_id), "score": round(float(score), 3),
                "box": [int(x0), int(y0), int(x1), int(y1)]}

    # ---------------- 后台线程 ----------------
    def _open(self):
        import serial                     # pyserial
        s = serial.Serial(self.port, self.baud, timeout=0.05,
                          write_timeout=0.2)
        if self.verbose:
            print(f"[串口] ✅ 已打开 {self.port} @ {self.baud}")
        return s

    def _worker(self):
        while not self._stop.is_set():
            try:
                rec = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            line = (json.dumps(rec, ensure_ascii=False,
                               separators=(",", ":")) + "\n").encode()

            if self.dry_run:
                print(line.decode().rstrip())
                self.sent += 1
                continue

            for attempt in range(2):          # 写失败就重开一次
                try:
                    if self._ser is None:
                        self._ser = self._open()
                    self._ser.write(line)
                    self.sent += 1
                    break
                except Exception as e:
                    if self.verbose:
                        print(f"[串口] ⚠️ 写失败({e})，重连…")
                    self._close()
                    time.sleep(0.5)

    def _close(self):
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    def close(self):
        self._stop.set()
        self._t.join(timeout=1.5)
        self._close()
        if self.verbose:
            print(f"[串口] 共发送 {self.sent} 条，丢弃 {self.dropped} 条")


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="串口输出自测")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--hz", type=float, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seconds", type=float, default=3)
    a = ap.parse_args()

    out = SerialOut(a.port, a.baud, a.hz, dry_run=a.dry_run)

    def mapper(x0, y0, x1, y1, cid, sc):
        return {"cls": ["tools", "medicine", "instrument", "food"][cid],
                "cn": ["工具", "药品", "仪器", "食品"][cid],
                "score": round(float(sc), 3),
                "box": [int(x0), int(y0), int(x1), int(y1)],
                "xyz": None, "angle": None}

    out.set_mapper(mapper)
    t0 = time.time()
    n = 0
    while time.time() - t0 < a.seconds:
        n += 1
        out.send(n, [(10, 20, 100, 120, n % 4, 0.9)], infer_ms=31.0)
        time.sleep(1 / 30.0)                  # 模拟 30FPS 推理
    out.close()
    print(f"[测试] 主循环跑了 {n} 帧（30FPS），串口按 {a.hz}Hz 限速发送")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
