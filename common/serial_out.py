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

    {"t":12.345,"frame":123,"seq":45,"n":1,
     "coord":"body","rev":2,"cam":[0.25,0.0,0.6],"cam_unit":"m",
     "calib":"buggy01-20261015","obj_unit":"mm","infer_ms":31.2,
     "obj":[
        {"cls_id":1,"cls":"medicine","score":0.69,
         "face":[X,Y,Z],          # ★ 面心（直接测到的，可靠，抖动~5mm）
         "normal":[nx,ny,nz],     # ★ 面朝向单位向量 —— 完整的"角度"，别只发 angle
         "cube":[X,Y,Z],          # 体心（推出来的，抖动~14mm，看 cube_ok）
         "edge":0.335,"edge_src":"auto",
         "n_faces":3,"pos_type":"cube","cube_ok":true,
         "range":0.842,"az":-3.11,"el":9.42,"nearest":0.815,
         "quality":"GOOD","n_face_pts":1723,
         "angle":48.2,"lean":7.3,"rms_mm":0.9}]}

  ── 顶层头（★ 电控必须读，否则分不清新旧协议）────────────────────
     coord     "body"   = 下面的 face/cube/normal 在【车体系】里
                         X前 / Y左 / Z上，原点 = 车体几何中心（带高度）
               "camera" = 退化模式（没装标定），就是相机系，别用
     rev       2        = 本协议版本。rev=1 是旧版（无 coord/lean/cam）
     cam       [x,y,z]  = 相机光心在【车体系】的位置，恒为米。
                          电控想自己算"到相机的距离"就用它减 face。
     cam_unit  "m"      = cam 的单位（恒为米，和 --mm 无关，见下）
     calib     "<id>"   哪份标定算出来的；null = 没标定（此时 coord="camera"）
     obj_unit  "m"/"mm" = ★【只管 obj[] 里那些长度字段】的单位。
                          顶层 cam 和 ROS 点云恒为米 —— 两者可能不同！
     infer_ms  这一帧的推理耗时（毫秒）

  ── 坐标系约定（★ 最容易搞错的地方）────────────────────────────
     face / cube   车体系，米（obj_unit=mm 时是毫米）
     normal        车体系里的单位向量（【不】随 obj_unit 变）
     lean          法向偏离【车体竖直】的角度：0°=铅垂面，90°=水平面
                   ← 换车体系后这才是电控要的"物体歪没歪"
     angle         法向与【相机光轴】的夹角，度
     (range,az,el) ★★ 这三个在【相机系】里算，不是车体系！
                   它们是一个球坐标三元组，range 必须"到相机光心"
                   才和 az/el 自洽；而且 quality 的阈值是按相机距离
                   实测标定的。**别拿 range 当"到车心的距离"** ——
                   要那个请自己算 |face − cam|。
       相机系：+X 右 / +Y 下 / +Z 前(光轴)，原点=深度相机光心
         az = atan2(X, Z)          正 = 在相机右边
         el = atan2(-Y, √(X²+Z²))  正 = 在相机上方
       ⚠️ 倒装(rotate=180)时这是【转正后】的光学系 —— 和屏幕上
          那张图的方向一致，不是传感器原始方向。

  ★★ 给电控的判读规则（实测数据支撑）：
     主数据用 face + normal —— 它们是平面拟合的直接结果。
     体心 cube 是推出来的：cube = face − (edge/2)·Σn，误差被 edge/2≈167mm 放大。
     实测抖动：face 5.0mm / cube 14.0mm（2.8 倍）/ normal→角度 0.45°。

     cube_ok == true   → n_faces==3 且边长是各面一致时量的，体心可用
     cube_ok == false  → 面数不足或边长不一致，体心是估的，建议只用 face+normal

     edge_src: auto     各面量出来一致 → 可信
               auto-max 各面差太多，取最大面（遮挡只会让面变小）→ 勉强
               cli      一个面都没量到，用了命令行兜底 → 不可信

     quality:  GOOD   可以直接拿去做抓取/精细操作（面心抖动 ~2mm）
               FAIR   可以引导机械臂靠近、大致对准（~10mm）
               COARSE 只够知道东西大概在哪个方向（>20mm），别用来定位

  xyz/angle 在没有深度相机时为 null（先把协议定下来，深度好了直接填）

用法（独立测试，不接硬件）：
    python3 serial_out.py --dry-run          # 打到 stdout 看格式
    python3 serial_out.py --port /dev/ttyGS0 # 真发
"""
import argparse
import json
import os
import queue
import sys
import threading
import time

# jsonable 和自己同目录。不管是被 import 还是被当脚本跑都要能找到它。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import jsonable                                          # noqa: E402


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
        self.encode_errs = 0                 # 编码失败丢掉的帧数（见 _worker）
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
            # ★★ 编码包在 try 里 —— 以前它裸露在外，一个 TypeError 就把这个
            #   后台线程【永久打死】：主循环的 send() 继续成功返回、队列继续
            #   涨，可是一个字都发不出去，而且没有任何报错。
            #   典型触发：某个字段是 numpy 数组（normal 就是）。见 jsonable.py。
            try:
                line = (jsonable.dumps(rec, ensure_ascii=False,
                                       separators=(",", ":")) + "\n").encode()
            except Exception as e:
                self.encode_errs += 1
                if self.encode_errs <= 3 or self.encode_errs % 100 == 0:
                    print(f"[串口] ❌ 这一帧编码失败（第 {self.encode_errs} 次）"
                          f"：{type(e).__name__}: {e}", file=sys.stderr)
                    print("       这帧丢掉了，但串口还活着。", file=sys.stderr)
                continue

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
            msg = f"[串口] 共发送 {self.sent} 条，丢弃 {self.dropped} 条"
            if self.encode_errs:
                msg += f"，★ 编码失败 {self.encode_errs} 条"
            print(msg)


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
