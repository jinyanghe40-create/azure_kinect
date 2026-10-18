#!/usr/bin/env python3
"""
Jetson 板上推理 —— SSDlite 检测器，TensorRT 加速

★ 不需要 pycuda！用 TensorRT 原生 Python API（自带 GPU 内存管理）。

依赖:
    - ssd_decoder.py（同目录，已在电脑上验证与 torchvision 逐位一致）
    - models/best_trt.engine  （或 best_trt.onnx，会自动编译）
    - tensorrt (JetPack 自带), numpy, opencv

用法:
    python3 trt_infer.py                     # 摄像头 /dev/video2
    python3 trt_infer.py --video xx.avi      # 视频文件
    python3 trt_infer.py --bench 100         # 只测速
    python3 trt_infer.py --rebuild           # 从 ONNX 重新编译 engine
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ssd_decoder import generate_anchors, postprocess   # noqa: E402

CFG = {
    "onnx": os.path.join(HERE, "..", "models", "best_trt.onnx"),
    "engine": os.path.join(HERE, "..", "models", "best_trt.engine"),
    "img_size": 320,
    "score_thresh": 0.4,
    "nms_iou": 0.55,
    "max_det": 20,
    "fp16": True,
    "classes": ["tools", "medicine", "instrument", "food"],
    "class_cn": {"tools": "工具", "medicine": "药品",
                 "instrument": "仪器", "food": "食品"},
    "colors": [(60, 76, 231), (60, 200, 200), (240, 140, 60), (60, 210, 60)],
}


# =====================================================================
#                     TensorRT 部分（无 pycuda）
# =====================================================================
def build_engine(onnx_path, engine_path, fp16=True):
    """ONNX -> TensorRT engine"""
    import tensorrt as trt
    print(f"[TRT] 编译 {os.path.basename(onnx_path)} -> engine")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ❌ {parser.get_error(i)}")
            raise RuntimeError("ONNX 解析失败")
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TRT] FP16 已启用")
    print("[TRT] 编译中（1-5 分钟）...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine 编译失败")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[TRT] ✅ 完成，用时 {time.time()-t0:.0f}s")
    return engine_path


class Engine:
    """TensorRT 推理封装 —— 用 execute_v2 + 自己管 device buffer"""

    def __init__(self, engine_path):
        import tensorrt as trt
        import cuda_rt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("engine 反序列化失败")
        self.context = self.engine.create_execution_context()

        self.bufs = {}      # name -> (host_array, device_ptr)
        self.input_name = None
        self.output_names = []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            n = int(np.prod(shape))
            host = np.empty(n, dtype=dtype)
            dev = cuda_rt.mem_alloc(host.nbytes)
            self.bufs[name] = (host, dev, shape)
            if self.engine.binding_is_input(i):
                self.input_name = name
            else:
                self.output_names.append(name)
        print(f"[TRT] 输入: {self.input_name}")
        print(f"[TRT] 输出: {[(n, self.bufs[n][2]) for n in self.output_names]}")

    def get_output(self, outs, name):
        """★ 按名字取输出（避免依赖顺序）"""
        idx = self.output_names.index(name)
        return outs[idx]

    def infer(self, x):
        """x: NCHW float32 -> list of output arrays"""
        import cuda_rt
        # 拷输入
        host, dev, shape = self.bufs[self.input_name]
        np.copyto(host, x.ravel())
        cuda_rt.memcpy_htod(dev, host)
        # 执行 —— TensorRT 8.x 的 bindings 要传 c_void_p 的数组
        order = [self.input_name] + self.output_names
        bindings = (ctypes.c_void_p * len(order))(
            *[self.bufs[n][1] for n in order])
        ok = self.context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("execute_v2 失败")
        # 拷输出
        outs = []
        cuda_rt.device_sync()
        for n in self.output_names:
            h, d, sh = self.bufs[n]
            cuda_rt.memcpy_dtoh(h, d)
            outs.append(h.reshape(sh).copy())
        return outs


# =====================================================================
# ★★ 归一化必须和模型内部的 transform 一致！
#     export_onnx.py 的 TRTWrapper 绕过了 GeneralizedRCNNTransform，
#     也就是 ONNX/engine 里"没有"归一化这一步，必须在这里补上。
#     ⚠️ ssdlite320_mobilenet_v3_large 用的**不是** ImageNet 的 0.485/0.229，
#        而是 Normalize(mean=0.5, std=0.5)，即等价于 2x-1。
#        实测（test_clip.avi 第21帧）: 参考模型 tools 0.96
#            (x-0.5)/0.5 → 0.96 ✅     只/255 → 0.89     ImageNet归一化 → 漏检 ❌
def preprocess(frame, size):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    chw = np.transpose(small, (2, 0, 1))[None]
    return (chw - 0.5) / 0.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default=CFG["onnx"])
    p.add_argument("--engine", default=CFG["engine"])
    p.add_argument("--video", default=None)
    p.add_argument("--device", type=int, default=-1,
                   help="-1=自动扫描 /dev/video*（默认）；也可指定 0/1/2...")
    p.add_argument("--list-cameras", action="store_true",
                   help="只打印摄像头诊断信息后退出")
    p.add_argument("--bench", type=int, default=0)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--score", type=float, default=CFG["score_thresh"])
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--print", dest="show_text", action="store_true",
                   help="无显示器：把每帧检测结果打成文本行（串口友好，纯 ASCII）")
    p.add_argument("--json", dest="json_out", action="store_true",
                   help="无显示器：每帧打一行 JSON（以后走串口就用这个格式）")
    p.add_argument("--frames", type=int, default=0,
                   help="跑多少帧后停；0=一直跑到 Ctrl+C")
    p.add_argument("--serial", default=None,
                   help="把结果发到这个串口，如 /dev/ttyGS0（micro-USB 口）"
                        "或 /dev/ttyTHS1（40 针排针 pin8/9）")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--hz", type=float, default=20.0, help="串口发送频率上限")
    p.add_argument("--save-dir", default=None,
                   help="把画好框的图存到这个目录（板子没显示器时用这个看结果）")
    p.add_argument("--save-every", type=int, default=1,
                   help="每几帧存一张图，默认 1")
    return p.parse_args()


def _try_open(node, verify=True):
    """开一个 V4L2 节点。verify=True 时还要真的读到一帧才算成功
    （元数据节点能 open 但永远读不到帧）
    依次尝试几种格式：MJPG 720p → YUYV 640x480 → 摄像头默认"""
    fmt_list = [("MJPG", 1280, 720), ("YUYV", 640, 480), (None, None, None)]
    for fourcc, w, h in fmt_list:
        cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        if fourcc:                   # Jetson 是 USB 口带宽小，MJPG 压缩流最稳
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if not verify:
            return cap
        for _ in range(10):          # 摄像头起流慢，多读几次
            ok, fr = cap.read()
            if ok and fr is not None:
                return cap
        cap.release()
        print(f"[INFO] {node} 用 {fourcc or '默认格式'} 读不到帧，换下一种...")
    return None


def open_camera(dev=-1):
    """dev<0: 自动扫描 /dev/video*，返回第一个真能出图的
    返回 (cap 或 None, 说明)"""
    import glob
    import re
    if dev >= 0:
        node = f"/dev/video{dev}"
        if os.path.exists(node):
            cap = _try_open(node)
            if cap is not None:
                return cap, node
        print(f"[ERROR] 打不开 {node}  （或打得开但读不到帧）")
        print("        详细原因: python3 cam_check.py")
        print("        想让程序自己找: python3 trt_infer.py")
        return None, node

    nodes = sorted(glob.glob("/dev/video*"),
                   key=lambda s: int(re.sub(r"\D", "", s) or 0))
    if not nodes:
        print("[ERROR] /dev/video* 一个都没有 → 是 USB 层没认到摄像头，不是 OpenCV 的问题")
        print("        ① lsusb | grep 045e    Kinect 枚举出来没有")
        print("        ② lsusb -t             插的是 USB3.0 蓝口还是 USB2.0 黑口")
        print("        ③ 看 Kinect 指示灯: 白=正常  橙/闪=供电不足")
        print("        ④ python3 cam_check.py 一次查全")
        return None, "auto"
    for node in nodes:
        cap = _try_open(node)
        if cap is not None:
            return cap, node
        print(f"[INFO] {node} 不可用，试下一个...")
    print("[ERROR] 所有 /dev/video* 都打不开 → python3 cam_check.py 看原因")
    return None, "auto"


def main():
    args = parse_args()

    if args.list_cameras:
        import cam_check
        argv_bak = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            return cam_check.main()
        finally:
            sys.argv = argv_bak

    # 板上没接显示器时自动关窗口，免得 cv2.imshow 直接报错
    if not os.environ.get("DISPLAY") and not args.no_show:
        print("[INFO] 没检测到 DISPLAY（无显示器），自动切无窗口模式")
        print("       要开窗口就在有桌面的终端里跑，或 export DISPLAY=:0")
        args.no_show = True

    if not os.path.exists(args.engine) or args.rebuild:
        if not os.path.exists(args.onnx):
            print(f"[ERROR] 找不到 {args.onnx}")
            return 1
        build_engine(args.onnx, args.engine, CFG["fp16"])

    print("[TRT] 加载 engine...")
    eng = Engine(args.engine)
    S = CFG["img_size"]
    ANCH = generate_anchors(S)

    if args.bench:
        x = np.random.randn(1, 3, S, S).astype(np.float32)
        for _ in range(5):
            eng.infer(x)
        t0 = time.time()
        for _ in range(args.bench):
            eng.infer(x)
        dt = (time.time() - t0) / args.bench
        print(f"\n[测速] {dt*1000:.1f} ms/帧  =  {1/dt:.1f} FPS")
        return 0

    src = args.video or f"/dev/video{args.device}"
    if args.video:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[ERROR] 打不开 {src}")
            return 1
    else:
        cap, src = open_camera(args.device)
        if cap is None:
            return 1
    print(f"[INFO] 输入 {src}")
    print("\n按键: q=退出  s=截图  空格=暂停  +/-=阈值\n")

    # ---- 串口输出（可选）----
    ser = None
    if args.serial:
        from serial_out import SerialOut

        def _mapper(x0, y0, x1, y1, cid, sc):
            c = CFG["classes"][cid]
            return {"cls": c, "cn": CFG["class_cn"][c],
                    "score": round(float(sc), 3),
                    "box": [int(x0), int(y0), int(x1), int(y1)],
                    "xyz": None, "angle": None}      # 深度好了直接填这里

        ser = SerialOut(args.serial, args.baud, args.hz)
        ser.set_mapper(_mapper)

    if not args.no_show:
        cv2.namedWindow("TRT", cv2.WINDOW_NORMAL)

    n = 0
    t_fps = time.time()
    t_start = time.time()
    fps = 0.0
    t_inf = 0.0
    paused = False
    score = args.score
    frame = None
    # 无显示器时，如果没说要打印什么，就跑 30 帧做个快速自检
    n_frames = args.frames
    if n_frames == 0 and args.no_show and not (args.show_text or args.json_out):
        n_frames = 30

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    break
                time.sleep(0.05)
                continue
            n += 1

        H, W = frame.shape[:2]
        x = preprocess(frame, S)
        t0 = time.time()
        out = eng.infer(x)
        t_inf = (time.time() - t0) * 1000

        # ★ 按名字取，不依赖顺序
        cls_l = eng.get_output(out, "cls_logits")[0]
        bbox_r = eng.get_output(out, "bbox_reg")[0]
        dets = postprocess(cls_l, bbox_r, ANCH, score,
                           CFG["nms_iou"], CFG["max_det"])
        # ★ postprocess 返回的是 320x320 输入尺度，必须放大回原图！
        sx, sy = W / float(S), H / float(S)
        dets = [(x0 * sx, y0 * sy, x1 * sx, y1 * sy, c, sc)
                for (x0, y0, x1, y1, c, sc) in dets]

        if ser is not None:
            ser.send(n, dets, t_inf)

        vis = frame.copy()
        for (x0, y0, x1, y1, cid, sc) in dets:
            col = CFG["colors"][cid % len(CFG["colors"])]
            p0 = (int(x0), int(y0)); p1 = (int(x1), int(y1))
            cv2.rectangle(vis, p0, p1, col, 3)
            cv2.putText(vis, f"{CFG['classes'][cid]} {sc:.2f}",
                        (p0[0], max(20, p0[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        if n % 10 == 0:
            fps = 10.0 / max(1e-6, time.time() - t_fps)
            t_fps = time.time()
        cv2.rectangle(vis, (0, 0), (W, 40), (32, 32, 32), -1)
        cv2.putText(vis, f"FPS {fps:4.1f}  infer {t_inf:5.1f}ms  "
                         f"thr {score:.2f}  obj {len(dets)}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2)

        if args.save_dir and n % max(1, args.save_every) == 0:
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir, exist_ok=True)
                print(f"[存图] 目录 {args.save_dir}")
            sp_ = os.path.join(args.save_dir, f"f{n:06d}.jpg")
            cv2.imwrite(sp_, vis)

        if args.show_text or args.json_out:
            items = [(CFG["classes"][c], float(sc), (int(x0), int(y0), int(x1), int(y1)))
                     for (x0, y0, x1, y1, c, sc) in dets]
            if args.json_out:
                import json
                rec = {"frame": n, "t": round(time.time() - t_start, 3),
                       "infer_ms": round(t_inf, 1), "w": W, "h": H,
                       "obj": [{"cls": c, "score": round(s, 3), "box": list(b)}
                               for (c, s, b) in items]}
                print(json.dumps(rec, ensure_ascii=False), flush=True)
            else:
                body = " | ".join(f"{c} {s:.2f} ({b[0]},{b[1]},{b[2]},{b[3]})"
                                  for (c, s, b) in items) or "-"
                print(f"[{n:06d}] {t_inf:5.1f}ms fps={fps:4.1f} obj={len(items)}  {body}",
                      flush=True)

        if not args.no_show:
            disp = vis
            if disp.shape[1] > 1280:
                s = 1280.0 / disp.shape[1]
                disp = cv2.resize(disp, None, fx=s, fy=s)
            cv2.imshow("TRT", disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('s'):
                p = f"/tmp/trt_{int(time.time())}.jpg"
                cv2.imwrite(p, vis); print(f"[SHOT] {p}")
            elif k == 32:
                paused = not paused
            elif k in (ord('+'), ord('=')):
                score = min(0.95, score + 0.05); print(f"[INFO] 阈值 {score:.2f}")
            elif k in (ord('-'), ord('_')):
                score = max(0.05, score - 0.05); print(f"[INFO] 阈值 {score:.2f}")
        else:
            if n_frames and n >= n_frames:
                break

    cap.release()
    if ser is not None:
        ser.close()
    if not args.no_show:
        cv2.destroyAllWindows()
    print(f"\n[DONE] {n} 帧，平均推理 {t_inf:.1f} ms ({1000/max(1e-6,t_inf):.0f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
