#!/usr/bin/env python3
"""
实时检测界面 —— 摄像头 + 框 + 类别

用你训好的 SSDlite(MobileNet 骨干) 检测器，对着摄像头实时画框。

用法:
    python3 live_detect.py                       # 用 Azure Kinect 彩色流
    python3 live_detect.py --device 0            # 用内置摄像头
    python3 live_detect.py --video xx.avi        # 跑视频文件（不用摄像头）

按键（点在窗口上再按）:
    q / ESC   退出
    s         截图到 /tmp/live_shots/
    space     暂停 / 继续
    + / -     调置信度阈值
    m         切换「只显示高置信」/「显示全部」
    h         显示/隐藏帮助

所有参数在下面【配置区】。
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch

# =====================================================================
#                            配 置 区
# =====================================================================
CFG = {
    "weights": os.path.expanduser("~/kinect/runs/det/best.pt"),
    "device": 2,                # /dev/video2 = Azure Kinect 彩色
    "width": 1280,
    "height": 720,
    "fps": 30,
    "fourcc": "MJPG",

    "score_thresh": 0.4,        # 置信度阈值（按 +/- 可实时调）
    "nms_iou": 0.45,
    "max_det": 20,              # 每帧最多显示几个框

    # 类别名（要和训练时一致），以及显示用的中文
    "classes": ["tools", "medicine", "instrument", "food"],
    "class_cn": {"tools": "工具", "medicine": "药品",
                 "instrument": "仪器", "food": "食品"},
    # 每类的框颜色 (BGR)
    "colors": [(60, 76, 231), (60, 200, 60), (240, 140, 60), (60, 210, 210)],

    "show_fps": True,
    "window_max_width": 1280,   # 窗口最大宽度（超了就缩小显示）
}

SHOT_DIR = "/tmp/live_shots"

# 中文字体（找不到就退回英文）
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class TextDrawer:
    """优先用 PIL 画中文；没字体就退回 cv2 画英文。"""

    def __init__(self, font_path, size=22):
        self.font = None
        self.ok = False
        if _HAS_PIL and font_path:
            try:
                self.font = ImageFont.truetype(font_path, size)
                self.ok = True
            except Exception as e:
                print(f"[WARN] 字体加载失败: {e}")

    def draw(self, bgr, items):
        """items: [(text, (x,y), (b,g,r)), ...]"""
        if not self.ok:
            for text, (x, y), col in items:
                cv2.putText(bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, col, 2)
            return bgr
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for text, (x, y), col in items:
            d.text((x, y), text, font=self.font,
                   fill=(col[2], col[1], col[0]))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# =====================================================================
#                            模型
# =====================================================================
def load_model(weights, device):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_detector import CFG as TCFG, build_detector

    ck = torch.load(weights, map_location=device, weights_only=False)
    classes = ck.get("classes", TCFG["classes"])
    tcfg = ck.get("cfg", TCFG)
    model = build_detector(tcfg, len(classes)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    img_size = tcfg.get("img_size", 320)
    print(f"[模型] {os.path.basename(weights)}")
    print(f"       F1={ck.get('f1', 0):.1%}  第 {ck.get('epoch','?')} 轮")
    print(f"       类别: {classes}   输入 {img_size}x{img_size}")
    return model, classes, img_size


@torch.no_grad()
def infer(model, frame_bgr, img_size, device, score_thresh, nms_iou, max_det):
    """返回 [(x0,y0,x1,y1, cls_id, score), ...]（原图坐标系）"""
    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (img_size, img_size))
    t = torch.from_numpy(small).permute(2, 0, 1).float().div(255.0)
    out = model([t.to(device)])[0]

    b = out["boxes"].cpu().numpy()
    s = out["scores"].cpu().numpy()
    l = out["labels"].cpu().numpy().astype(int) - 1

    keep = s >= score_thresh
    b, s, l = b[keep], s[keep], l[keep]
    if len(b) == 0:
        return []

    # 逐类 NMS
    final = []
    for c in np.unique(l):
        idx = np.where(l == c)[0]
        order = idx[np.argsort(-s[idx])]
        boxes = b[order]
        while len(order):
            final.append(order[0])
            if len(order) == 1:
                break
            x0 = np.maximum(boxes[0, 0], boxes[1:, 0])
            y0 = np.maximum(boxes[0, 1], boxes[1:, 1])
            x1 = np.minimum(boxes[0, 2], boxes[1:, 2])
            y1 = np.minimum(boxes[0, 3], boxes[1:, 3])
            iw = np.clip(x1 - x0, 0, None)
            ih = np.clip(y1 - y0, 0, None)
            inter = iw * ih
            a1 = (boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1])
            a2 = ((boxes[1:, 2] - boxes[1:, 0]) *
                  (boxes[1:, 3] - boxes[1:, 1]))
            iou = inter / np.maximum(a1 + a2 - inter, 1e-9)
            order = order[1:][iou < nms_iou]
            boxes = b[order]
    if not final:
        return []
    final = np.array(final)[np.argsort(-s[final])][:max_det]

    sx, sy = W / img_size, H / img_size
    res = []
    for i in final:
        x0, y0, x1, y1 = b[i]
        res.append((x0 * sx, y0 * sy, x1 * sx, y1 * sy, int(l[i]), float(s[i])))
    return res


# =====================================================================
#                            主流程
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="实时检测界面")
    p.add_argument("--device", type=int, default=CFG["device"])
    p.add_argument("--video", default=None, help="用视频文件代替摄像头")
    p.add_argument("--weights", default=CFG["weights"])
    p.add_argument("--score", type=float, default=CFG["score_thresh"])
    p.add_argument("--width", type=int, default=CFG["width"])
    p.add_argument("--height", type=int, default=CFG["height"])
    p.add_argument("--cpu", action="store_true", help="强制用 CPU")
    p.add_argument("--no-show", action="store_true",
                   help="不开窗口（无显示器时测性能）")
    p.add_argument("--frames", type=int, default=0,
                   help="处理这么多帧后自动退出（0=不限）")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(SHOT_DIR, exist_ok=True)

    if not os.path.exists(args.weights):
        print(f"[ERROR] 找不到权重: {args.weights}")
        print("        先训练: python train_detector.py")
        return 1

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    print(f"[设备] {device}", end="")
    if device.type == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print("  ⚠️ 用 CPU 会很慢，建议用 GPU")

    model, classes, img_size = load_model(args.weights, device)

    # 中文绘制
    font_path = find_font()
    if font_path:
        print(f"[字体] {font_path}")
    else:
        print("[警告] 没找到中文字体，标签将显示英文类别名")
    drawer = TextDrawer(font_path, size=max(18, args.width // 55))

    # 视频源
    if args.video:
        cap = cv2.VideoCapture(args.video)
        src_name = args.video
        is_cam = False
    else:
        dev = f"/dev/video{args.device}"
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        is_cam = True
        src_name = dev
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CFG["fourcc"]))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, CFG["fps"])

    if not cap.isOpened():
        print(f"[ERROR] 打不开 {src_name}")
        if is_cam:
            print("        检查 v4l2-ctl --list-devices")
        return 1
    ok, f0 = cap.read()
    if not ok:
        print(f"[ERROR] {src_name} 读不到帧")
        return 1
    H, W = f0.shape[:2]
    print(f"[输入] {src_name}  {W}x{H}")

    print(f"\n按键: q退出  s截图  space暂停  +/-阈值  m切换  h帮助")
    print(f"置信度阈值: {args.score:.2f}\n")

    show_help = False
    paused = False
    score = args.score
    show_all = False
    n_frame = 0
    n_infer = 0
    t_last = time.time()
    fps_disp = 0.0
    fps_infer = 0.0
    t_infer_start = time.time()
    infer_ms = 0.0

    if not args.no_show:
        cv2.namedWindow("live detect", cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    print("[INFO] 视频结束")
                    break
                print("[WARN] 读到空帧")
                time.sleep(0.05)
                continue
            n_frame += 1
        else:
            frame = last_frame
            time.sleep(0.03)

        last_frame = frame

        # ---- 推理 ----
        th = 0.0 if show_all else score
        t0 = time.time()
        dets = infer(model, frame, img_size, device, th, CFG["nms_iou"],
                     CFG["max_det"])
        infer_ms = (time.time() - t0) * 1000
        n_infer += 1

        # ---- 绘制 ----
        vis = frame.copy()
        labels = []
        for (x0, y0, x1, y1, cid, sc) in dets:
            col = CFG["colors"][cid % len(CFG["colors"])]
            x0i, y0i, x1i, y1i = map(int, (x0, y0, x1, y1))
            cv2.rectangle(vis, (x0i, y0i), (x1i, y1i), col, 3)

            en = classes[cid] if cid < len(classes) else f"cls{cid}"
            cn = CFG["class_cn"].get(en, "")
            text = f"{cn} {sc:.2f}" if cn else f"{en} {sc:.2f}"
            labels.append((text, (x0i + 4, max(4, y0i - 30)), col))

            # 角标线，让框更醒目
            L = max(12, (x1i - x0i) // 8)
            for (ax, ay, dx, dy) in ((x0i, y0i, 1, 1), (x1i, y0i, -1, 1),
                                     (x0i, y1i, 1, -1), (x1i, y1i, -1, -1)):
                cv2.line(vis, (ax, ay), (ax + dx * L, ay), col, 5)
                cv2.line(vis, (ax, ay), (ax, ay + dy * L), col, 5)

        if labels:
            vis = drawer.draw(vis, labels)

        # ---- 状态栏 ----
        now = time.time()
        if now - t_last >= 0.5:
            fps_disp = n_frame / max(1e-6, now - t_last) if not paused else 0
            t_last = now
            n_frame = 0
        bar_items = []
        info = (f"FPS {fps_disp:4.1f}   推理 {infer_ms:5.1f}ms   "
                f"阈值 {score:.2f}   目标 {len(dets)}")
        cv2.rectangle(vis, (0, 0), (W, 42), (32, 32, 32), -1)
        cv2.putText(vis, info, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (80, 255, 80), 2)
        if paused:
            cv2.putText(vis, "PAUSED", (W - 160, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 200, 255), 2)

        if show_help:
            lines = ["q/ESC quit", "s screenshot", "space pause",
                     "+/- threshold", "m show all", "h hide help"]
            y = 70
            for ln in lines:
                cv2.putText(vis, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (220, 220, 220), 1)
                y += 26

        if args.no_show:
            if args.frames and n_infer >= args.frames:
                break
            continue

        # ---- 显示 ----
        disp = vis
        if disp.shape[1] > CFG["window_max_width"]:
            sc = CFG["window_max_width"] / disp.shape[1]
            disp = cv2.resize(disp, None, fx=sc, fy=sc)
        cv2.imshow("live detect", disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            p = os.path.join(SHOT_DIR, f"live_{ts}.jpg")
            cv2.imwrite(p, vis)
            print(f"[SHOT] {p}")
        elif key == ord(' '):
            paused = not paused
            print(f"[INFO] {'暂停' if paused else '继续'}")
        elif key in (ord('+'), ord('=')):
            score = min(0.95, score + 0.05)
            print(f"[INFO] 阈值 -> {score:.2f}")
        elif key in (ord('-'), ord('_')):
            score = max(0.05, score - 0.05)
            print(f"[INFO] 阈值 -> {score:.2f}")
        elif key == ord('m'):
            show_all = not show_all
            print(f"[INFO] {'显示全部（含低置信）' if show_all else '只看高置信'}")
        elif key == ord('h'):
            show_help = not show_help

        if args.frames and n_infer >= args.frames:
            break

    cap.release()
    if not args.no_show:
        cv2.destroyAllWindows()
    print(f"\n[DONE] 共处理 {n_infer} 帧，平均推理 {infer_ms:.1f} ms "
          f"({1000/max(1e-6,infer_ms):.0f} FPS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
