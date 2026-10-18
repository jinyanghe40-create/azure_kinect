#!/usr/bin/env python3
"""
采集工具 v2 —— 录制方块视频片段，用于后续训练分类模型。

v2 修了两个坑:
  1. 按键改从「终端」直接读（termios 非规范模式），不再依赖 OpenCV 窗口焦点。
     上一版按空格没反应，就是因为焦点在终端上，cv2.waitKey 收不到。
  2. 录制中实时打印帧数/时长，终端里能看到进度。

用法:
    python3 capture_session.py --cls tools
    python3 capture_session.py --cls tools --distance 1.0 --lighting day --pose front
    python3 capture_session.py --list

按键（在跑脚本的那个终端里按）:
    SPACE  开始 / 停止录制
    s      截图一张（存到 shots/）
    q      退出
"""
import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime

import cv2

ROOT_DEFAULT = os.path.expanduser("~/kinect/dataset")
KNOWN_CLASSES = ["tools", "medicine", "instrument", "food", "block_E", "block_F"]
LIGHTING = ["day", "lamp", "shadow", "dark", "unknown"]
POSES = ["front", "pitch_up", "pitch_down", "yaw_left", "yaw_right", "roll", "unknown"]


# ---------------------------------------------------------------- 终端按键
class TermKeys:
    """从终端读原始按键，不依赖 GUI 窗口焦点。"""

    def __init__(self):
        self.saved = None
        self.enabled = False

    def __enter__(self):
        if sys.stdin.isatty():
            try:
                self.saved = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                self.enabled = True
            except Exception as e:
                print(f"[WARN] 无法设置终端原始模式（{e}），按键可能不灵")
        else:
            print("[WARN] stdin 不是终端，按键将不可用")
        return self

    def __exit__(self, *exc):
        self.restore()

    def restore(self):
        if self.enabled and self.saved is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.saved)
            except Exception:
                pass
            self.enabled = False

    def poll(self):
        """非阻塞读一个字符；没有按键返回 None。"""
        if not self.enabled:
            return None
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0)
        except Exception:
            return None
        if not r:
            return None
        try:
            ch = sys.stdin.read(1)
        except Exception:
            return None
        return ch if ch else None


# ---------------------------------------------------------------- 参数
def parse_args():
    p = argparse.ArgumentParser(description="采集方块视频片段")
    p.add_argument("--cls", default=None,
                   help=f"类别名，例如 tools。建议: {', '.join(KNOWN_CLASSES)}")
    p.add_argument("--root", default=ROOT_DEFAULT, help="数据集根目录")
    p.add_argument("--device", type=int, default=2, help="V4L2 设备号（Kinect 彩色是 2）")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--fourcc", default="MJPG", help="MJPG(默认) 或 XVID")
    p.add_argument("--distance", type=float, default=None, help="相机到方块的距离（米）")
    p.add_argument("--lighting", default="unknown", choices=LIGHTING)
    p.add_argument("--pose", default="unknown", choices=POSES)
    p.add_argument("--operator", default=os.environ.get("USER", "unknown"))
    p.add_argument("--no-preview", action="store_true", help="不显示预览窗口")
    p.add_argument("--list", action="store_true", help="只列出已有会话，不开相机")
    return p.parse_args()


# ---------------------------------------------------------------- 会话列表
def list_sessions(root):
    if not os.path.isdir(root):
        print(f"[INFO] 数据集目录还不存在: {root}")
        return
    rows = []
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in sorted(os.listdir(cls_dir)):
            if not f.endswith(".json"):
                continue
            try:
                meta = json.load(open(os.path.join(cls_dir, f)))
            except Exception:
                continue
            rows.append((cls, meta.get("session"), meta.get("frames"),
                         meta.get("distance_m"), meta.get("lighting"),
                         meta.get("pose"), meta.get("started_at")))
    if not rows:
        print(f"[INFO] {root} 下还没有会话")
        return
    print(f"{'类别':<10} {'会话':<8} {'帧数':>6} {'距离':>6} {'光照':<8} "
          f"{'姿态':<12} 时间")
    for r in rows:
        print(f"{r[0]:<10} {str(r[1]):<8} {str(r[2]):>6} {str(r[3]):>6} "
              f"{str(r[4]):<8} {str(r[5]):<12} {r[6]}")
    total = sum(r[2] or 0 for r in rows)
    print(f"\n合计 {len(rows)} 个会话，{total} 帧")


def next_session_id(cls_dir, ext):
    n = 1
    while os.path.exists(os.path.join(cls_dir, f"session_{n:03d}.{ext}")):
        n += 1
    return n


def has_gui():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# ---------------------------------------------------------------- 主流程
def main():
    args = parse_args()

    if args.list:
        list_sessions(args.root)
        return 0

    if not args.cls:
        print("[ERROR] 需要 --cls 指定类别，例如: python3 capture_session.py --cls tools")
        print(f"        推荐类别名: {', '.join(KNOWN_CLASSES)}")
        return 2

    cls_dir = os.path.join(args.root, args.cls)
    os.makedirs(cls_dir, exist_ok=True)

    device_path = f"/dev/video{args.device}"
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] 打不开 {device_path}")
        print("        检查: v4l2-ctl --list-devices")
        return 1

    # MJPG 必须在设分辨率之前设
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"[ERROR] {device_path} 能打开但读不到帧")
        cap.release()
        return 1
    real_h, real_w = frame.shape[:2]
    if (real_w, real_h) != (args.width, args.height):
        print(f"[WARN] 请求 {args.width}x{args.height}，实际 {real_w}x{real_h}")

    preview = has_gui() and not args.no_preview
    if not preview:
        print("[INFO] 不显示预览窗口（--no-preview 或无图形界面）")

    print(f"[INFO] 设备 {device_path}   实际分辨率 {real_w}x{real_h}")
    print(f"[INFO] 类别: {args.cls}   保存到: {cls_dir}")
    print("\n>>> 在【这个终端】里按键:  SPACE=开始/停止录制   s=截图   q=退出 <<<\n")

    writer = None
    recording = False
    frame_idx = 0
    session_id = None
    session_start = None
    out_path = None
    ext = "avi"
    last_report = 0.0

    keys = TermKeys()
    try:
        keys.__enter__()
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] 读到空帧")
                time.sleep(0.05)
                continue

            if recording and writer is not None:
                writer.write(frame)
                frame_idx += 1
                now = time.time()
                if now - last_report >= 1.0:      # 每秒报一次进度
                    el = now - session_start.timestamp()
                    print(f"\r[REC] {frame_idx} 帧  {el:5.1f}s  "
                          f"{frame_idx/max(el,0.001):4.1f} FPS   ", end="", flush=True)
                    last_report = now

            if preview:
                hud = frame.copy()
                status = f"REC {frame_idx}" if recording else "IDLE"
                color = (0, 0, 255) if recording else (0, 200, 0)
                cv2.putText(hud, f"{status}  {args.cls}", (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
                cv2.putText(hud, "keys in TERMINAL: SPACE rec | s shot | q quit",
                            (16, hud.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)
                cv2.imshow("capture", hud)
                cv2.waitKey(1)                     # 只用于刷新窗口，不读按键

            ch = keys.poll()

            if ch in ('q', 'Q', '\x03'):           # q 或 Ctrl-C
                break

            elif ch == ' ':                         # 开始 / 停止录制
                if not recording:
                    if session_id is None:
                        session_id = next_session_id(cls_dir, ext)
                    out_name = f"session_{session_id:03d}.{ext}"
                    out_path = os.path.join(cls_dir, out_name)
                    writer = cv2.VideoWriter(out_path,
                                             cv2.VideoWriter_fourcc(*args.fourcc),
                                             args.fps, (real_w, real_h))
                    if not writer.isOpened():
                        print(f"\n[ERROR] 无法创建 {out_path}（换 --fourcc XVID 试试）")
                        return 1
                    recording = True
                    frame_idx = 0
                    last_report = time.time()
                    session_start = datetime.now()
                    print(f"\n[REC] 开始录制 -> {out_name}"
                          f"   (再按一次空格停止)")
                else:
                    recording = False
                    writer.release()
                    writer = None
                    if frame_idx == 0:
                        print("\n[WARN] 这段是 0 帧，已删除，不写元数据")
                        try:
                            os.remove(out_path)
                        except OSError:
                            pass
                        session_id = None
                        continue
                    meta = {
                        "class": args.cls,
                        "session": f"{session_id:03d}",
                        "file": os.path.basename(out_path),
                        "frames": frame_idx,
                        "fps": args.fps,
                        "resolution": [real_w, real_h],
                        "fourcc": args.fourcc,
                        "distance_m": args.distance,
                        "lighting": args.lighting,
                        "pose": args.pose,
                        "operator": args.operator,
                        "started_at": session_start.isoformat(timespec="seconds"),
                        "duration_s": round(
                            (datetime.now() - session_start).total_seconds(), 2),
                        "depth_available": False,
                    }
                    json_path = os.path.join(cls_dir, f"session_{session_id:03d}.json")
                    with open(json_path, "w") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    print(f"\n[STOP] {frame_idx} 帧已保存 -> {os.path.basename(out_path)}"
                          f"   元数据 -> {os.path.basename(json_path)}")
                    session_id = None

            elif ch == 's':
                shots_dir = os.path.join(cls_dir, "shots")
                os.makedirs(shots_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                shot_path = os.path.join(shots_dir, f"shot_{ts}.jpg")
                cv2.imwrite(shot_path, frame)
                print(f"\n[SHOT] {shot_path}")

    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl-C")
    finally:
        keys.restore()

    if recording and writer is not None:
        writer.release()
        print(f"\n[WARN] 退出时正在录制，{frame_idx} 帧已保存到 {out_path}")

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    n_sessions = len([f for f in os.listdir(cls_dir) if f.endswith(".json")])
    print(f"\n[SUMMARY] {args.cls}: 共 {n_sessions} 个会话")
    print("[NEXT] 下一步: 抽帧后用 labelImg 或 CVAT 标注")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
