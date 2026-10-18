#!/usr/bin/env python3
"""板子↔电脑 TCP 协议的回归测试（不需要相机、不需要 ROS）。

为什么值得单独写一个：
    这个协议是"裸字节流"—— 一旦收发两边差一个字节，不会报错，
    只会**整体错位**，表现成"偶尔收到奇怪的东西然后断开"，
    极难定位。这次就踩了一次：
      发送端多写了一个 4 字节的 jpg 长度字段，接收端没读，
      结果每次都在 JPEG 中间错开 4 字节，连上 1 秒就断。
    所以：改协议之后，先跑这个。

它做的事：
    1. 起一个本地假服务器，按【接收端】的规则解析；
    2. 用【发送端】真代码发 3 帧（带合成图）；
    3. 校验 JPEG 头 ff d8 / 尾 ff d9、长度自洽、下一帧 magic 位置正确。

用法:
    python3 tools/test_protocol.py
"""
import json
import os
import socket
import struct
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "common"))
from ros_sender import RosSender                       # noqa: E402

MAGIC = b"K4PC"
PORT = 9199


def read_exact(sock, n):
    """TCP 是流，必须循环读满 n 字节（和接收端 cloud_bridge 一致）。"""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def server(results, n_frames):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(1)
    conn, _ = srv.accept()
    # 一次性把 3 帧的字节全读进来，再逐帧切 —— 这样"多出来/少掉"的
    # 字节会立刻暴露成 magic 对不上，而不是被下一帧悄悄吸收。
    for i in range(n_frames):
        m = read_exact(conn, 4)
        hl = struct.unpack("<I", read_exact(conn, 4))[0]
        hb = read_exact(conn, hl)
        head = json.loads(hb.decode("utf-8"))
        n = int(head["n"])
        pts = read_exact(conn, n * 12)
        jn = int(head.get("jpg_bytes", 0))
        jpg = read_exact(conn, jn) if jn else b""
        results.append({
            "i": i, "magic": m, "hlen_ok": len(hb) == hl, "n": n,
            "jn": jn, "jpg_len": len(jpg),
            "jpg_soi": jpg[:2] == b"\xff\xd8",
            "jpg_eoi": jpg[-2:] == b"\xff\xd9",
            "pts_len_ok": len(pts or b"") == n * 12,
        })
    conn.close()
    srv.close()


def main():
    n_frames = 3
    results = []
    th = threading.Thread(target=server, args=(results, n_frames), daemon=True)
    th.start()
    time.sleep(0.3)

    img = np.zeros((720, 1280, 3), np.uint8)
    img[:, :, 1] = 200
    img[100:300, 400:900, 2] = 255
    s = RosSender("127.0.0.1", PORT, max_points=100000, hz=2, verbose=False)
    for i in range(n_frames):
        pts = np.random.RandomState(i).rand(5000, 3).astype(np.float32)
        s.send(pts, [], {}, (1280, 720), image=img)
        time.sleep(0.7)
    time.sleep(1.0)
    s.close()
    th.join(timeout=5)

    ok = True
    for r in results:
        good = (r["magic"] == MAGIC and r["hlen_ok"] and r["pts_len_ok"]
                and r["jn"] == r["jpg_len"] and r["jpg_soi"] and r["jpg_eoi"])
        ok = ok and good
        print(f"帧{r['i']}: magic={'OK' if r['magic']==MAGIC else r['magic']!r} "
              f"头长自洽={r['hlen_ok']} 点数={r['n']} 点字节自洽={r['pts_len_ok']} "
              f"jpg声明={r['jn']} 实收={r['jpg_len']} "
              f"头={('OK' if r['jpg_soi'] else 'BAD')} "
              f"尾={('OK' if r['jpg_eoi'] else 'BAD')}")
    if len(results) != n_frames:
        print(f"❌ 只收到 {len(results)}/{n_frames} 帧"); ok = False
    print("\n" + ("✅ 协议自洽" if ok else
                  "❌ 协议错位！收发两边的字段定义对不上，去查 ros_sender 的 pkt "
                  "和 cloud_bridge 的 _loop"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
