#!/usr/bin/env python3
"""ttyUSB0 回环自检（宽容版）：换波特率 + 反复写，排除接触不良。

用法（在板子上）: python3 - < diag_loopback.py
"""
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"

for baud in (9600, 115200):
    print(f"\n===== baud={baud} =====")
    try:
        ser = serial.Serial(PORT, baud, timeout=1.0)
    except Exception as e:
        print(f"❌ 打不开: {e}")
        continue

    ser.reset_input_buffer()
    time.sleep(0.3)
    idle = ser.read(64)
    print(f"空闲: {len(idle)} 字节 {idle[:24]!r}")

    got = b""
    for i in range(20):
        msg = f"#{i:03d}:".encode() + b"U" * 30 + b"\n"
        n = ser.write(msg)
        ser.flush()
        time.sleep(0.05)
        got += ser.read(4096)
    ser.close()

    print(f"写了 20 次，共 {20 * 38} 字节；读回 {len(got)} 字节  {got[:60]!r}")
    if len(got) >= 300 and got.count(b"U") >= 300:
        print("✅ 回环正常 —— TXD 和 RXD 确实短接着")
    elif got.count(b"\x00") >= 30:
        print("❌ 大量 0x00 —— RX 被拉低，短接到 GND 了")
    elif got.count(b"\xff") >= 30:
        print("❌ 大量 0xFF —— RX 被拉高，短接到电源针了")
    elif len(got) == 0:
        print("❌ 全程 0 字节 —— TXD/RXD 之间没通")
    else:
        print(f"⚠️ 读回 {len(got)} 字节但对不上原文")
