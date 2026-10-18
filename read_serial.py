#!/usr/bin/env python3
"""
电脑端：读板子从串口发来的检测结果（JSON Lines），翻译成人看得懂的一行行。

板子把结果写到 /dev/ttyGS0（micro-USB 口），电脑这边就多出一个 /dev/ttyACM0，
本脚本读它。

用法:
    python3 read_serial.py                    # 自动找串口，人读格式
    python3 read_serial.py --port /dev/ttyACM0
    python3 read_serial.py --raw              # 只打原始 JSON（别翻译）
    python3 read_serial.py --log out.jsonl    # 同时存盘
    python3 read_serial.py --check --seconds 20   # ★ 自动判 PASS/FAIL

★ --check 是给"验收"用的，不要用眼睛看。它断言四件事：
    ① 每一行都以 { 开头  —— 出现别的 = getty/内核日志掺进来了（污染）
    ② 行频 ~2~3 行/秒     —— 对上真实帧率（推理+几何只有 ~2.2 FPS）
    ③ seq 单调不减        —— 没乱序/没重连丢包
    ④ face 是 3 个浮点数、不是 null
       —— null 说明几何没配上，电控会收到没有坐标的"空对象"

★ 关于单位：板子上如果带了 --mm，长度字段是【毫米】；头里的 obj_unit
  会写明（"mm"/"m"）。没有这个字段就只能猜，所以本脚本会把两种情况都标出来。
"""
import argparse
import glob
import json
import sys
import time


def find_port():
    for pat in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        c = sorted(glob.glob(pat))
        if c:
            return c[0]
    return None


def _vec(v, unit):
    """把 [x,y,z] 打成人读的字符串。带单位就标单位。"""
    if v is None:
        return "None"
    u = "" if unit in (None, "?") else unit
    try:
        return "(" + ",".join(f"{float(x):.1f}" if u == "mm" else f"{float(x):.3f}"
                              for x in v) + ")" + u
    except Exception:
        return str(v)


def fmt_obj(o, unit):
    """★ 主数据是 face + normal（平面拟合直接测到的）；
       cube 是推出来的，能不能信看 cube_ok。别只打 cls。"""
    cls = o.get("cls") or o.get("cn") or "?"
    cid = o.get("cls_id")
    s = f"{cls}"
    if cid is not None:
        s += f"#{cid}"
    try:
        s += f" {float(o.get('score') or 0):.2f}"
    except Exception:
        pass
    s += f" face={_vec(o.get('face'), unit)}"
    if o.get("range") is not None:
        s += f" r={o['range']}"
    # ★ cube_ok=False 时体心是估的，直接标出来，别让电控误用
    cok = o.get("cube_ok")
    if cok is not None:
        s += f" cube_ok={'T' if cok else 'F'}"
    if o.get("edge") is not None:
        s += f" edge={o['edge']}({o.get('edge_src') or '?'})"
    if o.get("n_faces") is not None:
        s += f" nf={o['n_faces']}"
    if o.get("lean") is not None:
        # ★ 法向偏离【车体竖直】的角度：0°=铅垂面，90°=水平面。
        #   和 angle（vs 相机光轴）不是一回事 —— 那个混着相机装的俯仰。
        s += f" lean={o['lean']}°"
    if o.get("rms_mm") is not None:
        s += f" rms={o['rms_mm']}mm"
    if o.get("quality"):
        s += f" q={o['quality']}"
    return s


def run_check(ser, seconds, log):
    """跑一段时间，自动判 PASS/FAIL。返回 0=PASS 1=FAIL。"""
    t0 = time.time()
    n_lines = n_obj = 0
    n_bad = 0            # 不是以 { 开头的行（污染）
    n_nojson = 0         # 以 { 开头但 json 解析失败
    n_face_null = 0      # face 是 null
    n_face_bad = 0       # face 不是 3 个数
    n_ok = 0             # 成功解析的行数
    n_no_coord = 0       # 没有 coord 头（= rev1 或没走 run_azure.py）
    n_body = 0           # coord == "body"（车体系，正确）
    n_uncal = 0          # coord == "camera"（没装标定，别接电控）
    coords_seen = set()  # 其它取值
    seqs = []
    junk = []
    print(f"[check] 采样 {seconds}s …")
    while time.time() - t0 < seconds:
        line = ser.readline()
        if not line:
            continue
        if log:
            log.write(line.decode(errors="replace"))
            log.flush()
        txt = line.decode(errors="replace").rstrip("\r\n")
        if not txt:
            continue
        n_lines += 1
        if not txt.lstrip().startswith("{"):
            # ★ 这就是 getty / 内核日志污染的直接证据
            n_bad += 1
            if len(junk) < 5:
                junk.append(txt[:70])
            continue
        try:
            r = json.loads(txt)
        except Exception:
            n_nojson += 1
            continue
        if "seq" in r:
            seqs.append(r["seq"])
        # ★★ 坐标系头：rev2 起必须有。没有它 = 对端在读【旧协议】，
        #   而旧协议的 face 是相机系 —— 同一个数组含义完全不同。
        n_ok += 1
        if r.get("coord") is None:
            n_no_coord += 1
        elif r.get("coord") == "body":
            n_body += 1
        elif r.get("coord") == "camera":
            n_uncal += 1
        else:
            coords_seen.add(str(r.get("coord")))
        for o in r.get("obj", []):
            n_obj += 1
            f = o.get("face")
            if f is None:
                n_face_null += 1
            elif not (isinstance(f, (list, tuple)) and len(f) == 3 and
                      all(isinstance(x, (int, float)) for x in f)):
                n_face_bad += 1

    el = max(1e-6, time.time() - t0)
    hz = n_lines / el
    print()
    print("──────── 结果 ────────")
    print(f"  行数        {n_lines}   频率 {hz:.2f} 行/秒")
    print(f"  对象总数    {n_obj}")
    print(f"  非 JSON 行  {n_bad}")
    if junk:
        for j in junk:
            print(f"      ! {j!r}")
    print(f"  JSON 坏了   {n_nojson}")
    print(f"  face=null   {n_face_null}")
    print(f"  face 畸形   {n_face_bad}")
    print(f"  坐标头      body={n_body}  camera={n_uncal}  缺={n_no_coord}"
          + (f"  未知={sorted(coords_seen)}" if coords_seen else ""))
    mono = all(b >= a for a, b in zip(seqs, seqs[1:])) if len(seqs) > 1 else True
    print(f"  seq 单调    {'是' if mono else '否'}  ("
          + (f"{seqs[0]}→{seqs[-1]}，{len(seqs)} 个" if seqs else "没收到 seq") + ")")

    fails = []
    if n_lines == 0:
        fails.append("一行都没收到 —— 串口没通/波特率/对端没在发")
    if n_bad:
        fails.append(f"有 {n_bad} 行不是 JSON —— getty 或内核日志掺进来了！"
                     " 跑 sudo bash ~/work/azure_kinect/ttygs_udev_setup.sh")
    if n_nojson:
        fails.append(f"有 {n_nojson} 行自称 JSON 但解析失败（可能被截断）")
    if n_lines and not (0.5 <= hz <= 30):
        fails.append(f"行频 {hz:.2f}/秒 不正常（真实帧率约 2~3，"
                     "上限 --hz 10）")
    if not mono and seqs:
        fails.append("seq 不单调 —— 有乱序或重连丢包")
    if n_face_null and n_face_null >= max(1, n_obj):
        fails.append(f"face 全是 null（{n_face_null}/{n_obj}）——"
                     " 几何没配上，电控会收到没有坐标的对象")
    # ★★ rev2 的硬要求
    if n_ok and n_no_coord:
        fails.append(f"{n_no_coord}/{n_ok} 行没有 coord 头 —— 对端在发 rev1 或"
                     " 没走 run_azure.py，电控分不清 face 是哪个系的坐标")
    if n_uncal:
        fails.append(f"{n_uncal}/{n_ok} 行 coord=\"camera\" —— 板子没装车体标定"
                     "（--allow-uncalibrated），坐标【不是】车体系，别接电控")
    if coords_seen:
        fails.append(f"coord 出现未知取值 {sorted(coords_seen)}")

    print()
    if fails:
        print("❌ FAIL")
        for f in fails:
            print(f"   · {f}")
        return 1
    print("✅ PASS（只有 JSON、行频正常、seq 单调、face 有值、coord=body）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--log", default=None)
    ap.add_argument("--check", action="store_true", help="自动验收，判 PASS/FAIL")
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()

    port = a.port or find_port()
    if not port:
        print("❌ 没找到串口。查一下：")
        print("   ① 板子 micro-USB 线插到电脑了吗")
        print("   ② ls /dev/ttyACM* /dev/ttyUSB*")
        print("   ③ 电脑在 dialout 组吗: id -Gn | grep dialout")
        return 1

    try:
        import serial
    except ImportError:
        print("❌ 电脑上没装 pyserial：pip install pyserial")
        return 1

    ser = serial.Serial(port, a.baud, timeout=1.0)
    log = open(a.log, "w") if a.log else None
    print(f"✅ 已打开 {port} @ {a.baud}")

    if a.check:
        try:
            return run_check(ser, a.seconds, log)
        except KeyboardInterrupt:
            return 1
        finally:
            ser.close()
            if log:
                log.close()

    print("   (Ctrl+C 退出)")
    print(f"{'时间':>8} {'帧':>6} {'seq':>6} {'单位':>4} {'坐标':>7} {'对象数':>5}  内容")
    print("-" * 110)
    n = 0
    t0 = time.time()
    try:
        while True:
            line = ser.readline()
            if not line:
                continue
            if log:
                log.write(line.decode(errors="replace"))
                log.flush()
            if a.raw:
                sys.stdout.write(line.decode(errors="replace"))
                continue
            try:
                r = json.loads(line)
            except Exception:
                # ★ 不静默：这条几乎一定是 getty/内核污染，要显眼
                print(f"  ⚠️ 非 JSON（getty 污染？）: {line[:70]!r}")
                continue
            n += 1
            unit = r.get("obj_unit") or "?"
            # ★ 坐标系必须显眼 —— rev2 之后 face 是【车体系】，rev1 是相机系。
            #   同一个 [0.31,0.02,0.62] 在两种坐标下是两个完全不同的点。
            coord = r.get("coord") or "?"
            rev = r.get("rev")
            ctag = f"{coord}" + (f"/r{rev}" if rev is not None else "")
            objs = [fmt_obj(o, unit) for o in r.get("obj", [])]
            el = time.time() - t0
            print(f"{el:>7.1f}s {r.get('frame', 0):>6} {r.get('seq', 0):>6} "
                  f"{unit:>4} {ctag:>7} {r.get('n', len(objs)):>5}  "
                  + (" | ".join(objs) if objs else "-"))
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        if log:
            log.close()
        el = max(1e-6, time.time() - t0)
        print(f"\n共收到 {n} 条，{n/el:.1f} 条/秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
