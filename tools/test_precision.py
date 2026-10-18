#!/usr/bin/env python3
"""精度自测工具：连上正在跑的 ROS 流，采样 N 帧，打印一份精度报告。

为什么要有这个：
    "精确度多少" 这种问题，光看单帧数字看不出来 —— 必须**统计**。
    而统计要三种不同的测法，各有各的意义：

    【1】静止重复性  —— 相机和方块都不动，量的是**算法+传感器噪声**
                       不需要任何外部量具，把方块放那儿别动就行
    【2】绝对尺度    —— 拿一个已知长度当基准（你那个方块 25cm）
                       直接给出"准不准"，而不只是"稳不稳"
    【3】垂直性自校验 —— 方块的三个面必须两两 90°。
                       这是**不用任何量具**就能拿到的绝对角度基准！
                       实测 88.57° → 绝对角度误差的上界就是 1.43°

★★ 多物体时必须按【物体】分组，不能按"数组第几个"取：
   桥接是按分数排序发物体的，分数一抖顺序就换。
   如果按 boxes[0]/arrows[0] 取，每帧会取到不同的物体 ——
   看起来就像"面心狂抖 150mm"（其实是两个物体在画面里相距 150mm）。
   桥接现在把 marker.id 按物体分块（每块 16 个），
   所以这里用 id // 16 就能稳定认出"这是第几个物体"，
   再按"离上一帧最近"跟踪同一个物体。

用法（电脑上，先 source ROS，并让板子把数据发过来）:
    source /opt/ros/humble/setup.bash

    /usr/bin/python3 tools/test_precision.py --frames 150 --true-edge 0.25
    /usr/bin/python3 tools/test_precision.py --frames 150 --true-edge 0.25 \
        --json /tmp/precision_report.json

要点：
    · 采样期间**相机和方块都不要动**，否则量到的是"你手的抖动"
    · 方块要**三个面都看得见**，垂直性自校验才有数据
    · 方块离相机 **0.3~0.6m** 是甜点区
    · 画面里最好**只放一个方块**；有多个时本工具会跟踪离上一帧最近的那个
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from visualization_msgs.msg import MarkerArray
except ImportError:
    print("❌ 没找到 rclpy。先 source /opt/ros/humble/setup.bash")
    sys.exit(1)

ID_BLOCK = 16          # 和 ros_bridge/cloud_bridge.py 里的 ID_BLOCK 保持一致
TCP_PORT = 9100        # 和 ros_bridge/cloud_bridge.py 里的 port 保持一致


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class ObjObs(object):
    """一个物体在一帧里的观测量（三片薄板 + 方块 + 箭头）。"""

    def __init__(self):
        self.faces = []          # [(法向, 面心), ...]  每个拟合出来的面
        self.box = None          # (体心, 边长)
        self.arrow = None        # (主面心, 主面法向) —— 管线选中的那个面
        self.nearest = None      # ★ 离相机最近的物体点（和卷尺量的就是同一个东西）


class Frame(object):
    def __init__(self, msg):
        groups = {}
        for k in msg.markers:
            o = groups.setdefault(k.id // ID_BLOCK, ObjObs())
            if k.ns == "face" and k.type == 1:
                R = quat_to_R(k.pose.orientation.x, k.pose.orientation.y,
                              k.pose.orientation.z, k.pose.orientation.w)
                o.faces.append((R[:, 2].copy(),
                                np.array([k.pose.position.x, k.pose.position.y,
                                          k.pose.position.z])))
            elif k.ns == "box" and k.type == 1:
                o.box = (np.array([k.pose.position.x, k.pose.position.y,
                                   k.pose.position.z]), float(k.scale.x))
            elif k.ns == "nearest" and k.type == 2:
                o.nearest = float(np.linalg.norm(
                    [k.pose.position.x, k.pose.position.y, k.pose.position.z]))
            elif k.ns == "normal" and k.type == 0 and len(k.points) >= 2:
                p0 = np.array([k.points[0].x, k.points[0].y, k.points[0].z])
                p1 = np.array([k.points[1].x, k.points[1].y, k.points[1].z])
                d = p1 - p0
                n = np.linalg.norm(d)
                if n > 1e-9:
                    o.arrow = (p0, d / n)
        # 只保留"几何信息完整"的物体
        self.objs = [o for o in groups.values()
                     if o.box is not None and o.arrow is not None]


class Collector(Node):
    def __init__(self, want):
        super().__init__("precision_test")
        self.want = want
        self.frames = []          # 已选中的物体（每帧一个）
        self.n_msgs = 0
        self.n_multi = 0
        self._last = None         # 上一帧选中的体心，用来做跟踪
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE
        self.create_subscription(MarkerArray, "/detections", self.cb, qos)

    def cb(self, msg):
        self.n_msgs += 1
        fr = Frame(msg)
        if not fr.objs:
            return
        if len(fr.objs) > 1:
            self.n_multi += 1
        # ★ 跟踪：选离上一帧最近的那个物体，这样物体排序变了也不受影响
        if self._last is None:
            # 第一帧：取面数最多的（最可靠）
            pick = max(fr.objs, key=lambda o: len(o.faces))
        else:
            pick = min(fr.objs,
                       key=lambda o: np.linalg.norm(o.box[0] - self._last))
        if self._last is not None and \
                np.linalg.norm(pick.box[0] - self._last) > 0.30:
            return        # 跳得太远，多半是换物体了，丢掉这帧
        self._last = pick.box[0].copy()
        self.frames.append(pick)
        if len(self.frames) <= 3 or len(self.frames) % 20 == 0:
            print(f"  已采 {len(self.frames)}/{self.want} 帧"
                  f"（收到 {self.n_msgs} 条消息）", flush=True)


def report(frames, true_edge, verbose):
    fc = np.array([f.arrow[0] for f in frames])
    nv = np.array([f.arrow[1] for f in frames])
    cc = np.array([f.box[0] for f in frames])
    ed = np.array([f.box[1] for f in frames])
    ang = np.degrees(np.arccos(np.clip(np.abs(nv[:, 2]), 0, 1)))
    n = len(frames)
    out = {}

    line = "═" * 66
    print("\n" + line)
    print(f" 精度自测报告      有效 {n} 帧")
    print(line)

    # ---- 1. 静止重复性 ----
    print("\n【1】静止重复性  ← 相机和方块都没动，这里量的是算法+传感器噪声")
    print("       （标准差 = 稳不稳；准不准看第 2、3 项）")
    for name, arr in (("面心", fc), ("体心", cc)):
        sd = arr.std(axis=0) * 1000
        print(f"    {name}   X ±{sd[0]:5.1f}   Y ±{sd[1]:5.1f}   "
              f"Z ±{sd[2]:5.1f} mm      合成 ±{np.linalg.norm(sd):5.1f} mm")
        out[name + "_std_mm"] = [round(float(v), 2) for v in sd]
        out[name + "_std_norm_mm"] = round(float(np.linalg.norm(sd)), 2)
    sd_n = nv.std(axis=0)
    print(f"    法向   每轴 ±({sd_n[0]:.4f}, {sd_n[1]:.4f}, {sd_n[2]:.4f})"
          f"   → 角度 ±{ang.std():.2f}°")
    print(f"    角度   均值 {ang.mean():.2f}°   "
          f"范围 {ang.min():.1f}~{ang.max():.1f}°   "
          f"(极差 {ang.max()-ang.min():.2f}°)")
    print(f"    边长   {ed.mean()*100:.2f} ± {ed.std()*1000:.1f} mm")
    fm = fc.mean(axis=0)
    cm_ = cc.mean(axis=0)
    print(f"    面心均值 (X,Y,Z) = ({fm[0]:+.4f}, {fm[1]:+.4f}, {fm[2]:+.4f}) m"
          f"   |面心| = {np.linalg.norm(fm):.4f} m")
    print(f"    体心均值 (X,Y,Z) = ({cm_[0]:+.4f}, {cm_[1]:+.4f}, {cm_[2]:+.4f}) m"
          f"   |体心| = {np.linalg.norm(cm_):.4f} m")
    print(f"    ★ 两者差 {np.linalg.norm(cm_-fm)*100:.1f} cm —— 这就是"
          f"「量最近那个面」和「量面中心」的差别")
    nrs = [f.nearest for f in frames if f.nearest is not None]
    if nrs:
        nr = float(np.mean(nrs))
        print(f"    ★ 最近点均值 |最近点| = {nr:.4f} m   ← 卷尺量的就是这个，"
              f"和它比！")
        out["nearest_mean_m"] = round(nr, 5)
    print("    ★ 卷尺要量的话，量【面心】那个点；更稳的做法看下面【距离怎么测才准】")
    out["face_center_mean_m"] = [round(float(v), 5) for v in fm]
    out["face_center_mean_norm_m"] = round(float(np.linalg.norm(fm)), 5)
    out["cube_center_mean_m"] = [round(float(v), 5) for v in cm_]
    out["cube_center_mean_norm_m"] = round(float(np.linalg.norm(cm_)), 5)
    out["angle_std_deg"] = round(float(ang.std()), 3)
    out["angle_mean_deg"] = round(float(ang.mean()), 3)
    out["angle_range_deg"] = [round(float(ang.min()), 3),
                              round(float(ang.max()), 3)]
    out["edge_mean_m"] = round(float(ed.mean()), 4)
    out["edge_std_mm"] = round(float(ed.std() * 1000), 2)

    fc_j = np.linalg.norm(fc.std(axis=0)) * 1000
    verdict = ("✅ 极好" if fc_j < 2 else "✅ 可用" if fc_j < 10
               else "⚠️ 偏大 → 查：相机松动 / 自动曝光在动 / 画面里有多个物体")
    print(f"    ★ 面心合成抖动 {fc_j:.1f}mm → {verdict}")

    # ---- 1c. 抖动 vs 漂移 ----
    # ★★ 关键区分：
    #    抖动 = 随机的（多帧平均能压下去）
    #    漂移 = 单调的（平均压不下去，说明是【东西真的在动】）
    #   做法：把采样劈成前后两半，比均值。差得多 = 漂移为主。
    h2 = n // 2
    if h2 >= 5:
        print("\n【1c】抖动 vs 漂移  ← 区分「随机噪声」和「东西在动」")
        for name, arr in (("面心", fc), ("体心", cc)):
            d = (arr[h2:].mean(axis=0) - arr[:h2].mean(axis=0)) * 1000
            dn = float(np.linalg.norm(d))
            sd = float(np.linalg.norm(arr.std(axis=0)) * 1000)
            verdict = ("★ 漂移为主 → 是东西在动（相机松 / 方块滑 / 曝光漂）"
                       if dn > sd else "随机噪声为主 → 算法+传感器噪声")
            print(f"    {name}  前半段→后半段 均值移动 "
                  f"({d[0]:+5.1f},{d[1]:+5.1f},{d[2]:+5.1f}) mm  "
                  f"合成 {dn:5.1f}mm（抖动 {sd:5.1f}mm）")
            print(f"           {verdict}")
            out["drift_" + name + "_mm"] = [round(float(v), 2) for v in d]
        da = ang[h2:].mean() - ang[:h2].mean()
        print(f"    角度  前半段 {ang[:h2].mean():.2f}° → "
              f"后半段 {ang[h2:].mean():.2f}°   变化 {da:+.2f}°")
        out["drift_angle_deg"] = round(float(da), 3)
        print("    ★ 判读：漂移 >> 抖动 → 别调算法，去固定相机/方块，"
              "或关掉自动曝光\n")

    # ---- 1b. 三个面分别看 ----
    cnt = {}
    for f in frames:
        cnt[len(f.faces)] = cnt.get(len(f.faces), 0) + 1
    print("\n【1b】各面单独看  ← 相机其实看到了多个面，只是输出只报「主面」")
    print(f"      面数分布：" + "  ".join(f"{k}面 {v}帧"
                                       for k, v in sorted(cnt.items())))
    last = frames[-1]
    if last.faces:
        b = last.box[0]
        for i, (nvec, c) in enumerate(last.faces):
            a = math.degrees(math.acos(min(1.0, abs(float(nvec[2])))))
            d = float(np.linalg.norm(c - b))
            tag = ""
            if abs(float(np.dot(nvec, last.arrow[1]))) > 0.999:
                tag = "   ← 这就是输出的「主面」"
            print(f"    面{i}  面心({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})  "
                  f"法向({nvec[0]:+.2f},{nvec[1]:+.2f},{nvec[2]:+.2f})  "
                  f"夹角{a:5.1f}°  离体心 {d*100:.1f}cm{tag}")
        print(f"    体心({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f})")
        print("    ★ 三个面心都算出来了，但串口/输出只发【主面】那一个 ——"
              "设计上就是「一个物体→一个抓取目标」")
        if cnt.get(1, 0) + cnt.get(2, 0) > n * 0.3:
            print("    ⚠️ 面数不足的帧偏多 → 换个角度让方块露出三个面，"
                  "体心才准")

    # ---- 2. 绝对尺度 ----
    print("\n【2】绝对尺度  ← 拿你给的已知长度当基准")
    if true_edge:
        err = (ed.mean() - true_edge) * 1000
        pct = (ed.mean() / true_edge - 1) * 100
        print(f"    基准 {true_edge*100:.2f} cm   "
              f"量出 {ed.mean()*100:.2f} cm   "
              f"误差 {err:+.1f} mm ({pct:+.2f}%)")
        out["true_edge_m"] = true_edge
        out["edge_error_mm"] = round(float(err), 2)
        out["edge_error_pct"] = round(float(pct), 3)
        if abs(pct) < 1:
            print("    ✅ 尺度很准")
        elif abs(pct) < 3:
            print("    ✅ 可用。偏小多半是面边缘被截断（对面心无影响，"
                  "只让体心少推一点点）")
        else:
            print("    ⚠️ 偏大，检查：① 相机内参标定 ② 基准值对不对 "
                  "③ 方块是不是被截掉一块")
    else:
        print("    （加 --true-edge 0.25 才能出这一项）")

    # ---- 3. 垂直性自校验 ----
    print("\n【3】垂直性自校验  ← 方块的三个面必须两两 90°，不用任何量具")
    devs = []
    for f in frames:
        ns = [x[0] for x in f.faces]
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                a = math.degrees(math.acos(
                    min(1.0, abs(float(np.dot(ns[i], ns[j]))))))
                devs.append(a - 90.0)
    if devs:
        devs = np.array(devs)
        p95 = float(np.percentile(np.abs(devs), 95))
        print(f"    样本 {len(devs)} 对，与 90° 的偏差：")
        print(f"      中位数 {np.median(devs):+.3f}°   "
              f"标准差 {devs.std():.3f}°")
        print(f"      95% 分位 |{p95:.3f}°|   最坏一帧 |{np.abs(devs).max():.3f}°|")
        print("    ★ 看【95% 分位】，别只看最坏那一帧 —— 单帧离群不算精度")
        print("    ★ 这就是**绝对角度精度的一个上界**（不用量角器就能拿到）")
        print("      注意：里面还含了「方块本身不是完美正方体」的误差")
        out["perp_dev_mean_deg"] = round(float(devs.mean()), 4)
        out["perp_dev_std_deg"] = round(float(devs.std()), 4)
        out["perp_dev_max_deg"] = round(float(np.abs(devs).max()), 4)
        out["perp_dev_p95_deg"] = round(p95, 4)
        if p95 < 0.5:
            print("    ✅ 优于 0.5°（95% 分位）")
        elif p95 < 1.5:
            print("    ✅ 优于 1.5°（95% 分位）")
        else:
            print("    ⚠️ 偏大：可能是深度噪声大（离太远/太近）、"
                  "面拟合不好，或画面里有多个物体被混在一起")
    else:
        print("    样本不足：需要一帧里同时看到 ≥2 个面。"
              "换个角度让方块露出三个面再测。")

    # ---- 3b. 正交基恒等式（最严的一条）----
    #  三个面两两垂直 ⟺ 三个法向构成标准正交基 ⟺ cos²θ0+cos²θ1+cos²θ2 = 1
    #  这一条同时用到三个面，比"两两看"更敏感，而且**不需要任何量具**。
    ident = []
    for f in frames:
        ns = [x[0] for x in f.faces]
        if len(ns) == 3:
            ident.append(sum(float(n[2]) ** 2 for n in ns))
    if ident:
        ident = np.array(ident)
        err = np.abs(ident - 1.0).mean()
        print(f"\n【3b】正交基恒等式 cos²θ0+cos²θ1+cos²θ2 = 1   ← 最严的一条")
        print(f"    样本 {len(ident)} 帧   实测 {ident.mean():.4f} ± {ident.std():.4f}"
              f"   平均偏离 {err*100:.2f}%")
        out["orthobasis_mean"] = round(float(ident.mean()), 5)
        out["orthobasis_err_pct"] = round(float(err * 100), 3)
        if err < 0.01:
            print("    ✅ 三面构成标准正交基（误差 <1%）—— 几何可信")
        elif err < 0.03:
            print("    ✅ 基本正交（误差 1~3%）")
        else:
            print("    ⚠️ 偏离偏大，面拟合有问题（噪声大/面太小/多物体混了）")

    # ---- 4. 面数分布 ----
    print("\n【4】面数分布  ← 决定「体心」可不可信")
    out["n_faces_hist"] = {str(k): v for k, v in sorted(cnt.items())}
    for k in sorted(cnt):
        p = cnt[k] * 100.0 / n
        tag = ("体心 = 精确解 ✅" if k >= 3 else
               "体心 = 估的 ⚠️" if k == 2 else "体心 = 猜的 ❌")
        print(f"    {k} 个面  {cnt[k]:4d} 帧 ({p:5.1f}%)   {tag}")
    if cnt.get(3, 0) * 100.0 / n < 50:
        print("    ⚠️ 三面比例偏低 → 电控应以 face+normal 为主数据，"
              "体心只当参考（cube_ok 会是 false）")

    print("\n" + line)
    print(" 怎么读这份报告：")
    print("  · 【1】是「稳不稳」 —— 相机固定、光照稳定时应该很小")
    print("  · 【2】【3】是「准不准」 —— 这两项才是电控关心的")
    print("  · 想测**距离相关的精度**：把方块放到量好的 30/40/50/60cm 处，")
    print("    各跑一次，对比报告里的 |P| 和卷尺读数")
    print(line)
    return out


def tcp_clients_on(port):
    """数一下这个端口上有几条【已建立】的 TCP 连接 —— 即板子连上来了没。

    读 /proc/net/tcp，不用 ss：既不起子进程，也**绝不真的去 connect**。
    桥接的 listen backlog 只有 1（cloud_bridge.py: srv.listen(1)），
    随手探一下就可能把那个位置占掉，反而把桥接搞坏。
    """
    want = f"{port:04X}"
    n = 0
    try:
        with open("/proc/net/tcp") as f:
            next(f)                                   # 跳过表头
            for line in f:
                c = line.split()
                if len(c) < 4 or c[3] != "01":         # 01 = ESTABLISHED
                    continue
                if c[1].split(":")[1] == want or c[2].split(":")[1] == want:
                    n += 1
    except Exception:
        return None                       # 读不到就说"不知道"，别硬说成 0
    return n


def explain_silence(n_pub):
    """一条数据都没等到时，直接点出卡在哪一环。

    链路是：板子主程序 →(TCP 9100)→ 桥接 →(/detections)→ 本脚本。
    任何一环断了，现象都是"脚本一动不动"，但原因完全不同 ——
    所以这里逐环报一次，省得再去 ps / ss 翻。
    """
    if n_pub == 0:
        print("   ❌ /detections 没有发布者 —— 桥接没在跑。另开一个终端：")
        print("        source /opt/ros/humble/setup.bash")
        print("        /usr/bin/python3 ~/jetson-deploy/ros_bridge/"
              "cloud_bridge.py")
        return
    print(f"   ✓ 桥接在跑（/detections 有 {n_pub} 个发布者），"
          f"但一条数据都没发过")
    n_cli = tcp_clients_on(TCP_PORT)
    if n_cli is None:
        print("   ？ 读不到 /proc/net/tcp，判断不了板子连没连上")
    elif n_cli == 0:
        print(f"   ❌ TCP {TCP_PORT} 上没有任何已建立的连接 —— 板子没连上来")
        print("      板子主程序多半没在跑，去板子上起：")
        print("        ssh nano@nano.local "
              "'bash ~/work/azure_kinect/start_azure.sh'")
    else:
        print(f"   ✓ 板子已连上 TCP {TCP_PORT}（{n_cli} 条连接），"
              f"但桥接没攒够一帧完整数据")
        print("      多半是相机没出帧，或板子程序刚起还在加载 engine")
    print("   ★ 桥接是【收到完整一帧才发】的，所以这里的 0 条"
          "不代表桥接坏了")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150,
                    help="采多少帧（默认 150；4FPS 下约 40 秒）")
    ap.add_argument("--true-edge", type=float, default=None,
                    help="方块真实边长(m)，比如 0.25 —— 用来算绝对尺度误差")
    ap.add_argument("--timeout", type=float, default=240.0,
                    help="最多等多少秒（默认 240）")
    ap.add_argument("--tape", type=float, default=None,
                    help="这一轮用卷尺量出来的距离(m)，比如 0.30。"
                         "存进 json 后可以用 tools/distance_report.py 汇总成表")
    ap.add_argument("--json", default=None, help="把报告存成 json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    rclpy.init()
    node = Collector(a.frames)
    print(f"开始采样：目标 {a.frames} 帧（最多等 {a.timeout:.0f} 秒）")
    print("★ 采样期间请不要移动相机和方块")
    print("★ 画面里最好只放一个方块\n")
    t0 = time.time()
    t_hb = t0                 # 上次心跳的时刻
    hb_gap = 5.0              # 5 秒后先说一次，之后 20 秒一次（别刷屏）
    seen = (0, 0)             # 上次心跳时的（消息数, 可用帧数）
    diagnosed = False
    try:
        while len(node.frames) < a.frames and time.time() - t0 < a.timeout:
            rclpy.spin_once(node, timeout_sec=0.2)
            now = time.time()
            if now - t_hb < hb_gap:
                continue
            t_hb = now
            hb_gap = 20.0
            cur = (node.n_msgs, len(node.frames))
            # ★ 只有一个心跳周期里【什么都没动】才说话。
            #   正常在采的样子是：每 20 帧一行进度（在 cb 里打的），
            #   所以这里安静 = 有进展，不需要再插嘴。
            if cur == seen:
                el = now - t0
                if node.n_msgs == 0:
                    print(f"  已等 {el:.0f}s，/detections 一条消息都没收到",
                          flush=True)
                    if not diagnosed:
                        diagnosed = True
                        explain_silence(node.count_publishers("/detections"))
                else:
                    print(f"  已等 {el:.0f}s，收到 {node.n_msgs} 条消息，"
                          f"但可用帧一直是 {len(node.frames)}"
                          f" —— 物体缺 box/normal，被丢了", flush=True)
            seen = cur
    except KeyboardInterrupt:
        print("\n（手动中断）")
    frames, n_msgs, n_multi = node.frames, node.n_msgs, node.n_multi
    n_pub = node.count_publishers("/detections")   # 销毁前先读，销毁后就没了
    node.destroy_node()
    rclpy.shutdown()

    if len(frames) < 5:
        print(f"\n❌ 只采到 {len(frames)} 帧有物体的数据，测不了。")
        if n_msgs == 0:
            explain_silence(n_pub)
        else:
            print(f"   收到了 {n_msgs} 条 /detections，但几乎没一帧能用 ——"
                  f" 问题在物体本身，不在链路：")
            print("   ① 方块在画面里吗（要能拟合出面 + 法向）")
            print("   ② 方块离相机是不是太远/太近（0.3~0.6m 最好）")
            print("   ③ RViz 里 /detections 能看到框吗")
        return 1

    if n_multi > n_msgs * 0.3:
        print(f"\n⚠️ 注意：{n_multi}/{n_msgs} 帧里有【多个物体】。"
              f"本工具已按'离上一帧最近'跟踪其中一个，")
        print("   但最好把其它东西拿走，只留一个方块再测，结果才干净。")

    rep = report(frames, a.true_edge, a.verbose)
    rep["n_frames"] = len(frames)
    rep["n_msgs"] = n_msgs
    rep["n_multi_object_frames"] = n_multi
    rep["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if a.tape:
        rep["tape_m"] = a.tape
        d = rep["face_center_mean_norm_m"]
        err = d - a.tape
        print(f"\n【距离对比】卷尺 {a.tape:.4f} m   管线 |P| {d:.4f} m   "
              f"差 {err*1000:+.1f} mm ({err/a.tape*100:+.2f}%)")
        print("   注意：这个差里含一个【固定偏移】—— 深度光心并不在相机外壳前沿，")
        print("   所以看增量更准：把方块移动量好的 10cm，看 |P| 变不变 10cm。")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 报告已存到 {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
