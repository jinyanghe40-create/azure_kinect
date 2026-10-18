#!/usr/bin/env python3
"""
============================================================
 电脑侧：ROS 2 桥接节点
============================================================
 接收板子通过 TCP 发来的点云 + 检测结果，发布成 ROS 2 话题：

    订阅方（RViz）                话题
    ─────────────────────────────────────────────
    点云        /camera/points      sensor_msgs/PointCloud2
    检测框+标注 /detections         visualization_msgs/MarkerArray
                                        ← 立方体框 + 类别标签 + 法向量箭头

 跑法（电脑上，必须先 source ROS）:
     source /opt/ros/humble/setup.bash
     /usr/bin/python3 ~/jetson-deploy/ros_bridge/cloud_bridge.py
 再开 RViz:
     rviz2 -d ~/jetson-deploy/ros_bridge/cloud.rviz

 ★ 为什么不让板子跑 ROS：
    板子是 Ubuntu 18.04，装不了 ROS 2 Humble（要 22.04）；
    而且 Jetson Nano 跑 RViz 会卡死。
    所以板子只发数据（common/ros_sender.py），电脑负责显示。
============================================================
"""
import json
import socket
import struct
import sys
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

MAGIC = b"K4PC"
# 每个类一个颜色（RGB 0~1），顺序要和训练时一致
CLS_COLOR = {
    "tools":      (0.95, 0.35, 0.15),   # 橙红
    "medicine":   (0.95, 0.85, 0.20),   # 黄
    "instrument": (0.20, 0.65, 0.95),   # 蓝
    "food":       (0.30, 0.85, 0.35),   # 绿
}
DEFAULT_COLOR = (0.8, 0.8, 0.8)


def read_exact(sock, n):
    """TCP 是流，必须循环读满 n 字节"""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class CloudBridge(Node):
    def __init__(self, port=9100):
        super().__init__("realsense_cloud_bridge")
        self.port = port

        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE

        self.pub_cloud = self.create_publisher(PointCloud2, "/camera/points", qos)
        self.pub_mark = self.create_publisher(MarkerArray, "/detections", qos)

        self.frames = 0
        self.points_total = 0
        self.t_last = time.time()
        self.hz = 0.0

        self._stop = threading.Event()
        self._th = threading.Thread(target=self._serve, daemon=True)
        self._th.start()
        self.get_logger().info(f"TCP 监听 0.0.0.0:{port}，等板子连进来…")

    # ---------------- TCP 服务 ----------------
    def _serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("0.0.0.0", self.port))
        except OSError as e:
            self.get_logger().error(
                f"端口 {self.port} 被占用（{e}）。先跑: pkill -f cloud_bridge")
            raise
        srv.listen(1)
        srv.settimeout(1.0)
        while not self._stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.get_logger().info(f"✅ 板子已连接: {addr[0]}:{addr[1]}")
            conn.settimeout(5.0)
            try:
                self._loop(conn)
            except Exception as e:
                self.get_logger().warn(f"连接断开: {type(e).__name__} {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                self.get_logger().info("板子断开，继续等待…")
        srv.close()

    def _loop(self, conn):
        while not self._stop.is_set():
            magic = read_exact(conn, 4)
            if magic is None:
                return
            if magic != MAGIC:
                self.get_logger().error(f"协议错乱（收到 {magic!r}），断开")
                return
            hlen = struct.unpack("<I", read_exact(conn, 4))[0]
            if hlen <= 0 or hlen > 4 << 20:
                self.get_logger().error(f"头长度异常 {hlen}，断开")
                return
            head = json.loads(read_exact(conn, hlen).decode("utf-8"))
            n = int(head["n"])
            raw = read_exact(conn, n * 12)
            if raw is None:
                return
            pts = np.frombuffer(raw, dtype="<f4").reshape(-1, 3)

            self.publish(pts, head, conn.getpeername()[0])

    # ---------------- 发话题 ----------------
    def publish(self, pts, head, peer):
        stamp = self.get_clock().now().to_msg()
        frame = head.get("frame_id", "camera_depth_optical_frame")

        # ---- ① 点云 ----
        msg = PointCloud2()
        msg.header = Header(stamp=stamp, frame_id=frame)
        msg.height = 1
        msg.width = len(pts)
        msg.is_bigendian = False
        msg.is_dense = True
        msg.point_step = 12
        msg.row_step = 12 * len(pts)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = pts.astype("<f4").tobytes()
        self.pub_cloud.publish(msg)

        # ---- ② 检测框 / 标签 / 法向量箭头 ----
        arr = MarkerArray()
        mid = 0
        for o in head.get("objects", []):
            xyz = o.get("xyz")
            if not xyz:
                continue
            col = CLS_COLOR.get(o.get("cls"), DEFAULT_COLOR)
            cls_txt = (o.get("cn") or "") + (o.get("cls") or "?")
            ang = o.get("angle")
            npts = o.get("n_points")
            rms = o.get("rms_mm")

            # 立方体线框（size 由板子告诉我们，默认 0.25m）
            m = Marker()
            m.header = Header(stamp=stamp, frame_id=frame)
            m.ns, m.id = "box", mid
            mid += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position = Point(x=float(xyz[0]), y=float(xyz[1]),
                                    z=float(xyz[2]))
            m.pose.orientation.w = 1.0
            e = float(o.get("edge", 0.25))
            m.scale.x = m.scale.y = m.scale.z = e
            m.color.r, m.color.g, m.color.b = col
            m.color.a = 0.35
            arr.markers.append(m)

            # 类别 + 夹角的文字标签
            t = Marker()
            t.header = Header(stamp=stamp, frame_id=frame)
            t.ns, t.id = "label", mid
            mid += 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position = Point(x=float(xyz[0]),
                                    y=float(xyz[1]),
                                    z=float(xyz[2]) + e * 0.7)
            t.pose.orientation.w = 1.0
            t.scale.z = 0.03
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 1.0
            txt = f"{cls_txt} {o.get('score', 0):.2f}"
            if ang is not None:
                txt += f"\n{ang:.1f}deg"
            if npts:
                txt += f"\n{npts}pts"
            if rms is not None:
                txt += f" rms{rms:.1f}mm"
            t.text = txt
            arr.markers.append(t)

            # 法向量箭头 —— 这个最能"看见夹角"
            nvec = o.get("normal")
            if nvec:
                a = Marker()
                a.header = Header(stamp=stamp, frame_id=frame)
                a.ns, a.id = "normal", mid
                mid += 1
                a.type = Marker.ARROW
                a.action = Marker.ADD
                L = e * 0.9
                a.points = [
                    Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                    Point(x=float(xyz[0]) + float(nvec[0]) * L,
                          y=float(xyz[1]) + float(nvec[1]) * L,
                          z=float(xyz[2]) + float(nvec[2]) * L),
                ]
                a.scale.x = 0.01          # 箭杆粗细
                a.scale.y = 0.02          # 箭头粗细
                a.scale.z = 0.03          # 箭头长度
                a.color.r, a.color.g, a.color.b = 1.0, 0.2, 0.2
                a.color.a = 1.0
                arr.markers.append(a)

        self.pub_mark.publish(arr)

        # ---- ③ 统计 ----
        self.frames += 1
        self.points_total += len(pts)
        now = time.time()
        if now - self.t_last >= 1.0:
            self.hz = self.frames / (now - self.t_last)
            self.get_logger().info(
                f"[{peer}] {len(pts)} 点  物体 {len(head.get('objects', []))}  "
                f"{self.hz:.1f} Hz  累计 {self.frames} 帧")
            self.frames = 0
            self.t_last = now

    def shutdown(self):
        self._stop.set()


def main():
    rclpy.init()
    node = CloudBridge(port=9100)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        # Ctrl+C 时 ROS 2 会抛 ExternalShutdownException，属正常
        if type(e).__name__ != "ExternalShutdownException":
            raise
    finally:
        # ⚠️ Ctrl+C / 被 kill 时 rclpy 自己可能已经关过 context 了，
        #    这里再关会报 "rcl_shutdown already called"，所以每步都护住。
        try:
            node.shutdown()
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
