#!/usr/bin/env python3
"""Pure Pursuit node for the base f1tenth_stack.

Same safety scheme and /drive convention as the disparity_extender node:
UDP deadman + watchdog, steering normalized by the physical lock and
sign-flipped to match the servo. Pose comes from an Odometry topic; the
path is the map centerline (skeletonized, BFS-traced, savgol-smoothed).
"""

import socket
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from cv2 import IMREAD_GRAYSCALE, imread
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from yaml import safe_load

map_path: str = "maps/my_map.yaml"
odom_topic: str = "/pf/pose/odom"
lookahead: float = 1.5
wheelbase: float = 0.3302
max_steer: float = 0.4189
steer_sign: float = -1.0
steer_clamp: float = 0.95
speed: float = 3.0
deadman_timeout: float = 0.3
deadman_port: int = 5005

OCC_THRESH: int = 250
SMOOTH_WINDOW: int = 21
ADJ = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def quat_to_yaw(x, y, z, w):
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _neighbors(skel, r, c, h, w):
    return [
        (r + dr, c + dc)
        for dr, dc in ADJ
        if 0 <= r + dr < h and 0 <= c + dc < w and skel[r + dr, c + dc]
    ]


def compute_centerline(yaml_path):
    """Skeletonize the map, trace the loop from a seed via BFS, smooth it."""
    p = Path(yaml_path)
    meta = safe_load(p.read_text())
    raw = imread(str(p.parent / meta["image"]), IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(p.parent / meta["image"])
    res = float(meta["resolution"])
    ox, oy, _ = meta["origin"]
    h, w = raw.shape

    skel = skeletonize(raw >= OCC_THRESH)
    pts = np.argwhere(skel)
    origin_px = np.array([h - 1 + oy / res, -ox / res])
    start = tuple(int(v) for v in pts[np.argmin(((pts - origin_px) ** 2).sum(1))])

    nbrs = _neighbors(skel, *start, h, w)
    if len(nbrs) < 2:
        raise RuntimeError(f"Skeleton seed {start} has {len(nbrs)} neighbours")
    src, target = nbrs[0], nbrs[1]

    parent = {src: src}
    q = deque([src])
    found = False
    while q and not found:
        r, c = q.popleft()
        for nb in _neighbors(skel, r, c, h, w):
            if nb in parent or nb == start:
                continue
            parent[nb] = (r, c)
            if nb == target:
                found = True
                break
            q.append(nb)
    if not found:
        raise RuntimeError("BFS could not close the loop; skeleton is broken")

    path = [start]
    n = target
    while n != src:
        path.append(n)
        n = parent[n]
    path.append(src)
    path.reverse()

    rc = np.array(path)
    world = np.column_stack([ox + rc[:, 1] * res, oy + (h - 1 - rc[:, 0]) * res])
    return savgol_filter(world, SMOOTH_WINDOW, 3, axis=0, mode="wrap")


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit")
        self.waypoints = compute_centerline(map_path)
        self.get_logger().info(f"Centerline: {len(self.waypoints)} points")

        self.wx = np.ascontiguousarray(self.waypoints[:, 0])
        self.wy = np.ascontiguousarray(self.waypoints[:, 1])
        self.n_wp = len(self.waypoints)
        self.last_idx = 0
        self.lookahead_sq = lookahead * lookahead
        self.search_window = max(50, self.n_wp // 8)

        self.pose = None
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", deadman_port))
        self.sock.setblocking(False)
        self.last_deadman = 0.0
        self.create_timer(0.02, self.poll_deadman)
        self.create_timer(0.05, self.watchdog)

    def destroy_node(self):
        try:
            self.sock.close()
        except OSError:
            pass
        super().destroy_node()

    def poll_deadman(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(64)
            except BlockingIOError:
                break
            try:
                self.sock.sendto(b"ok", addr)
            except OSError:
                pass
            if data == b"1":
                self.last_deadman = self.get_clock().now().nanoseconds * 1e-9

    def is_armed(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self.last_deadman) < deadman_timeout

    def watchdog(self):
        if not self.is_armed():
            self.publish_drive(0.0, 0.0)

    def publish_drive(self, steering, speed):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.drive.steering_angle = float(steering)
        m.drive.speed = float(speed)
        self.drive_pub.publish(m)

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.pose = (p.x, p.y, quat_to_yaw(o.x, o.y, o.z, o.w))
        self.control_step()

    def control_step(self):
        if self.pose is None or not self.is_armed():
            return
        x, y, theta = self.pose

        n = self.n_wp
        w = self.search_window
        idxs = (self.last_idx + np.arange(-w, w + 1)) % n
        dxs = self.wx[idxs] - x
        dys = self.wy[idxs] - y
        d2 = dxs * dxs + dys * dys
        nearest = int(idxs[int(np.argmin(d2))])
        self.last_idx = nearest

        gx, gy = self.wx[nearest], self.wy[nearest]
        best_d2 = -1.0
        found = False
        for j in range(1, n):
            k = (nearest + j) % n
            cx, cy = self.wx[k], self.wy[k]
            ddx, ddy = cx - x, cy - y
            cd2 = ddx * ddx + ddy * ddy
            if cd2 >= self.lookahead_sq:
                gx, gy = cx, cy
                found = True
                break
            if cd2 > best_d2:
                best_d2 = cd2
                gx, gy = cx, cy

        dx, dy = gx - x, gy - y
        local_y = -np.sin(theta) * dx + np.cos(theta) * dy
        L2 = dx * dx + dy * dy
        delta = np.arctan(2.0 * local_y * wheelbase / L2) if L2 > 1e-9 else 0.0
        steer = steer_sign * float(
            np.clip(delta / max_steer, -steer_clamp, steer_clamp)
        )
        self.publish_drive(steer, speed)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
