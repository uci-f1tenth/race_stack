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
odom_topic: str = "/pf/pose/odom"  # particle_filter; "/ego_racecar/odom" in sim
lookahead: float = 1.5  # m
wheelbase: float = 0.3302  # m
max_steer: float = 0.4189  # rad, physical lock (sim STEER_FACTOR = 1/this)
steer_sign: float = -1.0  # flip to match the car's servo polarity
steer_clamp: float = 0.95  # normalized steering limit
speed: float = 3.0  # m/s
deadman_timeout: float = 0.3  # s
deadman_port: int = 5005

OCC_THRESH: int = 250  # grayscale >= this is free space
SMOOTH_WINDOW: int = 21  # savgol window (odd)
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

    # BFS the loop the long way round (start node is removed from the graph).
    parent = {src: src}
    q = deque([src])
    while q:
        r, c = q.popleft()
        for n in _neighbors(skel, r, c, h, w):
            if n in parent or n == start:
                continue
            parent[n] = (r, c)
            if n == target:
                q.clear()
                break
            q.append(n)

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

    def poll_deadman(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(64)
            except BlockingIOError:
                break
            try:
                self.sock.sendto(b"ok", addr)  # echo so GUI knows we're alive
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
        pos = np.array([x, y])

        # Nearest waypoint, then first one >= lookahead ahead (wrap-around).
        nearest = int(np.argmin(np.linalg.norm(self.waypoints - pos, axis=1)))
        n = len(self.waypoints)
        gx, gy = self.waypoints[nearest]
        for j in range(1, n):
            gx, gy = self.waypoints[(nearest + j) % n]
            if np.hypot(gx - x, gy - y) >= lookahead:
                break

        # Pure-pursuit curvature -> steering, in the disparity_extender
        # convention: normalize by the lock, clamp, then sign-flip.
        dx, dy = gx - x, gy - y
        local_y = -np.sin(theta) * dx + np.cos(theta) * dy
        L2 = dx * dx + dy * dy
        delta = np.arctan(2.0 * local_y * wheelbase / L2) if L2 > 1e-12 else 0.0
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
