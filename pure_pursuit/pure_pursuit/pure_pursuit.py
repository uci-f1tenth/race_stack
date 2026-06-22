#!/usr/bin/env python3
"""Pure Pursuit node for the base f1tenth_stack.

Same /drive convention as the disparity_extender node: steering normalized by
the physical lock and sign-flipped to match the servo. Pose comes from an
Odometry topic; the path is the map centerline (skeletonized, BFS-traced,
savgol-smoothed).
"""

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
curv_gain: float = 0.5
min_speed_factor: float = 0.3

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
        # Wrap-aware central differences (closed loop -> no seam spike).
        dx = (np.roll(self.wx, -1) - np.roll(self.wx, 1)) * 0.5
        dy = (np.roll(self.wy, -1) - np.roll(self.wy, 1)) * 0.5
        d2x = np.roll(self.wx, -1) - 2.0 * self.wx + np.roll(self.wx, 1)
        d2y = np.roll(self.wy, -1) - 2.0 * self.wy + np.roll(self.wy, 1)
        self.wcurv = np.abs(dx * d2y - dy * d2x) / np.maximum(
            (dx * dx + dy * dy) ** 1.5, 1e-6
        )
        self.last_idx = 0
        self.first_fix = True
        self.lookahead_sq = lookahead * lookahead
        self.search_window = max(50, self.n_wp // 8)

        self.pose = None
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

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
        if self.pose is None:
            return
        x, y, theta = self.pose
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        n = self.n_wp
        if self.first_fix:
            # One-shot global nearest so a far-from-origin start can't fool
            # the windowed search.
            d2_all = (self.wx - x) ** 2 + (self.wy - y) ** 2
            nearest = int(np.argmin(d2_all))
            self.first_fix = False
        else:
            w = self.search_window
            idxs = (self.last_idx + np.arange(-w, w + 1)) % n
            dxs = self.wx[idxs] - x
            dys = self.wy[idxs] - y
            d2 = dxs * dxs + dys * dys
            nearest = int(idxs[int(np.argmin(d2))])
        self.last_idx = nearest

        ahead_idxs = (nearest + np.arange(n)) % n
        ddx = self.wx[ahead_idxs] - x
        ddy = self.wy[ahead_idxs] - y
        ahead = (cos_t * ddx + sin_t * ddy) > 0.0
        d2 = ddx * ddx + ddy * ddy
        past = ahead & (d2 >= self.lookahead_sq)
        if past.any():
            k = int(ahead_idxs[int(np.argmax(past))])
        elif ahead.any():
            k = int(ahead_idxs[int(np.argmax(np.where(ahead, d2, -1.0)))])
        else:
            k = nearest
        gx, gy = self.wx[k], self.wy[k]

        dx, dy = gx - x, gy - y
        local_y = -sin_t * dx + cos_t * dy
        L2 = dx * dx + dy * dy
        # Standard pure-pursuit law: stable for any goal distance.
        delta = np.arctan2(2.0 * wheelbase * local_y, L2) if L2 > 1e-6 else 0.0
        steer = steer_sign * float(
            np.clip(delta / max_steer, -steer_clamp, steer_clamp)
        )
        curv = self.wcurv[nearest]
        speed_factor = 1.0 / (1.0 + curv * curv_gain)
        current_speed = speed * max(speed_factor, min_speed_factor)
        self.publish_drive(steer, current_speed)


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
