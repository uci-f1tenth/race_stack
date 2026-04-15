#!/usr/bin/env python3
"""RL racing inference node.

Uses nav2_amcl for localization and extracts a centerline via
skeletonization of the occupancy grid — no dependency on the
Rust training environment.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
import torch
import torch.nn as nn
import typer
import yaml
from PIL import Image
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scipy.ndimage import convolve, label, uniform_filter1d
from scipy.spatial import KDTree
from sensor_msgs.msg import Imu, LaserScan
from skimage.morphology import remove_small_objects, skeletonize
from std_msgs.msg import Float32
from tf_transformations import euler_from_quaternion

# ── Constants ────────────────────────────────────────────────────────

STEER_MAX = 0.5236  # rad
STEER_VEL_MAX = 3.2  # rad s⁻¹
SPEED_MAX = 22.88  # m s⁻¹
N_LIDAR_RAW = 1080
N_LIDAR_OBS = 108
N_LOOK = 10
LIDAR_MAX = 10.0  # m
LIDAR_MIN = 0.06  # m
LOOP_HZ = 40.0
TOPIC_NS = "/autodrive/roboracer_1"

# ── Centerline extraction ────────────────────────────────────────────

_NBR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)


def _load_free_space(
    map_yaml: Path,
) -> tuple[np.ndarray, float, list[float], int]:
    """Parse a ROS map YAML and return (free mask, resolution, origin, height)."""
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    img_path = (map_yaml.parent / meta["image"]).resolve()
    res = float(meta["resolution"])
    origin = [float(v) for v in meta["origin"]]
    free_thresh = float(meta.get("free_thresh", 0.196))
    negate = bool(int(meta.get("negate", 0)))
    raw = np.array(Image.open(str(img_path)).convert("L"), dtype=np.float64)
    occ = raw / 255.0 if negate else (255.0 - raw) / 255.0
    return occ < free_thresh, res, origin, raw.shape[0]


def _prune_branches(skel: np.ndarray, max_iters: int = 500) -> np.ndarray:
    """Remove endpoint pixels iteratively; a closed loop has none."""
    skel = skel.copy()
    for _ in range(max_iters):
        nbrs = convolve(skel.astype(np.int32), _NBR_KERNEL, mode="constant")
        endpoints = skel & (nbrs == 1)
        if not endpoints.any():
            break
        skel[endpoints] = False
    return skel


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())[1:]
    return labeled == (np.argmax(sizes) + 1)


def _order_nn(pts: np.ndarray, start: int = 0) -> np.ndarray:
    """Nearest-neighbour walk starting from *start*."""
    n = len(pts)
    tree = KDTree(pts)
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    for _ in range(n - 1):
        _, idxs = tree.query(pts[order[-1]], k=min(64, n))
        for i in idxs:
            if not visited[i]:
                order.append(i)
                visited[i] = True
                break
    return pts[np.array(order)]


def _smooth_loop(pts: np.ndarray, win: int) -> np.ndarray:
    if len(pts) < 2 * win + 1:
        return pts
    pad = win
    sx = uniform_filter1d(np.pad(pts[:, 0], pad, mode="wrap"), 2 * pad + 1)[
        pad:-pad
    ]
    sy = uniform_filter1d(np.pad(pts[:, 1], pad, mode="wrap"), 2 * pad + 1)[
        pad:-pad
    ]
    return np.column_stack([sx, sy])


def load_centerline(map_yaml: Path, smooth_win: int = 5) -> np.ndarray:
    """Extract an ordered, smoothed closed centerline (N×2 world coords).

    Pipeline: threshold → skeletonize → largest component → prune branches
    → NN-order starting from the pixel nearest the map origin → pixel→world
    → ensure forward direction matches origin θ → circular smooth.
    """
    free, res, origin, img_h = _load_free_space(map_yaml)
    free = remove_small_objects(free, min_size=100)

    skel = skeletonize(free)
    skel = _keep_largest_component(skel)
    skel = _prune_branches(skel)

    ys, xs = np.where(skel)
    if len(ys) == 0:
        raise RuntimeError(f"Skeletonization of '{map_yaml}' yielded no pixels.")

    pixel_pts = np.column_stack([xs, ys]).astype(np.float64)

    # Start the walk from the skeleton pixel nearest the map origin
    # (mirrors the Rust extract_main_loop behaviour).
    origin_px = np.array([-origin[0] / res, img_h - 1.0 + origin[1] / res])
    start = int(np.argmin(np.sum((pixel_pts - origin_px) ** 2, axis=1)))
    ordered = _order_nn(pixel_pts, start=start)

    # Pixel → world (image y is flipped w.r.t. world y)
    wx = origin[0] + ordered[:, 0] * res
    wy = origin[1] + (img_h - 1 - ordered[:, 1]) * res
    centerline = np.column_stack([wx, wy])

    # Ensure the traversal direction agrees with the map origin yaw,
    # so that look-ahead indices point *forward* along the track.
    if len(centerline) >= 2:
        fwd = math.atan2(
            centerline[1, 1] - centerline[0, 1],
            centerline[1, 0] - centerline[0, 0],
        )
        err = (fwd - origin[2] + 3.0 * math.pi) % (2.0 * math.pi) - math.pi
        if abs(err) > math.pi / 2.0:
            centerline[1:] = centerline[1:][::-1]

    return _smooth_loop(centerline, win=smooth_win)


# ── Policy network ───────────────────────────────────────────────────


class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── ROS 2 node ───────────────────────────────────────────────────────


def _qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=depth,
    )


class RacingNode(Node):
    """Subscribes to lidar / IMU / steering feedback, publishes steering and
    throttle commands at ~40 Hz."""

    def __init__(
        self,
        map_yaml: Path,
        checkpoint: Path,
        throttle_scale: float,
    ) -> None:
        super().__init__("racing_agent")

        # ── Centerline ───────────────────────────────────────────────
        self.skel = load_centerline(map_yaml)
        self.n_wps = len(self.skel)
        self.skel_tree = KDTree(self.skel)

        seg_lens = np.linalg.norm(
            np.diff(self.skel, axis=0, append=self.skel[:1]), axis=1
        )
        self.look_step = max(1, round(1.0 / float(seg_lens.mean())))

        # ── Model ────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.obs_mean: np.ndarray = ckpt["obs_rms_mean"].numpy()
        self.obs_var: np.ndarray = ckpt["obs_rms_var"].numpy()

        obs_dim = self.obs_mean.shape[0]
        self.policy = Policy(obs_dim)
        self.policy.load_state_dict(ckpt["model"], strict=False)
        self.policy.to(device).eval()
        self.device = device

        # ── Pre-computed lidar subsampling indices ───────────────────
        self.lidar_idx = np.round(
            np.linspace(0, N_LIDAR_RAW - 1, N_LIDAR_OBS)
        ).astype(int)

        # ── Mutable state ────────────────────────────────────────────
        self.x = self.y = self.theta = 0.0
        self.prev_x = self.prev_y = 0.0
        self.speed = 0.0
        self.steer_fb = 0.0  # normalised [-1, 1] from hardware
        self.yaw_rate = 0.0
        self.lidar = np.full(N_LIDAR_RAW, LIDAR_MAX)
        self.pose_ok = False
        self.prev_t: float | None = None
        self.throttle_scale = throttle_scale

        # ── TF listener (AMCL: map → roboracer_1) ───────────────────
        self.tf_buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buf, self)

        # ── Pub / Sub ────────────────────────────────────────────────
        qos = _qos()
        self.create_subscription(
            LaserScan, f"{TOPIC_NS}/lidar", self._on_lidar, qos
        )
        self.create_subscription(
            Float32, f"{TOPIC_NS}/steering", self._on_steer, qos
        )
        self.create_subscription(Imu, f"{TOPIC_NS}/imu", self._on_imu, qos)

        self.pub_steer = self.create_publisher(
            Float32, f"{TOPIC_NS}/steering_command", qos
        )
        self.pub_throttle = self.create_publisher(
            Float32, f"{TOPIC_NS}/throttle_command", qos
        )

        self.create_timer(1.0 / LOOP_HZ, self._tick)
        self.get_logger().info(
            f"Ready  wps={self.n_wps}  look_step={self.look_step}  "
            f"throttle={self.throttle_scale:.2f}  obs_dim={obs_dim}"
        )

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_lidar(self, msg: LaserScan) -> None:
        self.lidar = np.asarray(msg.ranges, dtype=np.float64)

    def _on_steer(self, msg: Float32) -> None:
        self.steer_fb = float(msg.data)

    def _on_imu(self, msg: Imu) -> None:
        self.yaw_rate = msg.angular_velocity.z

    # ── Pose via AMCL TF tree ────────────────────────────────────────

    def _update_pose(self) -> bool:
        try:
            tf = self.tf_buf.lookup_transform(
                "map",
                "roboracer_1",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
            t, q = tf.transform.translation, tf.transform.rotation
            self.x, self.y = t.x, t.y
            _, _, self.theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.pose_ok = True
        except Exception:
            pass
        return self.pose_ok

    # ── Speed estimation (pose finite-difference + low-pass) ─────────

    def _estimate_speed(self, dt: float) -> None:
        if dt <= 0:
            return
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        raw = math.hypot(dx, dy) / dt
        # Sign: project displacement onto heading
        ch, sh = math.cos(self.theta), math.sin(self.theta)
        sign = 1.0 if (dx * ch + dy * sh) >= 0.0 else -1.0
        # First-order low-pass (τ ≈ 0.17 s)
        alpha = min(1.0, 6.0 * dt)
        self.speed += alpha * (sign * raw - self.speed)
        self.prev_x, self.prev_y = self.x, self.y

    # ── Observation vector ───────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        lidar = np.clip(self.lidar[self.lidar_idx], LIDAR_MIN, LIDAR_MAX)

        x, y, th = self.x, self.y, self.theta
        ch, sh = math.cos(th), math.sin(th)

        # Closest waypoint (O(log N) via KDTree instead of argmin)
        _, wi = self.skel_tree.query([x, y])
        cx, cy = self.skel[wi]
        nx, ny = self.skel[(wi + 1) % self.n_wps]
        cth = math.atan2(ny - cy, nx - cx)

        heading_err = (cth - th + math.pi) % (2.0 * math.pi) - math.pi
        lateral_err = -(x - cx) * math.sin(cth) + (y - cy) * math.cos(cth)

        # Look-ahead waypoints in body frame
        look = np.empty(N_LOOK * 2, dtype=np.float64)
        for k in range(N_LOOK):
            wp = (wi + (k + 1) * self.look_step) % self.n_wps
            dx, dy = self.skel[wp, 0] - x, self.skel[wp, 1] - y
            look[2 * k] = dx * ch + dy * sh
            look[2 * k + 1] = -dx * sh + dy * ch

        vel = float(np.clip(self.speed, -SPEED_MAX, SPEED_MAX))
        steer_rad = self.steer_fb * STEER_MAX

        obs = np.concatenate(
            [lidar, [vel, steer_rad, self.yaw_rate, heading_err, lateral_err], look]
        )
        return np.clip(
            (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8), -10.0, 10.0
        )

    # ── Control loop ─────────────────────────────────────────────────

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = (
            float(np.clip(now - self.prev_t, 1e-4, 0.1))
            if self.prev_t is not None
            else 1.0 / LOOP_HZ
        )
        self.prev_t = now

        if not self._update_pose():
            return
        self._estimate_speed(dt)

        # Forward pass
        obs = self._build_obs()
        with torch.no_grad():
            t_obs = torch.tensor(
                obs, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            action = self.policy(t_obs).cpu().numpy().ravel().clip(-1.0, 1.0)

        # Rate-limited steering integration (base = current physical steering)
        steer_cmd = float(
            np.clip(
                self.steer_fb * STEER_MAX + action[0] * STEER_VEL_MAX * dt,
                -STEER_MAX,
                STEER_MAX,
            )
        )

        self.pub_steer.publish(Float32(data=steer_cmd / STEER_MAX))
        self.pub_throttle.publish(Float32(data=float(action[1]) * self.throttle_scale))


# ── CLI ──────────────────────────────────────────────────────────────

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help="Launch the RL racing agent on ROS 2.",
)


@app.command()
def main(
    map_yaml: Path = typer.Argument(
        ..., exists=True, readable=True, help="ROS map .yaml file."
    ),
    checkpoint: Path = typer.Argument(
        ..., exists=True, readable=True, help="Agent .pt checkpoint."
    ),
    throttle: float = typer.Option(
        1.0, "-t", "--throttle", min=0.0, max=1.0, help="Throttle scale [0‑1]."
    ),
) -> None:
    """Drive the car using a trained PPO policy."""
    rclpy.init()
    node = RacingNode(map_yaml, checkpoint, throttle_scale=throttle)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    app()