#!/usr/bin/env python3
"""RL racing inference node for the RoboRacer Sim Racing League.

Localization is handled by a lightweight particle filter over the
occupancy grid — only permitted input topics are used at runtime.
No dependency on /tf, /ips, /odom, or any other restricted stream.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
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
from scipy.ndimage import convolve, distance_transform_edt, label, uniform_filter1d
from scipy.spatial import KDTree
from sensor_msgs.msg import Imu, JointState, LaserScan
from skimage.morphology import remove_small_objects, skeletonize
from std_msgs.msg import Float32

# ── Vehicle constants (from technical guide) ─────────────────────────

WHEELBASE = 0.3240  # m
WHEEL_RADIUS = 0.0590  # m
TRACK_WIDTH = 0.2360  # m
CAR_LENGTH = 0.5000  # m
CAR_WIDTH = 0.2700  # m

STEER_MAX = 0.5236  # rad  (±30 deg)
STEER_VEL_MAX = 3.2  # rad/s
SPEED_MAX = 22.88  # m/s

# Encoder: 16 PPR × 120 gear ratio = 1920 ticks/rev
ENCODER_PPR = 16
ENCODER_GEAR_RATIO = 120
TICKS_PER_REV = ENCODER_PPR * ENCODER_GEAR_RATIO
WHEEL_CIRC = 2.0 * math.pi * WHEEL_RADIUS

# ── LIDAR constants (from technical guide) ───────────────────────────

N_LIDAR_RAW = 1080  # 270° / 0.25°
N_LIDAR_OBS = 108
LIDAR_FOV = math.radians(270.0)
LIDAR_MAX = 10.0  # m
LIDAR_MIN = 0.06  # m
LIDAR_SCAN_RATE = 40  # Hz

# LIDAR frame: 0.2733 m ahead of rear axle along x
LIDAR_X_OFFSET = 0.2733
LIDAR_Y_OFFSET = 0.0

# ── RL observation constants ─────────────────────────────────────────

N_LOOK = 10
LOOP_HZ = 40.0
TOPIC_NS = "/autodrive/roboracer_1"

# ── Particle filter tuning ───────────────────────────────────────────

N_PARTICLES = 500
N_BEAMS_PF = 36  # subsampled beams for likelihood computation
SIGMA_HIT = 0.15  # m – sensor model std dev
ALPHA_SLOW = 0.001
ALPHA_FAST = 0.1
RESAMPLE_THRESHOLD = 0.5  # fraction of N_eff / N

# Motion noise (proportional)
ALPHA_V = 0.10  # linear velocity noise fraction
ALPHA_W = 0.15  # angular velocity noise fraction
ALPHA_VW = 0.02  # cross-coupling v → ω
ALPHA_WV = 0.02  # cross-coupling ω → v

# ── Skeletonization helpers ──────────────────────────────────────────

_NBR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)


def _load_map(
    map_yaml: Path,
) -> tuple[np.ndarray, np.ndarray, float, list[float], int]:
    """Return (free_mask, gray_image, resolution, origin, height)."""
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    img_path = (map_yaml.parent / meta["image"]).resolve()
    res = float(meta["resolution"])
    origin = [float(v) for v in meta["origin"]]
    free_thresh = float(meta.get("free_thresh", 0.196))
    negate = bool(int(meta.get("negate", 0)))
    raw = np.array(Image.open(str(img_path)).convert("L"), dtype=np.float64)
    occ = raw / 255.0 if negate else (255.0 - raw) / 255.0
    return occ < free_thresh, raw, res, origin, raw.shape[0]


def _prune_branches(skel: np.ndarray, max_iters: int = 500) -> np.ndarray:
    skel = skel.copy()
    for _ in range(max_iters):
        nbrs = convolve(skel.astype(np.int32), _NBR_KERNEL, mode="constant")
        endpoints = skel & (nbrs == 1)
        if not endpoints.any():
            break
        skel[endpoints] = False
    return skel


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    labeled, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())[1:]
    return labeled == (np.argmax(sizes) + 1)


def _order_nn(pts: np.ndarray, start: int = 0) -> np.ndarray:
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
    sx = uniform_filter1d(np.pad(pts[:, 0], win, mode="wrap"), 2 * win + 1)[win:-win]
    sy = uniform_filter1d(np.pad(pts[:, 1], win, mode="wrap"), 2 * win + 1)[win:-win]
    return np.column_stack([sx, sy])


def _extract_centerline(
    free: np.ndarray, res: float, origin: list[float], img_h: int, smooth_win: int = 5
) -> np.ndarray:
    free_clean = remove_small_objects(free, min_size=100)
    skel = skeletonize(free_clean)
    skel = _keep_largest(skel)
    skel = _prune_branches(skel)
    ys, xs = np.where(skel)
    if len(ys) == 0:
        raise RuntimeError("Skeletonization produced no centerline pixels.")
    pixel_pts = np.column_stack([xs, ys]).astype(np.float64)
    origin_px = np.array([-origin[0] / res, img_h - 1.0 + origin[1] / res])
    start = int(np.argmin(np.sum((pixel_pts - origin_px) ** 2, axis=1)))
    ordered = _order_nn(pixel_pts, start=start)
    wx = origin[0] + ordered[:, 0] * res
    wy = origin[1] + (img_h - 1 - ordered[:, 1]) * res
    cl = np.column_stack([wx, wy])
    if len(cl) >= 2:
        fwd = math.atan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0])
        err = (fwd - origin[2] + 3.0 * math.pi) % (2.0 * math.pi) - math.pi
        if abs(err) > math.pi / 2.0:
            cl[1:] = cl[1:][::-1]
    return _smooth_loop(cl, win=smooth_win)


# ── Particle filter (likelihood-field model) ─────────────────────────


class ParticleFilter:
    """Monte-Carlo localisation using the occupancy grid EDT."""

    def __init__(
        self,
        free_mask: np.ndarray,
        resolution: float,
        origin: list[float],
        img_h: int,
        n_particles: int = N_PARTICLES,
    ) -> None:
        self.res = resolution
        self.inv_res = 1.0 / resolution
        self.ox, self.oy = origin[0], origin[1]
        self.img_h = img_h
        self.map_h, self.map_w = free_mask.shape
        self.n = n_particles

        # EDT: distance (m) from each cell to nearest obstacle
        obstacle_mask = ~free_mask
        self.edt = distance_transform_edt(~obstacle_mask) * resolution

        # Pre-compute beam angles for the PF subset
        full_angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, N_LIDAR_RAW)
        pf_idx = np.round(np.linspace(0, N_LIDAR_RAW - 1, N_BEAMS_PF)).astype(int)
        self.pf_beam_angles = full_angles[pf_idx]
        self.pf_beam_idx = pf_idx  # indices into the raw scan

        # Initialise particles uniformly on free space
        free_ys, free_xs = np.where(free_mask)
        choice = np.random.choice(len(free_ys), size=n_particles, replace=True)
        self.particles = np.empty((n_particles, 3))  # x, y, θ
        self.particles[:, 0] = origin[0] + free_xs[choice] * resolution
        self.particles[:, 1] = origin[1] + (img_h - 1 - free_ys[choice]) * resolution
        self.particles[:, 2] = np.random.uniform(-math.pi, math.pi, n_particles)
        self.weights = np.full(n_particles, 1.0 / n_particles)

        self.w_slow = 0.0
        self.w_fast = 0.0

    # ── Coordinate helpers ───────────────────────────────────────────

    def _world_to_px(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        px = ((x - self.ox) * self.inv_res).astype(np.int32)
        py = (self.img_h - 1 - ((y - self.oy) * self.inv_res)).astype(np.int32)
        return px, py

    def _edt_lookup(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Look up EDT values; out-of-bounds → 0 (treat as collision)."""
        valid = (px >= 0) & (px < self.map_w) & (py >= 0) & (py < self.map_h)
        dist = np.zeros(px.shape, dtype=np.float64)
        dist[valid] = self.edt[py[valid], px[valid]]
        return dist

    # ── Motion model (velocity model with noise) ─────────────────────

    def predict(self, v: float, omega: float, dt: float) -> None:
        """Propagate particles using a noisy velocity motion model."""
        # Add proportional noise
        v_mag = abs(v) + 1e-6
        w_mag = abs(omega) + 1e-6
        sv = ALPHA_V * v_mag + ALPHA_WV * w_mag
        sw = ALPHA_W * w_mag + ALPHA_VW * v_mag

        vn = v + np.random.normal(0, sv, self.n)
        wn = omega + np.random.normal(0, sw, self.n)
        gn = np.random.normal(0, 0.01, self.n)  # heading drift

        th = self.particles[:, 2]

        # Handle near-zero angular velocity
        mask = np.abs(wn) > 1e-6
        dx = np.zeros(self.n)
        dy = np.zeros(self.n)
        dth = wn * dt + gn * dt

        # Arc motion
        r = np.zeros(self.n)
        r[mask] = vn[mask] / wn[mask]
        new_th = th + dth
        dx[mask] = r[mask] * (np.sin(new_th[mask]) - np.sin(th[mask]))
        dy[mask] = r[mask] * (-np.cos(new_th[mask]) + np.cos(th[mask]))

        # Straight-line motion
        straight = ~mask
        dx[straight] = vn[straight] * np.cos(th[straight]) * dt
        dy[straight] = vn[straight] * np.sin(th[straight]) * dt

        self.particles[:, 0] += dx
        self.particles[:, 1] += dy
        self.particles[:, 2] = (new_th + math.pi) % (2 * math.pi) - math.pi

    # ── Sensor model (likelihood field) ──────────────────────────────

    def update(self, scan: np.ndarray) -> None:
        """Weight particles by lidar likelihood-field model."""
        ranges = scan[self.pf_beam_idx]
        valid = (ranges > LIDAR_MIN) & (ranges < LIDAR_MAX - 0.01)
        if valid.sum() < 3:
            return

        angles = self.pf_beam_angles[valid]
        r = ranges[valid]

        # For each particle, compute beam endpoints in world frame
        th = self.particles[:, 2]  # (N,)
        # Global angle of each beam for each particle: (N, B)
        global_angles = th[:, None] + angles[None, :]

        # Lidar origin in world frame for each particle
        lx = (
            self.particles[:, 0]
            + LIDAR_X_OFFSET * np.cos(th)
            - LIDAR_Y_OFFSET * np.sin(th)
        )
        ly = (
            self.particles[:, 1]
            + LIDAR_X_OFFSET * np.sin(th)
            + LIDAR_Y_OFFSET * np.cos(th)
        )

        # Beam endpoints: (N, B)
        ex = lx[:, None] + r[None, :] * np.cos(global_angles)
        ey = ly[:, None] + r[None, :] * np.sin(global_angles)

        # Look up EDT at each endpoint
        px, py = self._world_to_px(ex.ravel(), ey.ravel())
        d = self._edt_lookup(px, py).reshape(self.n, -1)

        # Log-likelihood per particle (sum over beams)
        log_l = -0.5 * np.sum(d**2 / (SIGMA_HIT**2), axis=1)
        log_l -= log_l.max()  # numerical stability

        self.weights *= np.exp(log_l)
        total = self.weights.sum()
        if total < 1e-300:
            self.weights.fill(1.0 / self.n)
        else:
            self.weights /= total

        # Adaptive resampling via effective sample size
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < RESAMPLE_THRESHOLD * self.n:
            self._systematic_resample()

    # ── Resampling ───────────────────────────────────────────────────

    def _systematic_resample(self) -> None:
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # guard against float drift
        u0 = np.random.uniform(0, 1.0 / self.n)
        u = u0 + np.arange(self.n) / self.n
        idx = np.searchsorted(cumsum, u)
        self.particles = self.particles[idx].copy()
        self.weights.fill(1.0 / self.n)

    # ── Pose estimate ────────────────────────────────────────────────

    def estimate(self) -> tuple[float, float, float]:
        """Weighted circular-mean pose."""
        x = float(np.average(self.particles[:, 0], weights=self.weights))
        y = float(np.average(self.particles[:, 1], weights=self.weights))
        sin_th = float(np.average(np.sin(self.particles[:, 2]), weights=self.weights))
        cos_th = float(np.average(np.cos(self.particles[:, 2]), weights=self.weights))
        theta = math.atan2(sin_th, cos_th)
        return x, y, theta

    def spread(self) -> float:
        """Weighted standard deviation of position (m) — convergence metric."""
        x, y, _ = self.estimate()
        dx = self.particles[:, 0] - x
        dy = self.particles[:, 1] - y
        return float(np.sqrt(np.average(dx**2 + dy**2, weights=self.weights)))


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
    """Subscribes to permitted input topics only, publishes steering +
    throttle commands at ~40 Hz."""

    def __init__(
        self,
        map_yaml: Path,
        checkpoint: Path,
        throttle_scale: float,
    ) -> None:
        super().__init__("racing_agent")

        # ── Load map ─────────────────────────────────────────────────
        free_mask, _, res, origin, img_h = _load_map(map_yaml)

        # ── Centerline ───────────────────────────────────────────────
        self.skel = _extract_centerline(free_mask, res, origin, img_h)
        self.n_wps = len(self.skel)
        self.skel_tree = KDTree(self.skel)
        seg_lens = np.linalg.norm(
            np.diff(self.skel, axis=0, append=self.skel[:1]), axis=1
        )
        self.look_step = max(1, round(1.0 / float(seg_lens.mean())))

        # ── Particle filter ──────────────────────────────────────────
        self.pf = ParticleFilter(free_mask, res, origin, img_h)
        self.localised = False
        self.convergence_thresh = 0.30  # m – position spread to consider localised

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

        # ── Lidar subsampling indices (1080 → 108) ──────────────────
        self.lidar_idx = np.round(np.linspace(0, N_LIDAR_RAW - 1, N_LIDAR_OBS)).astype(
            int
        )

        # ── Mutable state ────────────────────────────────────────────
        self.x = self.y = self.theta = 0.0
        self.speed = 0.0
        self.steer_fb = 0.0  # normalised [-1, 1]
        self.yaw_rate = 0.0
        self.lidar = np.full(N_LIDAR_RAW, LIDAR_MAX)
        self.left_ticks: int | None = None
        self.right_ticks: int | None = None
        self.prev_left: int | None = None
        self.prev_right: int | None = None
        self.prev_t: float | None = None
        self.throttle_scale = throttle_scale

        # ── Subscriptions (permitted input topics only) ──────────────
        qos = _qos()
        self.create_subscription(LaserScan, f"{TOPIC_NS}/lidar", self._on_lidar, qos)
        self.create_subscription(Float32, f"{TOPIC_NS}/steering", self._on_steer, qos)
        self.create_subscription(
            Float32, f"{TOPIC_NS}/throttle", self._on_throttle, qos
        )
        self.create_subscription(Imu, f"{TOPIC_NS}/imu", self._on_imu, qos)
        self.create_subscription(
            JointState, f"{TOPIC_NS}/left_encoder", self._on_left_enc, qos
        )
        self.create_subscription(
            JointState, f"{TOPIC_NS}/right_encoder", self._on_right_enc, qos
        )

        # ── Publishers ───────────────────────────────────────────────
        self.pub_steer = self.create_publisher(
            Float32, f"{TOPIC_NS}/steering_command", qos
        )
        self.pub_throttle = self.create_publisher(
            Float32, f"{TOPIC_NS}/throttle_command", qos
        )

        self.create_timer(1.0 / LOOP_HZ, self._tick)
        self.get_logger().info(
            f"Ready  wps={self.n_wps}  look_step={self.look_step}  "
            f"throttle={self.throttle_scale:.2f}  obs_dim={obs_dim}  "
            f"particles={N_PARTICLES}"
        )

    # ── Sensor callbacks ─────────────────────────────────────────────

    def _on_lidar(self, msg: LaserScan) -> None:
        self.lidar = np.asarray(msg.ranges, dtype=np.float64)

    def _on_steer(self, msg: Float32) -> None:
        self.steer_fb = float(msg.data)  # normalised [-1, 1]

    def _on_throttle(self, msg: Float32) -> None:
        pass  # not needed for obs; kept if future use

    def _on_imu(self, msg: Imu) -> None:
        self.yaw_rate = msg.angular_velocity.z

    def _on_left_enc(self, msg: JointState) -> None:
        if msg.position:
            self.left_ticks = int(round(msg.position[0]))

    def _on_right_enc(self, msg: JointState) -> None:
        if msg.position:
            self.right_ticks = int(round(msg.position[0]))

    # ── Odometry from encoders ───────────────────────────────────────

    def _encoder_velocity(self, dt: float) -> float:
        """Compute linear speed (m/s) from encoder tick deltas."""
        if (
            self.left_ticks is None
            or self.right_ticks is None
            or self.prev_left is None
            or self.prev_right is None
            or dt <= 0
        ):
            self.prev_left = self.left_ticks
            self.prev_right = self.right_ticks
            return self.speed  # hold previous estimate

        dl = (self.left_ticks - self.prev_left) * WHEEL_CIRC / TICKS_PER_REV
        dr = (self.right_ticks - self.prev_right) * WHEEL_CIRC / TICKS_PER_REV
        self.prev_left = self.left_ticks
        self.prev_right = self.right_ticks

        dist = (dl + dr) / 2.0
        v = dist / dt

        # Low-pass filter (τ ≈ 0.1 s)
        alpha = min(1.0, 10.0 * dt)
        self.speed += alpha * (v - self.speed)
        return self.speed

    # ── Observation vector ───────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        lidar = np.clip(self.lidar[self.lidar_idx], LIDAR_MIN, LIDAR_MAX)

        x, y, th = self.x, self.y, self.theta
        ch, sh = math.cos(th), math.sin(th)

        _, wi = self.skel_tree.query([x, y])
        cx, cy = self.skel[wi]
        nx, ny = self.skel[(wi + 1) % self.n_wps]
        cth = math.atan2(ny - cy, nx - cx)

        heading_err = (cth - th + math.pi) % (2.0 * math.pi) - math.pi
        lateral_err = -(x - cx) * math.sin(cth) + (y - cy) * math.cos(cth)

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

    # ── Main control loop ────────────────────────────────────────────

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = (
            float(np.clip(now - self.prev_t, 1e-4, 0.1))
            if self.prev_t is not None
            else 1.0 / LOOP_HZ
        )
        self.prev_t = now

        # Speed from encoders
        self._encoder_velocity(dt)

        # Steering angle for motion model
        steer_rad = self.steer_fb * STEER_MAX
        # Bicycle model: ω = v * tan(δ) / L
        omega_model = (
            self.speed * math.tan(steer_rad) / WHEELBASE
            if abs(self.speed) > 0.01
            else self.yaw_rate
        )

        # ── Particle filter predict + update ─────────────────────────
        self.pf.predict(self.speed, omega_model, dt)
        self.pf.update(self.lidar)
        self.x, self.y, self.theta = self.pf.estimate()

        spread = self.pf.spread()
        if not self.localised:
            if spread < self.convergence_thresh:
                self.localised = True
                self.get_logger().info(
                    f"Localised! spread={spread:.3f} m  "
                    f"pose=({self.x:.2f}, {self.y:.2f}, {math.degrees(self.theta):.1f}°)"
                )
            else:
                # Hold still while localising — send zero commands
                self.pub_steer.publish(Float32(data=0.0))
                self.pub_throttle.publish(Float32(data=0.0))
                return

        # ── Policy inference ─────────────────────────────────────────
        obs = self._build_obs()
        with torch.no_grad():
            t_obs = torch.tensor(
                obs, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            action = self.policy(t_obs).cpu().numpy().ravel().clip(-1.0, 1.0)

        # Rate-limited steering
        steer_cmd = float(
            np.clip(
                steer_rad + action[0] * STEER_VEL_MAX * dt,
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
        1.0, "-t", "--throttle", min=0.0, max=1.0, help="Throttle scale [0-1]."
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
