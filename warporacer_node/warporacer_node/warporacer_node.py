#!/usr/bin/env python3
"""Warporacer inference node.

Runs the trained PPO actor from `warporacer/main.py` on a real car. The
observation matches training exactly:

    obs[0]              = commanded steering angle (rad, integrated locally)
    obs[1]              = forward velocity (m/s, from odom)
    obs[2 : 2+N_LIDAR]  = downsampled LIDAR ranges (m)
    obs[2+N_LIDAR : +5] = pose prior, only for checkpoints trained with --pose:
                          where map -> base_link (cartographer localization in the
                          training map) puts the car relative to the nearest
                          centerline waypoint; see pose_features()

The actor outputs (steer_v_norm, accel_norm) in [-1, 1]; both are integrated
into the steering and speed setpoints we publish on /drive, matching the sim
integration scheme.

Driving is gated on fresh lidar (and, for --pose checkpoints, on cartographer
having matched the car to the saved map and a fresh pose); the deadman_switch
node gates the motor output
downstream. While our /drive isn't reaching the motors (/deadman_armed false:
held still, or driven manually from the page) the integrated steering and speed
are reset, so they can't wind up and lurch when control comes back.
The published steering_angle keeps the policy's sign
(sim: +delta = left turn); the VESC applies the servo polarity/gain downstream.
"""

import os

# Pin BLAS/OpenMP to a single thread BEFORE numpy imports its backend. The actor
# is a tiny MLP; multi-threaded BLAS only adds thread-dispatch jitter to the
# 60 Hz control loop for matrices this small.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray

# torch is used only to load the checkpoint (inference is pure numpy); keep it
# single-threaded so loading doesn't spin up a thread pool.
torch.set_num_threads(1)

# Operational knobs — overridable via env vars for bring-up without recompiling.
# Default: warporacer_node/agent_final.pt, installed by setup.py, so the node
# doesn't depend on the directory it's launched from.
checkpoint_path: str = os.environ.get(
    "WARPORACER_CHECKPOINT",
    os.path.join(get_package_share_directory("warporacer_node"), "agent_final.pt"),
)
speed_scale: float = float(os.environ.get("WARPORACER_SPEED_SCALE", "1.0"))
inference_v_min: float = float(os.environ.get("WARPORACER_V_MIN", "1.0"))

# Forward velocity (obs[1]) comes from the car's odometry (race_bringup imu_odom).
odom_topic: str = os.environ.get("WARPORACER_ODOM_TOPIC", "/odom")
scan_topic: str = "/scan"
control_hz: float = 60.0
# Steering authority multiplier on the policy's commanded angle. 1.0 = full
# trained authority; <1 detunes a twitchy car. NOT a sign flip — sim +delta =
# left already matches /drive, and the VESC owns servo polarity/gain.
steer_gain: float = float(os.environ.get("WARPORACER_STEER_GAIN", "0.7"))
steer_clamp: float = 0.95
# Cap on how far the commanded speed may lead the measured speed. Prevents v_cmd
# winding up to v_max while the car is held still, e.g. against a wall (a
# full-speed lurch when it frees) without limiting normal acceleration.
v_lead_margin: float = float(os.environ.get("WARPORACER_V_LEAD_MARGIN", "2.0"))
scan_stale_timeout: float = 0.2  # s, max age of latest scan before we refuse to drive
# Pose prior (--pose checkpoints): map -> base_link from cartographer localization.
map_frame: str = "map"
base_frame: str = "base_link"
pose_stale_timeout: float = (
    0.3  # s, max age of map -> base_link before we refuse to drive
)
# Warn if the lidar FOV under-covers the training FOV by more than this.
fov_tolerance: float = np.radians(2.0)
range_min_floor: float = 0.05  # hard lower bound used if msg.range_min is unset/zero


def pose_features(x, y, yaw, centerline, out):
    """Pose prior exactly as warporacer's sim computes it (sim.py step_kernel, minus
    the training noise): progress, signed offset (+ = left) and heading relative to
    the centerline waypoint nearest (x, y). centerline is (M, 3): x, y, heading."""
    k = int(np.argmin((centerline[:, 0] - x) ** 2 + (centerline[:, 1] - y) ** 2))
    cx, cy, th = centerline[k]
    phase = 2.0 * np.pi * k / len(centerline)
    d = yaw - th
    out[0] = np.sin(phase)
    out[1] = np.cos(phase)
    out[2] = -(x - cx) * np.sin(th) + (y - cy) * np.cos(th)
    out[3] = np.sin(d)
    out[4] = np.cos(d)


def load_actor(path: str):
    """Pull the actor MLP weights from the checkpoint as plain numpy arrays.

    Inference then runs as a hand-rolled `W @ x + b` + tanh forward pass (see
    control_step): no torch in the 60 Hz loop, which removes the torch
    dispatch/thread-pool jitter that was the main source of control-rate
    variation. torch is used only here, to read the checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"checkpoint not found at {os.path.abspath(path)} — "
            "put agent_final.pt in race_stack/warporacer_node/ and rebuild, "
            "or set WARPORACER_CHECKPOINT"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sd = ckpt["agent"]

    def arr(key):
        # torch Linear weight is (out, in); y = W @ x + b matches x @ W.T + b.
        return np.ascontiguousarray(sd[key].numpy(), dtype=np.float32)

    # actor.0/2/4 are the three Linear layers (the Tanhs in between have no params).
    weights = (
        arr("actor.0.weight"),
        arr("actor.0.bias"),
        arr("actor.2.weight"),
        arr("actor.2.bias"),
        arr("actor.4.weight"),
        arr("actor.4.bias"),
    )
    obs_mean = ckpt["obs_mean"].numpy().astype(np.float32)
    obs_var = ckpt["obs_var"].numpy().astype(np.float32)
    return weights, obs_mean, obs_var, cfg


class WarporacerNode(Node):
    def __init__(self):
        super().__init__("warporacer")

        weights, obs_mean, obs_var, cfg = load_actor(checkpoint_path)
        self.w0, self.b0, self.w1, self.b1, self.w2, self.b2 = weights
        # Preallocated hidden-layer buffers so the per-tick forward pass does no
        # heap allocation (filled in place via np.dot(out=...)).
        self.h0 = np.zeros(self.w0.shape[0], dtype=np.float32)
        self.h1 = np.zeros(self.w1.shape[0], dtype=np.float32)
        self.act_buf = np.zeros(self.w2.shape[0], dtype=np.float32)
        self.obs_mean = obs_mean
        self.obs_inv_std = 1.0 / np.sqrt(obs_var + 1e-8)

        self.obs_dim = int(cfg["obs_dim"])
        self.num_lidar = int(cfg["num_lidar"])
        self.lidar_fov = float(cfg["lidar_fov"])
        self.lidar_range = float(cfg["lidar_range"])
        self.steer_v_max = float(cfg["steer_v_max"])
        self.a_max = float(cfg["a_max"])
        self.steer_min = float(cfg["steer_min"])
        self.steer_max = float(cfg["steer_max"])
        self.v_min = float(cfg["v_min"])
        self.v_max = float(cfg["v_max"])
        self.v_min_effective = max(self.v_min, inference_v_min)
        self.speed_scale = float(np.clip(speed_scale, 0.0, 1.0))
        # Use the training-time dt so the integration of delta/v_cmd matches
        # what the policy was trained on. Fall back to control_hz only for
        # legacy checkpoints that didn't persist it.
        self.dt = float(cfg.get("dt", 1.0 / control_hz))
        self.pose = bool(cfg.get("pose", False))
        n_pose = 5 if self.pose else 0
        if self.obs_dim != 2 + self.num_lidar + n_pose:
            raise ValueError(
                f"checkpoint obs_dim {self.obs_dim} != 2 + {self.num_lidar} lidar "
                f"+ {n_pose} pose"
            )

        self.target_angles = np.linspace(
            -self.lidar_fov / 2, self.lidar_fov / 2, self.num_lidar, dtype=np.float32
        )

        # Internal sim-mirrored state (delta, v) — delta has no on-car sensor,
        # so we integrate the commanded steering velocity ourselves. v_cmd is
        # re-seeded from v_meas whenever driving resumes (lidar fresh again, or
        # the deadman arms; see control_step) so the VESC sees no speed jump.
        self.delta = 0.0
        self.v_cmd = 0.0
        self.v_meas = 0.0
        self.lidar_buf = np.zeros(self.num_lidar, dtype=np.float32)
        self.pose_buf = np.zeros(n_pose, dtype=np.float32)
        if self.pose:
            self.centerline = np.asarray(cfg["centerline"], dtype=np.float64)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            # Until cartographer matches the localization run to the saved map it
            # reports the car at the map origin, so don't trust map -> base_link yet.
            self.localized = False
            self.create_subscription(
                MarkerArray, "/constraint_list", self.constraints_cb, 1
            )
        # Persistent control-loop buffers — filled in place each tick to avoid
        # per-tick heap allocation in the 60 Hz loop.
        self.obs_buf = np.zeros(self.obs_dim, dtype=np.float32)
        self.norm_buf = np.zeros(self.obs_dim, dtype=np.float32)
        self.lidar_idx = None  # nearest-beam lookup, rebuilt if the scan size changes
        self.lidar_n_in = 0
        self.scan_range_min = range_min_floor
        self.last_scan_t = 0.0
        self.was_driving = False
        self.deadman_armed = True  # until the deadman says otherwise
        # Control-rate diagnostics: count ticks over a ~1 s window and report the
        # achieved Hz plus the worst tick-to-tick gap (jitter). If this prints
        # well under target, the loop is integrating delta/v_cmd with a dt faster
        # than it actually runs — the classic cause of steering oscillation.
        self._hz_count = 0
        self._hz_t0 = 0.0
        self._hz_last_t = 0.0
        self._hz_max_dt = 0.0

        self.create_subscription(LaserScan, scan_topic, self.scan_cb, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.create_subscription(Bool, "/deadman_armed", self.deadman_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self._drive_msg = AckermannDriveStamped()
        self._drive_msg.header.frame_id = "base_link"

        self.create_timer(self.dt, self.control_step)

        if self.pose:
            seg = np.diff(
                self.centerline[:, :2], axis=0, append=self.centerline[:1, :2]
            )
            self.get_logger().info(
                f"pose prior on: {map_frame} -> {base_frame} from cartographer, "
                f"trained on {cfg.get('map', '?')} ({len(self.centerline)} waypoints, "
                f"{np.hypot(*seg.T).sum():.1f} m loop); start localization in that map"
            )
        self.get_logger().info(
            f"warporacer loaded: obs_dim={self.obs_dim}, "
            f"num_lidar={self.num_lidar}, dt={self.dt:.4f}s, "
            f"speed_scale={self.speed_scale:.2f}, "
            f"v_cmd range=[{self.v_min_effective:.2f}, {self.v_max:.2f}] m/s"
        )

    def publish_drive(self, steering: float, speed: float):
        m = self._drive_msg
        m.header.stamp = self.get_clock().now().to_msg()
        m.drive.steering_angle = float(steering)
        m.drive.speed = float(speed)
        self.drive_pub.publish(m)

    def scan_cb(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if self.lidar_idx is None or ranges.size != self.lidar_n_in:
            self.lidar_n_in = ranges.size
            in_angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment
            self.lidar_idx = np.abs(
                in_angles[None, :] - self.target_angles[:, None]
            ).argmin(axis=1)
            self.scan_range_min = max(float(msg.range_min), range_min_floor)
            fov_in = float(msg.angle_max - msg.angle_min)
            fov_train = float(self.lidar_fov)
            if fov_in + fov_tolerance < fov_train:
                self.get_logger().warn(
                    f"lidar FOV {np.degrees(fov_in):.1f}° is smaller than "
                    f"training FOV {np.degrees(fov_train):.1f}° — out-of-range "
                    "target beams will reuse the edge value (distribution shift)."
                )
            self.get_logger().info(
                f"lidar mapped: {ranges.size} input beams → {self.num_lidar} "
                f"trained beams; angle_min={msg.angle_min:.3f}, "
                f"angle_max={msg.angle_max:.3f}, "
                f"invalid<{self.scan_range_min:.3f}m → {self.lidar_range:.1f}m"
            )
        sampled = ranges[self.lidar_idx]
        np.nan_to_num(
            sampled,
            copy=False,
            nan=self.lidar_range,
            posinf=self.lidar_range,
            neginf=self.lidar_range,
        )
        sampled[sampled < self.scan_range_min] = self.lidar_range
        np.clip(sampled, 0.0, self.lidar_range, out=sampled)
        self.lidar_buf = sampled
        self.last_scan_t = self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg: Odometry):
        self.v_meas = float(msg.twist.twist.linear.x)

    def deadman_cb(self, msg: Bool):
        self.deadman_armed = msg.data

    def constraints_cb(self, msg: MarkerArray):
        # Constraints between the localization trajectory and the saved map's.
        if not self.localized and any(
            m.ns == "Inter constraints, different trajectories" and m.points
            for m in msg.markers
        ):
            self.localized = True
            self.get_logger().info("localized: cartographer matched the car to the map")

    def update_pose(self, now: float) -> bool:
        """Fill pose_buf from the latest map -> base_link; False if not localized yet,
        missing or stale."""
        if not self.localized:
            self.get_logger().warn(
                "waiting for cartographer to find the car on the map (about 10 s "
                "after localization starts)",
                throttle_duration_sec=2.0,
            )
            return False
        try:
            tf = self.tf_buffer.lookup_transform(map_frame, base_frame, Time())
        except TransformException as e:
            self.get_logger().warn(
                f"no {map_frame} -> {base_frame} ({e}); is cartographer localization "
                "running?",
                throttle_duration_sec=2.0,
            )
            return False
        age = now - Time.from_msg(tf.header.stamp).nanoseconds * 1e-9
        if age > pose_stale_timeout:
            self.get_logger().warn(
                f"{map_frame} -> {base_frame} is {age:.2f} s old",
                throttle_duration_sec=2.0,
            )
            return False
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        pose_features(t.x, t.y, yaw, self.centerline, self.pose_buf)
        if self.pose_buf[4] < 0.0:
            self.get_logger().warn(
                "car faces against the trained direction of travel (or localization "
                "is off): turn it around, or retrain with --reverse",
                throttle_duration_sec=2.0,
            )
        return True

    def _track_rate(self, now: float):
        if self._hz_last_t > 0.0:
            self._hz_max_dt = max(self._hz_max_dt, now - self._hz_last_t)
        self._hz_last_t = now
        if self._hz_t0 == 0.0:
            self._hz_t0 = now
            return
        self._hz_count += 1
        elapsed = now - self._hz_t0
        if elapsed >= 1.0:
            hz = self._hz_count / elapsed
            self.get_logger().info(
                f"control loop {hz:.1f} Hz (target {1.0 / self.dt:.0f}), "
                f"worst gap {self._hz_max_dt * 1e3:.1f} ms"
            )
            self._hz_t0 = now
            self._hz_count = 0
            self._hz_max_dt = 0.0

    def control_step(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        self._track_rate(now)
        scan_fresh = (
            self.last_scan_t > 0.0 and (now - self.last_scan_t) < scan_stale_timeout
        )
        pose_fresh = not self.pose or self.update_pose(now)

        if not scan_fresh or not pose_fresh or not self.deadman_armed:
            # No fresh lidar or pose (fail safe rather than drive on stale data), or
            # the deadman isn't passing our /drive through. Drop the integrated
            # setpoints so we don't lurch when sensing resumes or control returns.
            self.delta = 0.0
            self.v_cmd = 0.0
            self.was_driving = False
            self.publish_drive(0.0, 0.0)
            return
        if not self.was_driving:
            # First tick back: re-seed the commanded speed from the
            # actual speed so the inner VESC loop doesn't see a brake-then-accel
            # jump.
            self.v_cmd = float(np.clip(self.v_meas, self.v_min_effective, self.v_max))
            self.was_driving = True

        obs = self.obs_buf
        obs[0] = self.delta
        obs[1] = self.v_meas
        obs[2 : 2 + self.num_lidar] = self.lidar_buf
        obs[2 + self.num_lidar :] = self.pose_buf

        norm = self.norm_buf
        np.subtract(obs, self.obs_mean, out=norm)
        np.multiply(norm, self.obs_inv_std, out=norm)
        np.clip(norm, -10.0, 10.0, out=norm)

        # Actor forward pass in pure numpy (two tanh hidden layers), computed in
        # place so the control loop does no allocation and no torch dispatch.
        np.dot(self.w0, norm, out=self.h0)
        np.add(self.h0, self.b0, out=self.h0)
        np.tanh(self.h0, out=self.h0)
        np.dot(self.w1, self.h0, out=self.h1)
        np.add(self.h1, self.b1, out=self.h1)
        np.tanh(self.h1, out=self.h1)
        np.dot(self.w2, self.h1, out=self.act_buf)
        np.add(self.act_buf, self.b2, out=self.act_buf)
        act = self.act_buf

        steer_v = float(np.clip(act[0], -1.0, 1.0)) * self.steer_v_max
        accel = float(np.clip(act[1], -1.0, 1.0)) * self.a_max

        self.delta = float(
            np.clip(self.delta + steer_v * self.dt, self.steer_min, self.steer_max)
        )
        self.v_cmd = float(
            np.clip(
                self.v_cmd + accel * self.dt,
                self.v_min_effective,
                min(self.v_max, self.v_meas + v_lead_margin),
            )
        )

        steer_out = steer_gain * float(
            np.clip(
                self.delta, -steer_clamp * self.steer_max, steer_clamp * self.steer_max
            )
        )
        self.publish_drive(steer_out, self.v_cmd * self.speed_scale)


def main(args=None):
    rclpy.init(args=args)
    node = WarporacerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
