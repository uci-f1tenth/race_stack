#!/usr/bin/env python3
"""Warporacer inference node.

Runs the trained PPO actor from `warporacer/main.py` on a real car using only
on-board sensors. The observation matches training exactly:

    obs[0]              = commanded steering angle (rad, integrated locally)
    obs[1]              = forward velocity (m/s, from odom)
    obs[2 : 2+N_LIDAR]  = downsampled LIDAR ranges (m)

The actor outputs (steer_v_norm, accel_norm) in [-1, 1]; both are integrated
into the steering and speed setpoints we publish on /drive, matching the sim
integration scheme.

Driving is gated on fresh lidar; the race-stack joystick deadman gates the
motor output downstream. The published steering_angle keeps the policy's sign
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
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# torch is used only to load the checkpoint (inference is pure numpy); keep it
# single-threaded so loading doesn't spin up a thread pool.
torch.set_num_threads(1)

# Operational knobs — overridable via env vars for bring-up without recompiling.
checkpoint_path: str = os.environ.get("WARPORACER_CHECKPOINT", "agent_final.pt")
speed_scale: float = float(os.environ.get("WARPORACER_SPEED_SCALE", "1.0"))
inference_v_min: float = float(os.environ.get("WARPORACER_V_MIN", "0.0"))

# VESC odometry carries forward velocity (obs[1]); /pf/pose/odom only exists in
# the AutoDRIVE sim bridge, so default to the real-car /odom. Override per car.
odom_topic: str = os.environ.get("WARPORACER_ODOM_TOPIC", "/odom")
scan_topic: str = "/scan"
control_hz: float = 60.0
# Steering authority multiplier on the policy's commanded angle. 1.0 = full
# trained authority; <1 detunes a twitchy car. NOT a sign flip — sim +delta =
# left already matches /drive, and the VESC owns servo polarity/gain.
steer_gain: float = float(os.environ.get("WARPORACER_STEER_GAIN", "0.7"))
steer_clamp: float = 0.95
# Cap on how far the commanded speed may lead the measured speed. Prevents v_cmd
# winding up to v_max while the car is held still under the mux (a full-speed
# lurch on arm) without limiting normal acceleration (the VESC tracks v_cmd).
v_lead_margin: float = float(os.environ.get("WARPORACER_V_LEAD_MARGIN", "2.0"))
scan_stale_timeout: float = 0.2  # s, max age of latest scan before we refuse to drive
fov_tolerance: float = np.radians(
    2.0
)  # warn if lidar FOV under-covers training FOV by more than this
range_min_floor: float = 0.05  # hard lower bound used if msg.range_min is unset/zero


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
            "set WARPORACER_CHECKPOINT or run from the directory containing it"
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
        (self.w0, self.b0, self.w1, self.b1, self.w2, self.b2) = weights
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

        self.target_angles = np.linspace(
            -self.lidar_fov / 2, self.lidar_fov / 2, self.num_lidar, dtype=np.float32
        )

        # Internal sim-mirrored state (delta, v) — delta has no on-car sensor,
        # so we integrate the commanded steering velocity ourselves. v_cmd is
        # re-seeded from v_meas whenever lidar becomes fresh again (see
        # control_step) so we don't slam the brakes after a scan dropout.
        self.delta = 0.0
        self.v_cmd = 0.0
        self.v_meas = 0.0
        self.lidar_buf = np.zeros(self.num_lidar, dtype=np.float32)
        # Persistent control-loop buffers — filled in place each tick to avoid
        # per-tick heap allocation in the 60 Hz loop.
        self.obs_buf = np.zeros(self.obs_dim, dtype=np.float32)
        self.norm_buf = np.zeros(self.obs_dim, dtype=np.float32)
        self.lidar_idx = None  # nearest-beam lookup, built on first scan
        self.scan_range_min = range_min_floor
        self.last_scan_t = 0.0
        self.was_driving = False
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
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self._drive_msg = AckermannDriveStamped()
        self._drive_msg.header.frame_id = "base_link"

        self.create_timer(self.dt, self.control_step)

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
        if self.lidar_idx is None or self.lidar_idx.size != self.num_lidar:
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

    def _track_rate(self, now: float):
        if self._hz_last_t > 0.0:
            dt = now - self._hz_last_t
            if dt > self._hz_max_dt:
                self._hz_max_dt = dt
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

        if not scan_fresh:
            # No fresh lidar — fail safe rather than drive on stale data. Drop
            # the integrated setpoints so we don't lurch when scans resume.
            self.delta = 0.0
            self.v_cmd = 0.0
            self.was_driving = False
            self.publish_drive(0.0, 0.0)
            return
        if not self.was_driving:
            # First fresh-lidar tick: re-seed the commanded speed from the
            # actual speed so the inner VESC loop doesn't see a brake-then-accel
            # jump.
            self.v_cmd = float(np.clip(self.v_meas, self.v_min_effective, self.v_max))
            self.was_driving = True

        obs = self.obs_buf
        obs[0] = self.delta
        obs[1] = self.v_meas / speed_scale
        obs[2 : 2 + self.num_lidar] = self.lidar_buf

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
