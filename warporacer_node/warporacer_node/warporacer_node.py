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

import numpy as np
import rclpy
import torch
import torch.nn as nn
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# Tiny MLP on CPU at 60 Hz: pin to one thread — faster here and removes the
# OpenMP thread-pool sync jitter from the control loop.
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


class Actor(nn.Module):
    """Reconstructs only the actor branch of `warporacer.main.Agent`."""

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs):
        return self.actor(obs)


def load_actor(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"checkpoint not found at {os.path.abspath(path)} — "
            "set WARPORACER_CHECKPOINT or run from the directory containing it"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    actor = Actor(cfg["obs_dim"], cfg["act_dim"], cfg["hidden"])
    sd = {k: v for k, v in ckpt["agent"].items() if k.startswith("actor.")}
    actor.load_state_dict(sd)
    actor.eval()
    obs_mean = ckpt["obs_mean"].numpy().astype(np.float32)
    obs_var = ckpt["obs_var"].numpy().astype(np.float32)
    return actor, obs_mean, obs_var, cfg


class WarporacerNode(Node):
    def __init__(self):
        super().__init__("warporacer")

        self.actor, obs_mean, obs_var, cfg = load_actor(checkpoint_path)
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
        self.lidar_idx = None  # nearest-beam lookup, built on first scan
        self.scan_range_min = range_min_floor
        self.last_scan_t = 0.0
        self.was_driving = False

        self.create_subscription(LaserScan, scan_topic, self.scan_cb, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.create_timer(self.dt, self.control_step)

        self.get_logger().info(
            f"warporacer loaded: obs_dim={self.obs_dim}, "
            f"num_lidar={self.num_lidar}, dt={self.dt:.4f}s, "
            f"speed_scale={self.speed_scale:.2f}, "
            f"v_cmd range=[{self.v_min_effective:.2f}, {self.v_max:.2f}] m/s"
        )

    def publish_drive(self, steering: float, speed: float):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
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
        sampled = ranges[self.lidar_idx].astype(np.float32, copy=True)
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

    def control_step(self):
        now = self.get_clock().now().nanoseconds * 1e-9
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

        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[0] = self.delta
        obs[1] = self.v_meas
        obs[2 : 2 + self.num_lidar] = self.lidar_buf

        norm = ((obs - self.obs_mean) * self.obs_inv_std).clip(-10.0, 10.0)
        with torch.inference_mode():
            act = self.actor(torch.from_numpy(norm)).numpy()

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
