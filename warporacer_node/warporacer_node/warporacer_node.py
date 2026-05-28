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

Safety scheme matches pure_pursuit / disparity_extender: UDP deadman packet
+ watchdog. /drive is sign-flipped to match the F1TENTH servo polarity.
"""

import socket

import numpy as np
import rclpy
import torch
import torch.nn as nn
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

checkpoint_path: str = "agent_final.pt"
odom_topic: str = "/pf/pose/odom"
scan_topic: str = "/scan"
control_hz: float = 60.0
steer_sign: float = -1.0  # match F1TENTH servo convention
steer_clamp: float = 0.95
deadman_timeout: float = 0.3
deadman_port: int = 5005


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
        self.dt = 1.0 / control_hz

        self.target_angles = np.linspace(
            -self.lidar_fov / 2, self.lidar_fov / 2, self.num_lidar, dtype=np.float32
        )

        # Internal sim-mirrored state (delta, v) — delta has no on-car sensor,
        # so we integrate the commanded steering velocity ourselves.
        self.delta = 0.0
        self.v_cmd = 0.0
        self.v_meas = 0.0
        self.lidar_buf = np.zeros(self.num_lidar, dtype=np.float32)
        self.lidar_idx = None  # nearest-beam lookup, built on first scan
        self.have_scan = False

        self.create_subscription(LaserScan, scan_topic, self.scan_cb, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", deadman_port))
        self.sock.setblocking(False)
        self.last_deadman = 0.0

        self.create_timer(0.02, self.poll_deadman)
        self.create_timer(self.dt, self.control_step)

        self.get_logger().info(
            f"warporacer loaded: obs_dim={self.obs_dim}, "
            f"num_lidar={self.num_lidar}, dt={self.dt:.4f}s"
        )

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

    def is_armed(self) -> bool:
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self.last_deadman) < deadman_timeout

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
        sampled = ranges[self.lidar_idx]
        np.nan_to_num(
            sampled,
            copy=False,
            nan=self.lidar_range,
            posinf=self.lidar_range,
            neginf=self.lidar_range,
        )
        np.clip(sampled, 0.0, self.lidar_range, out=sampled)
        self.lidar_buf = sampled
        self.have_scan = True

    def odom_cb(self, msg: Odometry):
        self.v_meas = float(msg.twist.twist.linear.x)

    def control_step(self):
        if not self.is_armed():
            # Disarmed: zero outputs and forget integrated setpoints so we
            # don't lurch when re-armed.
            self.delta = 0.0
            self.v_cmd = 0.0
            self.publish_drive(0.0, 0.0)
            return
        if not self.have_scan:
            self.publish_drive(0.0, 0.0)
            return

        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[0] = self.delta
        obs[1] = self.v_meas
        obs[2 : 2 + self.num_lidar] = self.lidar_buf

        norm = ((obs - self.obs_mean) * self.obs_inv_std).clip(-10.0, 10.0)
        with torch.no_grad():
            act = self.actor(torch.from_numpy(norm)).numpy()

        steer_v = float(np.clip(act[0], -1.0, 1.0)) * self.steer_v_max
        accel = float(np.clip(act[1], -1.0, 1.0)) * self.a_max

        self.delta = float(
            np.clip(self.delta + steer_v * self.dt, self.steer_min, self.steer_max)
        )
        self.v_cmd = float(
            np.clip(self.v_cmd + accel * self.dt, self.v_min, self.v_max)
        )

        steer_out = steer_sign * float(
            np.clip(self.delta / self.steer_max, -steer_clamp, steer_clamp)
            * self.steer_max
        )
        self.publish_drive(steer_out, self.v_cmd)


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
