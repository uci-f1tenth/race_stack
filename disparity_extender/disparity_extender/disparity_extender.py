#!/usr/bin/env python3

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from numpy.lib.stride_tricks import sliding_window_view
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan

# Geometry / safety
bubble_size: int = 300  # lidar points per gap-search window
max_range: float = 20.0  # m, used for inf/NaN and rejected-beam fill
max_speed: float = 1.0  # m/s
slow_distance: float = 4.0  # m, speed ramps linearly below this
turn_slowdown: float = 0.7  # 0..1, fraction of max speed shaved at full lock
min_speed_factor: float = 0.3  # floor on the steering-based speed multiplier
lidar_height: float = 0.10  # m above ground — measure on your car
wall_height: float = 0.20  # m, 8" walls
tilt_reject_threshold: float = 0.05  # only reject beams when actually tilted
steering_p: float = -0.8  # proportional gain on normalized steering
steering_clamp: float = 0.95  # symmetric normalized steering clamp
scan_stale_timeout: float = 0.2  # s, stop the car if no fresh /scan within this


class DisparityExtender(Node):
    def __init__(self):
        super().__init__("disparity_extender")
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.imu_sub = self.create_subscription(Imu, "/sensors/imu", self.imu_cb, 50)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.q = (0.0, 0.0, 0.0, 1.0)
        self.cos_a = self.sin_a = None  # per-beam sin/cos, cached on first scan
        # Failsafe: the only place we publish /drive is scan_callback, so a lidar
        # dropout would otherwise freeze the last command. This watchdog brakes
        # if /scan goes stale (mirrors the warporacer node).
        self.last_scan_t = 0.0
        self.create_timer(0.05, self._watchdog)

    def _watchdog(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_scan_t == 0.0 or (now - self.last_scan_t) > scan_stale_timeout:
            self.publish_drive(0.0, 0.0)

    def publish_drive(self, steering: float, speed: float):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.drive.steering_angle = steering
        m.drive.speed = speed
        self.drive_pub.publish(m)

    def imu_cb(self, msg):
        o = msg.orientation
        self.q = (o.x, o.y, o.z, o.w)

    def _extend_disparities(self, ranges):
        """Distance-aware disparity extension: pad each range jump by the
        angular width subtended by car_half_width at the closer range."""
        out = ranges.copy()
        n = ranges.size
        diffs = np.diff(ranges)
        inc = self.angle_increment

        # Range jumps UP: current beam is the close edge, extend right.
        for i in np.flatnonzero(diffs > disparity_threshold):
            r = ranges[i]
            pad = int(np.ceil(np.arctan2(car_half_width, max(r, 0.05)) / inc))
            end = min(n, i + 1 + pad)
            np.minimum(out[i:end], r, out=out[i:end])

        # Range jumps DOWN: next beam is the close edge, extend left.
        for i in np.flatnonzero(diffs < -disparity_threshold):
            r = ranges[i + 1]
            pad = int(np.ceil(np.arctan2(car_half_width, max(r, 0.05)) / inc))
            start = max(0, i + 1 - pad)
            np.minimum(out[start : i + 2], r, out=out[start : i + 2])

        return out

    def scan_callback(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32).copy()
        np.nan_to_num(
            ranges, copy=False, nan=max_range, posinf=max_range, neginf=max_range
        )

        # Cache per-beam sin/cos on first scan (geometry is fixed).
        if self.cos_a is None or self.cos_a.size != ranges.size:
            ang = msg.angle_min + np.arange(ranges.size) * msg.angle_increment
            self.cos_a, self.sin_a = np.cos(ang), np.sin(ang)
            self.angle_increment = msg.angle_increment

        # Reject beams that miss the 0..wall_height band — only when the IMU
        # actually reports a tilt, so flat-ground noise can't blank far beams.
        qx, qy, qz, qw = self.q
        a = 2.0 * (qx * qz - qw * qy)
        b = 2.0 * (qy * qz + qw * qx)
        if abs(a) > tilt_reject_threshold or abs(b) > tilt_reject_threshold:
            hit_z = lidar_height + ranges * (a * self.cos_a + b * self.sin_a)
            ranges = np.where((hit_z < 0.0) | (hit_z > wall_height), max_range, ranges)

        # Disparity extender — fill gaps at range discontinuities.
        # ranges = self._extend_disparities(ranges)

        # Trim noisy edges; pick window center with max-min clearance.
        sixth = ranges.size // 6
        if sixth > 0:
            ranges = ranges[sixth:-sixth]
        n_win = ranges.size - bubble_size + 1
        if n_win <= 0:
            return
        w_idx = int(np.argmax(sliding_window_view(ranges, bubble_size).min(axis=1)))
        i = w_idx + bubble_size // 2

        # Steering: normalize over achievable window centers so the [-1, 1]
        # range is actually reachable. Speed: forward clearance × turn penalty.
        if n_win > 1:
            steering_raw = steering_p * (2.0 * w_idx / (n_win - 1) - 1.0)
        else:
            steering_raw = 0.0
        steering = -float(np.clip(steering_raw, -steering_clamp, steering_clamp))
        speed_d = min(ranges[i] / slow_distance, 1.0)
        speed_s = max(1.0 - abs(steering) * turn_slowdown, min_speed_factor)
        speed = max_speed * min(speed_d, speed_s)
        self.publish_drive(steering, speed)


def main(args=None):
    rclpy.init(args=args)
    node = DisparityExtender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
