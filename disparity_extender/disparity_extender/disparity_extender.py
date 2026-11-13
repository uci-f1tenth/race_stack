#!/usr/bin/env python3

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from numpy.lib.stride_tricks import sliding_window_view
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# Geometry / safety
bubble_size: int = 300  # lidar points per gap-search window
max_range: float = 20.0  # m, used for inf/NaN and rejected-beam fill
max_speed: float = 1.0  # m/s
slow_distance: float = 4.0  # m, speed ramps linearly below this
turn_slowdown: float = 0.7  # 0..1, fraction of max speed shaved at full lock
min_speed_factor: float = 0.3  # floor on the steering-based speed multiplier
steering_p: float = -0.8  # proportional gain on normalized steering
steering_clamp: float = 0.95  # symmetric normalized steering clamp
scan_stale_timeout: float = 0.2  # s, stop the car if no fresh /scan within this


class DisparityExtender(Node):
    def __init__(self):
        super().__init__("disparity_extender")
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
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

    def scan_callback(self, msg):
        self.last_scan_t = self.get_clock().now().nanoseconds * 1e-9
        ranges = np.array(msg.ranges, dtype=np.float32)
        np.nan_to_num(
            ranges, copy=False, nan=max_range, posinf=max_range, neginf=max_range
        )

        # Trim noisy edges; pick window center with max-min clearance.
        sixth = ranges.size // 6
        if sixth > 0:
            ranges = ranges[sixth:-sixth]
        n_win = ranges.size - bubble_size + 1
        if n_win <= 0:
            self.publish_drive(0.0, 0.0)
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
