import socket

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from numpy.lib.stride_tricks import sliding_window_view
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan

# Geometry / safety
bubble_size: int = 300  # lidar points per gap-search window
max_range: float = 20.0  # m, used for inf/NaN and rejected-beam fill
max_speed: float = 3.0  # m/s
slow_distance: float = 6.0  # m, speed ramps linearly below this
deadman_timeout: float = 0.3  # seconds since last "armed" packet
deadman_port: int = 5005  # UDP port for the deadman GUI
lidar_height: float = 0.10  # m above ground — measure on your car
wall_height: float = 0.20  # m, 8" walls


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

    def is_armed(self) -> bool:
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self.last_deadman) < deadman_timeout

    def watchdog(self):
        if not self.is_armed():
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

    def scan_callback(self, msg):
        if not self.is_armed():
            return
        ranges = np.where(np.isfinite(msg.ranges), msg.ranges, max_range)

        # Cache per-beam sin/cos on first scan (geometry is fixed).
        if self.cos_a is None or self.cos_a.size != ranges.size:
            ang = msg.angle_min + np.arange(ranges.size) * msg.angle_increment
            self.cos_a, self.sin_a = np.cos(ang), np.sin(ang)

        # Reject beams that miss the 0..wall_height band given current tilt.
        qx, qy, qz, qw = self.q
        a = 2.0 * (qx * qz - qw * qy)
        b = 2.0 * (qy * qz + qw * qx)
        hit_z = lidar_height + ranges * (a * self.cos_a + b * self.sin_a)
        ranges = np.where((hit_z < 0.0) | (hit_z > wall_height), max_range, ranges)

        # Trim noisy edges; pick window center with max-min clearance.
        sixth = ranges.size // 6
        ranges = ranges[sixth:-sixth]
        windows = sliding_window_view(ranges, bubble_size)
        i = int(np.argmax(windows.min(axis=1))) + bubble_size // 2

        steering = 2.0 * i / (ranges.size - 1) - 1.0
        speed = min(ranges[i] / slow_distance, 1.0) * max_speed
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
