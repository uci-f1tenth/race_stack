import socket

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# Constants
min_angle: float = -np.pi / 2.0  # radians
max_angle: float = np.pi / 2.0  # radians
bubble_size: int = 300  # lidar points
deadman_timeout: float = 0.3  # seconds since last "armed" packet
deadman_port: int = 5005  # UDP port for the deadman GUI


def index_to_angle(index: int, num_points: int) -> float:
    angle_increment = (max_angle - min_angle) / (num_points - 1)
    angle = min_angle + index * angle_increment
    return angle


def find_best_point(lidar_range_array: np.ndarray) -> int:
    best_index = 0
    best_min_distance = 0.0
    for i in range(len(lidar_range_array) - bubble_size + 1):
        window = lidar_range_array[i : i + bubble_size]
        min_distance = np.min(window)
        if min_distance > best_min_distance:
            best_min_distance = min_distance
            best_index = i
    return best_index + bubble_size // 2


def compute_speed(target_distance: float) -> float:
    if target_distance < 6.0:
        return target_distance / 6.0
    return 1.0


class DisparityExtender(Node):
    def __init__(self):
        super().__init__("disparity_extender")
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
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
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = self.get_clock().now().to_msg()
        ack_msg.header.frame_id = "base_link"
        ack_msg.drive.steering_angle = steering
        ack_msg.drive.speed = speed
        self.drive_pub.publish(ack_msg)

    def scan_callback(self, msg):
        if not self.is_armed():
            return
        lidar_range_array = np.array(msg.ranges)
        lidar_range_array = np.where(
            np.isfinite(lidar_range_array), lidar_range_array, 20.0
        )  # Clipping infinity and NaN
        sixth = lidar_range_array.size // 6
        lidar_range_array = lidar_range_array[sixth:-sixth]
        best_point_index = find_best_point(lidar_range_array)
        best_point_angle = index_to_angle(best_point_index, lidar_range_array.size)
        steering = best_point_angle / (np.pi / 2.0)
        target_distance = lidar_range_array[best_point_index]
        speed = compute_speed(target_distance)
        self.publish_drive(steering, speed)


def main(args=None):
    rclpy.init(args=args)
    disparity_extender = DisparityExtender()
    try:
        rclpy.spin(disparity_extender)
    except KeyboardInterrupt:
        pass
    finally:
        disparity_extender.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
