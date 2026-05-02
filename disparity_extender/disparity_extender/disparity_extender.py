from typing import Any

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

# Constants
min_angle: float = -np.pi / 2.0  # radians
max_angle: float = np.pi / 2.0  # radians
bubble_size: int = 120  # lidar points


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

    def scan_callback(self, msg):

        lidar_range_array = np.array(msg.ranges)
        lidar_range_array = np.where(
            np.isinf(lidar_range_array), 20.0, lidar_range_array
        )  # Clipping infinity
        sixth = lidar_range_array.size // 6
        lidar_range_array = lidar_range_array[sixth:-sixth]

        best_point_index = find_best_point(lidar_range_array)

        best_point_angle = index_to_angle(best_point_index, lidar_range_array.size)
        steering = best_point_angle / (np.pi / 2.0)

        target_distance = lidar_range_array[best_point_index]
        speed = compute_speed(target_distance) / 5

        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = self.get_clock().now().to_msg()
        ack_msg.header.frame_id = "base_link"
        ack_msg.drive.steering_angle = steering
        ack_msg.drive.speed = speed
        self.drive_pub.publish(ack_msg)


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
