"""Check speed_to_erpm_gain: drive 1 m by odometry, then measure the real distance.

Hold the deadman page while it runs. If the car went farther than 1 m, lower
speed_to_erpm_gain in race_bringup/config/vesc.yaml in proportion (and vice versa).
"""

import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class DriveOneMeter(Node):
    def __init__(self):
        super().__init__("drive_one_meter")
        self.pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.create_subscription(Odometry, "/odom", self.cb, 10)
        self.last_x = None
        self.last_y = None
        self.traveled = 0.0
        self.target = 1.0
        self.done = False

    def cb(self, msg):
        if self.done:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.last_x is not None:
            self.traveled += math.hypot(x - self.last_x, y - self.last_y)
        self.last_x, self.last_y = x, y
        cmd = AckermannDriveStamped()
        if self.traveled >= self.target:
            cmd.drive.speed = 0.0
            self.done = True
            self.get_logger().info(f"Done. Traveled {self.traveled:.3f} m")
        else:
            cmd.drive.speed = 1.0
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = DriveOneMeter()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
