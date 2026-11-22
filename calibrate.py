"""Check speed_to_erpm_gain against a tape measure.

Drives 1 m by odometry (hold the deadman page), stops, waits for the car to come
to rest, then prints the odometry distance for the whole move, coasting
included. Measure how far the car really went and set speed_to_erpm_gain in
race_bringup/config/vesc.yaml to  gain * odometry distance / measured distance.
"""

import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

target: float = 1.0  # m by odometry before commanding a stop
rest_msgs: int = 25  # consecutive zero-speed odom messages (~0.5 s) = at rest


class DriveOneMeter(Node):
    def __init__(self):
        super().__init__("drive_one_meter")
        self.pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.create_subscription(Odometry, "/odom", self.cb, 10)
        self.last_x = None
        self.last_y = None
        self.traveled = 0.0
        self.stopping = False
        self.still = 0
        self.done = False

    def cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.last_x is not None:
            self.traveled += math.hypot(x - self.last_x, y - self.last_y)
        self.last_x, self.last_y = x, y
        if self.traveled >= target and not self.stopping:
            self.stopping = True
            self.get_logger().info("1 m reached, stopping")
        cmd = AckermannDriveStamped()
        cmd.drive.speed = 0.0 if self.stopping else 1.0
        self.pub.publish(cmd)
        # The car keeps rolling after the stop command, so keep counting until it
        # is at rest; otherwise the tape measures more than odometry reports.
        self.still = self.still + 1 if msg.twist.twist.linear.x == 0.0 else 0
        if self.stopping and self.still >= rest_msgs:
            self.done = True
            self.get_logger().info(
                f"At rest. Odometry distance {self.traveled:.3f} m; measure the real "
                "distance and set speed_to_erpm_gain = gain * odometry / measured"
            )


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
