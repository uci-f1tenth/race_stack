import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class DriveOneMeter(Node):
    def __init__(self):
        super().__init__("drive_one_meter")
        self.pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.sub = self.create_subscription(Odometry, "/odom", self.cb, 10)
        self.start_x = None
        self.target = 1.0
        self.done = False

    def cb(self, msg):
        if self.done:
            return
        x = msg.pose.pose.position.x
        if self.start_x is None:
            self.start_x = x
        traveled = x - self.start_x
        cmd = AckermannDriveStamped()
        if traveled >= self.target:
            cmd.drive.speed = 0.0
            self.done = True
            self.get_logger().info(f"Done. Traveled {traveled:.3f} m")
        else:
            cmd.drive.speed = 1.0
        self.pub.publish(cmd)


def main():
    rclpy.init()
    rclpy.spin(DriveOneMeter())


if __name__ == "__main__":
    main()
