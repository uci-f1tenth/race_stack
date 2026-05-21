"""Bridge between F1Tenth /drive (AckermannDriveStamped) and AutoDRIVE Roboracer.

AutoDRIVE expects two separate normalized Float32 commands:
  /autodrive/roboracer_1/steering_command   in [-1, 1] (left negative, right positive)
  /autodrive/roboracer_1/throttle_command   in [-1, 1] (forward positive)

Our driver nodes already publish a normalized steering value in
AckermannDriveStamped.drive.steering_angle (see disparity_extender / pure_pursuit),
so steering passes through. Speed (m/s) is mapped linearly to a normalized
throttle via `max_speed`; tune to match the AutoDRIVE vehicle's top speed.
"""

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from std_msgs.msg import Float32


class AutodriveBridge(Node):
    def __init__(self):
        super().__init__("autodrive_bridge")
        self.declare_parameter("vehicle", "roboracer_1")
        self.declare_parameter("max_speed", 3.0)
        self.declare_parameter("drive_topic", "/drive")

        vehicle = self.get_parameter("vehicle").value
        self.max_speed = float(self.get_parameter("max_speed").value)
        drive_topic = self.get_parameter("drive_topic").value

        base = f"/autodrive/{vehicle}"
        self.steer_pub = self.create_publisher(Float32, f"{base}/steering_command", 10)
        self.throttle_pub = self.create_publisher(Float32, f"{base}/throttle_command", 10)
        self.create_subscription(AckermannDriveStamped, drive_topic, self.cb, 10)
        self.get_logger().info(
            f"Bridging {drive_topic} -> {base}/{{steering,throttle}}_command "
            f"(max_speed={self.max_speed} m/s)"
        )

    def cb(self, msg: AckermannDriveStamped):
        steer = max(-1.0, min(1.0, float(msg.drive.steering_angle)))
        throttle = max(-1.0, min(1.0, float(msg.drive.speed) / self.max_speed))
        self.steer_pub.publish(Float32(data=steer))
        self.throttle_pub.publish(Float32(data=throttle))


def main(args=None):
    rclpy.init(args=args)
    node = AutodriveBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
