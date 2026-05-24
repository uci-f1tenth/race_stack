#!/usr/bin/env python3
"""F1TENTH <-> AutoDRIVE bridge.

Lets a stock F1TENTH ROS 2 stack drive the RoboRacer digital twin in the
AutoDRIVE Simulator without modification.

F1TENTH side (what user code expects)             AutoDRIVE side (devkit)
----------------------------------------          --------------------------------------------
sub  /drive  ackermann_msgs/AckermannDriveStamped  pub  /autodrive/roboracer_1/throttle_command  std_msgs/Float32
                                                   pub  /autodrive/roboracer_1/steering_command  std_msgs/Float32
pub  /scan   sensor_msgs/LaserScan                 sub  /autodrive/roboracer_1/lidar
pub  /odom   nav_msgs/Odometry                     sub  /autodrive/roboracer_1/ips   geometry_msgs/Point
                                                   sub  /autodrive/roboracer_1/imu   sensor_msgs/Imu
pub  /imu/data sensor_msgs/Imu                     sub  /autodrive/roboracer_1/imu
pub  /ego_racecar/odom nav_msgs/Odometry           (mirror of /odom for sim-style nodes)
pub  /pf/pose/odom nav_msgs/Odometry               (mirror, for nodes that expect particle filter)

Throttle is derived from the commanded speed using a simple proportional
map saturated at the RoboRacer top speed (22.88 m/s). Steering is
normalized by the physical steering-angle lock (0.5236 rad) and clipped
to [-1, 1]. The sign convention matches AutoDRIVE's steering actuator;
flip STEER_SIGN if your stack expects the opposite polarity.
"""

import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster

MAX_SPEED = 22.88
MAX_STEER = 0.5236
THROTTLE_KP = 1.0 / 4.0
STEER_SIGN = 1.0
ODOM_FRAME = "map"
BASE_FRAME = "ego_racecar/base_link"
LASER_FRAME = "ego_racecar/laser"


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class F1TenthAutoDriveBridge(Node):
    def __init__(self):
        super().__init__("f1tenth_autodrive_bridge")

        self.throttle_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/throttle_command", 10
        )
        self.steer_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/steering_command", 10
        )

        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.imu_pub = self.create_publisher(Imu, "/imu/data", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.odom_sim_pub = self.create_publisher(Odometry, "/ego_racecar/odom", 10)
        self.odom_pf_pub = self.create_publisher(Odometry, "/pf/pose/odom", 10)

        self.create_subscription(AckermannDriveStamped, "/drive", self.drive_cb, 10)
        self.create_subscription(
            LaserScan, "/autodrive/roboracer_1/lidar", self.lidar_cb, 10
        )
        self.create_subscription(Imu, "/autodrive/roboracer_1/imu", self.imu_cb, 10)
        self.create_subscription(Point, "/autodrive/roboracer_1/ips", self.ips_cb, 10)
        self.create_subscription(
            Float32, "/autodrive/roboracer_1/throttle", self.throttle_fb_cb, 10
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.last_position = None
        self.last_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.last_angular_velocity = (0.0, 0.0, 0.0)
        self.last_speed = 0.0

        self.create_timer(0.02, self.publish_odom)

        self.get_logger().info("F1TENTH <-> AutoDRIVE bridge ready.")

    def drive_cb(self, msg: AckermannDriveStamped):
        speed = float(msg.drive.speed)
        steering = float(msg.drive.steering_angle)

        throttle = float(np.clip(THROTTLE_KP * speed, -1.0, 1.0))
        if abs(speed) >= MAX_SPEED:
            throttle = math.copysign(1.0, speed)

        steer_norm = float(np.clip(STEER_SIGN * steering / MAX_STEER, -1.0, 1.0))

        t = Float32()
        t.data = throttle
        s = Float32()
        s.data = steer_norm
        self.throttle_pub.publish(t)
        self.steer_pub.publish(s)

    def lidar_cb(self, msg: LaserScan):
        out = LaserScan()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = LASER_FRAME
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = msg.ranges
        out.intensities = msg.intensities
        self.scan_pub.publish(out)

    def imu_cb(self, msg: Imu):
        out = Imu()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = BASE_FRAME
        out.orientation = msg.orientation
        out.angular_velocity = msg.angular_velocity
        out.linear_acceleration = msg.linear_acceleration
        out.orientation_covariance = msg.orientation_covariance
        out.angular_velocity_covariance = msg.angular_velocity_covariance
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance
        self.imu_pub.publish(out)

        self.last_orientation = msg.orientation
        self.last_angular_velocity = (
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        )

    def ips_cb(self, msg: Point):
        self.last_position = (msg.x, msg.y, msg.z)

    def throttle_fb_cb(self, msg: Float32):
        self.last_speed = float(msg.data) * MAX_SPEED

    def publish_odom(self):
        if self.last_position is None:
            return

        now = self.get_clock().now().to_msg()
        x, y, z = self.last_position
        q = self.last_orientation
        wx, wy, wz = self.last_angular_velocity
        yaw = yaw_from_quat(q)
        vx = self.last_speed * math.cos(yaw)
        vy = self.last_speed * math.sin(yaw)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = ODOM_FRAME
        odom.child_frame_id = BASE_FRAME
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x = self.last_speed
        odom.twist.twist.angular.x = wx
        odom.twist.twist.angular.y = wy
        odom.twist.twist.angular.z = wz

        self.odom_pub.publish(odom)
        self.odom_sim_pub.publish(odom)

        odom_map = Odometry()
        odom_map.header.stamp = now
        odom_map.header.frame_id = ODOM_FRAME
        odom_map.child_frame_id = BASE_FRAME
        odom_map.pose.pose.position.x = x
        odom_map.pose.pose.position.y = y
        odom_map.pose.pose.position.z = z
        odom_map.pose.pose.orientation = q
        odom_map.twist.twist.linear.x = vx
        odom_map.twist.twist.linear.y = vy
        odom_map.twist.twist.angular.z = wz
        self.odom_pf_pub.publish(odom_map)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = ODOM_FRAME
        tf.child_frame_id = BASE_FRAME
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
        tf.transform.rotation = q
        self.tf_broadcaster.sendTransform(tf)

        tf_laser = TransformStamped()
        tf_laser.header.stamp = now
        tf_laser.header.frame_id = BASE_FRAME
        tf_laser.child_frame_id = LASER_FRAME
        tf_laser.transform.translation.x = 0.2733
        tf_laser.transform.translation.y = 0.0
        tf_laser.transform.translation.z = 0.096
        tf_laser.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(tf_laser)


def main(args=None):
    rclpy.init(args=args)
    node = F1TenthAutoDriveBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
