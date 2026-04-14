#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def yaw_to_quaternion(yaw: float):
    # roll = pitch = 0
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return (0.0, 0.0, qz, qw)


class EncoderOdomPublisher(Node):
    def __init__(self):
        super().__init__('encoder_odom_publisher')

        # --- Parameters (defaults from guide) ---
        self.declare_parameter('wheel_radius', 0.059)      # m
        self.declare_parameter('track_width', 0.236)       # m
        self.declare_parameter('ticks_per_rev', 16 * 120)  # encoder PPR * conversion ratio
        self.declare_parameter('left_topic', '/autodrive/roboracer_1/left_encoder')
        self.declare_parameter('right_topic', '/autodrive/roboracer_1/right_encoder')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        self.r = float(self.get_parameter('wheel_radius').value)
        self.L = float(self.get_parameter('track_width').value)
        self.ticks_per_rev = float(self.get_parameter('ticks_per_rev').value)
        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.m_per_tick = (2.0 * math.pi * self.r) / self.ticks_per_rev

        # Pose state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Last tick values
        self.last_left_ticks = None
        self.last_right_ticks = None

        # Latest incoming ticks
        self.curr_left_ticks = None
        self.curr_right_ticks = None

        self.last_time = self.get_clock().now()

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)

        self.create_subscription(JointState, left_topic, self.left_cb, 10)
        self.create_subscription(JointState, right_topic, self.right_cb, 10)

        # Timer for integration/publication
        self.timer = self.create_timer(0.02, self.update)  # 50 Hz

        self.get_logger().info('Simple encoder odom publisher started.')

    def _extract_ticks(self, msg: JointState):
        # Prefer position[0] (common in JointState), fallback to velocity[0]/effort[0]
        if len(msg.position) > 0:
            return float(msg.position[0])
        if len(msg.velocity) > 0:
            return float(msg.velocity[0])
        if len(msg.effort) > 0:
            return float(msg.effort[0])
        return None

    def left_cb(self, msg: JointState):
        self.curr_left_ticks = self._extract_ticks(msg)

    def right_cb(self, msg: JointState):
        self.curr_right_ticks = self._extract_ticks(msg)

    def update(self):
        if self.curr_left_ticks is None or self.curr_right_ticks is None:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return

        if self.last_left_ticks is None:
            self.last_left_ticks = self.curr_left_ticks
            self.last_right_ticks = self.curr_right_ticks
            self.last_time = now
            return

        # Tick deltas
        dleft_ticks = self.curr_left_ticks - self.last_left_ticks
        dright_ticks = self.curr_right_ticks - self.last_right_ticks

        self.last_left_ticks = self.curr_left_ticks
        self.last_right_ticks = self.curr_right_ticks
        self.last_time = now

        # Wheel travel
        dl = dleft_ticks * self.m_per_tick
        dr = dright_ticks * self.m_per_tick

        # Differential drive kinematics
        ds = 0.5 * (dr + dl)
        dtheta = (dr - dl) / self.L

        # Midpoint integration
        self.x += ds * math.cos(self.yaw + 0.5 * dtheta)
        self.y += ds * math.sin(self.yaw + 0.5 * dtheta)
        self.yaw += dtheta

        vx = ds / dt
        vth = dtheta / dt

        qx, qy, qz, qw = yaw_to_quaternion(self.yaw)

        # Publish TF: odom -> base_link
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

        # Publish nav_msgs/Odometry
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = vx
        odom.twist.twist.angular.z = vth

        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = EncoderOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()