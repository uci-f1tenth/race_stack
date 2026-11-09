#!/usr/bin/env python3
"""Car odometry: VESC wheel speed for translation, VESC gyro for yaw.

Reads the VESC driver directly (/sensors/core for motor ERPM, /sensors/imu/raw
for the gyro) and publishes /odom plus the odom->base_link TF. This replaces
vesc_to_odom: its yaw rate came from the *commanded* servo angle, so calibration
error, understeer and tire slip all landed in the heading, and its stock speed
sign is inverted relative to ackermann_to_vesc (the car only had it right via an
uncommitted patch to f1tenth_system).

The VESC driver publishes the gyro in deg/s with no frame_id. Its z axis points
up on this car (acc_z = +1 g at rest), so gyro z is yaw rate with the ROS sign.
If the IMU goes quiet the heading is held until it comes back.
"""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster
from vesc_msgs.msg import VescStateStamped

core_topic: str = "/sensors/core"
imu_topic: str = "/sensors/imu/raw"
imu_timeout: float = 0.1  # s, hold heading if the gyro is older than this
stationary_speed: float = 0.05  # m/s, below this the wheels count as stopped
bias_alpha: float = 0.01  # gyro bias EMA weight per sample (~2 s at 52 Hz)
# A still car's gyro reads within this of its bias (MEMS offsets are a few deg/s
# at most); anything larger is real rotation, e.g. lifted or sliding.
bias_gate: float = math.radians(5.0)
max_dt: float = 0.5  # s, don't integrate across a longer gap


def stamp_s(stamp) -> float:
    return Time.from_msg(stamp).nanoseconds * 1e-9


class ImuOdom(Node):
    def __init__(self):
        super().__init__("imu_odom")
        # Same mapping ackermann_to_vesc uses: erpm = gain * speed + offset.
        self.erpm_gain = self.declare_parameter("speed_to_erpm_gain", 0.0).value
        self.erpm_offset = self.declare_parameter("speed_to_erpm_offset", 0.0).value
        if self.erpm_gain == 0.0:
            raise RuntimeError(
                "speed_to_erpm_gain not set: launch with race_bringup/config/vesc.yaml"
            )

        # Heading is integrated at the gyro's own timestamps and interpolated to
        # each wheel-speed stamp; the two streams arrive in no fixed order, so
        # holding the latest gyro sample would lag the heading by ~10 ms.
        self.yaw = 0.0  # rad, at time yaw_t
        self.yaw_t = None
        self.yaw_rate = 0.0  # rad/s, bias-corrected rate at yaw_t
        self.gyro_t = None  # stamp of the last IMU message, for staleness
        self.bias = 0.0
        self.bias_samples = 0
        self.stationary = True
        self.x = self.y = 0.0
        self.last_t = None
        self.last_heading = 0.0
        self.using_imu = None

        self.create_subscription(Imu, imu_topic, self.imu_cb, 50)
        self.create_subscription(VescStateStamped, core_topic, self.core_cb, 50)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 50)
        self.tf_pub = TransformBroadcaster(self)
        self.odom = Odometry()
        self.odom.header.frame_id = "odom"
        self.odom.child_frame_id = "base_link"
        self.tf = TransformStamped()
        self.tf.header.frame_id = "odom"
        self.tf.child_frame_id = "base_link"

    def advance_yaw(self, t: float, rate: float, trapezoid: bool):
        if self.yaw_t is not None and t <= self.yaw_t:
            # Older than the last update (IMU resuming after a hold): don't step
            # the integration time backwards and count that span twice.
            self.yaw_rate = rate
            return
        if self.yaw_t is not None and t - self.yaw_t < max_dt:
            mean_rate = 0.5 * (rate + self.yaw_rate) if trapezoid else rate
            self.yaw += mean_rate * (t - self.yaw_t)
        self.yaw_t, self.yaw_rate = t, rate

    def imu_cb(self, msg: Imu):
        if self.gyro_t is None and msg.linear_acceleration.z < 0.8:
            self.get_logger().warn(
                f"VESC IMU z reads {msg.linear_acceleration.z:.2f} g at startup, "
                "not ~+1 g: if the VESC was remounted, gyro z is no longer yaw"
            )
        deviation = math.radians(msg.angular_velocity.z) - self.bias
        if self.stationary and abs(deviation) < bias_gate:
            # Wheels stopped and gyro quiet: hold heading and learn the bias
            # (running mean at first, then a slow EMA).
            self.bias_samples += 1
            self.bias += max(bias_alpha, 1.0 / self.bias_samples) * deviation
            rate = 0.0
        else:
            # Moving, or turning with the wheels stopped (lifted, sliding).
            rate = deviation
        self.gyro_t = stamp_s(msg.header.stamp)
        self.advance_yaw(self.gyro_t, rate, trapezoid=True)

    def core_cb(self, msg: VescStateStamped):
        t = stamp_s(msg.header.stamp)
        v = (msg.state.speed - self.erpm_offset) / self.erpm_gain
        if abs(v) < stationary_speed:
            v = 0.0
        self.stationary = v == 0.0
        use_imu = self.gyro_t is not None and t - self.gyro_t < imu_timeout
        # The very first VESC message can beat the first IMU one; don't call
        # that a dropout.
        if use_imu != self.using_imu and (use_imu or self.last_t is not None):
            if use_imu:
                self.get_logger().info("yaw from VESC gyro")
            else:
                self.get_logger().warn("no fresh IMU: holding heading")
            self.using_imu = use_imu
        if not use_imu:
            self.advance_yaw(t, 0.0, trapezoid=False)
        # Gyro heading carried to this stamp (either side of the last sample).
        heading = self.yaw + self.yaw_rate * (t - self.yaw_t)

        if self.last_t is not None and 0.0 < t - self.last_t < max_dt:
            dt = t - self.last_t
            mid = 0.5 * (self.last_heading + heading)
            self.x += v * math.cos(mid) * dt
            self.y += v * math.sin(mid) * dt
        self.last_t, self.last_heading = t, heading

        m = self.odom
        m.header.stamp = msg.header.stamp
        m.pose.pose.position.x = self.x
        m.pose.pose.position.y = self.y
        q = m.pose.pose.orientation
        q.z, q.w = math.sin(0.5 * heading), math.cos(0.5 * heading)
        m.twist.twist.linear.x = v
        m.twist.twist.angular.z = self.yaw_rate
        self.odom_pub.publish(m)

        tf = self.tf
        tf.header.stamp = msg.header.stamp
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.rotation = q
        self.tf_pub.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = ImuOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
