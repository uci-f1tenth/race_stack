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

# Steering smoothing
error_ema_alpha: float = 0.3  # 0..1, lower = smoother but laggier
steering_slew_per_frame: float = 0.15  # max change in steering output per scan


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


class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limit: float = 1.0,
        integral_limit: float = 0.5,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time: float | None = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, error: float, now: float) -> float:
        if self.prev_time is None:
            self.prev_time = now
            self.prev_error = error
            return float(
                np.clip(self.kp * error, -self.output_limit, self.output_limit)
            )

        dt = now - self.prev_time
        if dt <= 0.0:
            return float(
                np.clip(self.kp * error, -self.output_limit, self.output_limit)
            )

        self.integral += error * dt
        self.integral = float(
            np.clip(self.integral, -self.integral_limit, self.integral_limit)
        )
        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = float(np.clip(output, -self.output_limit, self.output_limit))

        self.prev_error = error
        self.prev_time = now
        return output


class DisparityExtender(Node):
    def __init__(self):
        super().__init__("disparity_extender")
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        # Conservative PID for noisy lidar input from loose suspension.
        # No D term — D would amplify per-frame tilt noise.
        self.steering_pid = PIDController(kp=0.5, ki=0.0, kd=0.0)

        # EMA-filtered error feeds the PID; None until first scan.
        self.filtered_error: float | None = None
        # Last published steering, for slew limiting.
        self.last_steering: float = 0.0

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
            self.steering_pid.reset()
            self.filtered_error = None
            self.last_steering = 0.0
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

        # Raw error normalized to [-1, 1].
        raw_error = best_point_angle / (np.pi / 2.0)

        # 1) Low-pass filter the error to reject lidar tilt noise.
        if self.filtered_error is None:
            self.filtered_error = raw_error
        else:
            self.filtered_error = (
                error_ema_alpha * raw_error
                + (1.0 - error_ema_alpha) * self.filtered_error
            )

        # 2) PID on filtered error.
        now = self.get_clock().now().nanoseconds * 1e-9
        pid_output = self.steering_pid.update(self.filtered_error, now)

        # 3) Slew rate limit on the output.
        delta = pid_output - self.last_steering
        delta = float(np.clip(delta, -steering_slew_per_frame, steering_slew_per_frame))
        steering = self.last_steering + delta
        self.last_steering = steering

        target_distance = lidar_range_array[best_point_index]
        speed = compute_speed(target_distance) * 3  # max speed 3 m/s
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
