#!/usr/bin/env python3

import socket

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node

teleop_port: int = 5006
packet_timeout: float = 0.3  # s, stop if no packet for this long
max_speed: float = 2.0  # m/s, hard clamp on incoming command
max_steer: float = 0.4  # rad, hard clamp on incoming command
max_accel: float = 2.0  # m/s², slew rate when increasing |speed|
max_decel: float = 4.0  # m/s², slew rate when decreasing |speed| or reversing
max_steer_rate: float = 3.0  # rad/s, slew rate on steering


class RcTeleop(Node):
    def __init__(self):
        super().__init__("rc_teleop")
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", teleop_port))
        self.sock.setblocking(False)
        self.steer_target = 0.0
        self.speed_target = 0.0
        self.steer_cmd = 0.0
        self.speed_cmd = 0.0
        self.last_pkt = 0.0
        self.last_pub = 0.0
        self.create_timer(0.02, self.poll)
        self.create_timer(0.05, self.publish_tick)

    def poll(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(64)
            except BlockingIOError:
                break
            try:
                self.sock.sendto(b"ok", addr)
            except OSError:
                pass
            try:
                s, v = data.decode().split(",")
                self.steer_target = max(-max_steer, min(max_steer, float(s)))
                self.speed_target = max(-max_speed, min(max_speed, float(v)))
                self.last_pkt = self.get_clock().now().nanoseconds * 1e-9
            except (ValueError, UnicodeDecodeError):
                pass

    def publish_tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = now - self.last_pub if self.last_pub > 0.0 else 0.05
        dt = max(0.0, min(0.1, dt))  # clamp to absorb timer jitter and first tick
        self.last_pub = now

        armed = (now - self.last_pkt) < packet_timeout
        target_steer = self.steer_target if armed else 0.0
        target_speed = self.speed_target if armed else 0.0

        steer_step = max_steer_rate * dt
        self.steer_cmd += max(
            -steer_step, min(steer_step, target_steer - self.steer_cmd)
        )

        # Asymmetric: gentle when speeding up (reduces wheel slip on launch),
        # firmer when slowing or reversing direction.
        speeding_up = (
            target_speed * self.speed_cmd >= 0.0
            and abs(target_speed) > abs(self.speed_cmd)
        )
        speed_step = (max_accel if speeding_up else max_decel) * dt
        self.speed_cmd += max(
            -speed_step, min(speed_step, target_speed - self.speed_cmd)
        )

        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.drive.steering_angle = self.steer_cmd
        m.drive.speed = self.speed_cmd
        self.drive_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = RcTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
