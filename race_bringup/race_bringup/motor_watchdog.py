#!/usr/bin/env python3
"""Stop the motor if drive commands stop reaching vesc_driver.

ackermann_to_vesc turns every mux output into a motor speed and a servo command,
and the deadman keeps that stream going at 50 Hz even while the car is held
still. If it stops (the deadman, mux or ackermann_to_vesc died), the VESC would
keep the last speed until its own timeout (1 s by default, then it coasts), so
this commands zero speed straight to vesc_driver until the stream comes back.
It watches the servo topic because it publishes on the motor one.

Losing vesc_driver itself, the USB link or the Jetson still falls back to the
VESC's own timeout.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

stale_s: float = 0.15  # the chain normally publishes at >= 50 Hz
rate_hz: float = 50.0


class MotorWatchdog(Node):
    def __init__(self):
        super().__init__("motor_watchdog")
        self.last_cmd_t = -math.inf
        self.seen = False
        self.stopping = False
        self.zero = Float64()
        self.create_subscription(Float64, "commands/servo/position", self.servo_cb, 10)
        self.speed_pub = self.create_publisher(Float64, "commands/motor/speed", 10)
        self.create_timer(1.0 / rate_hz, self.check)

    def servo_cb(self, msg: Float64):
        self.last_cmd_t = time.monotonic()
        self.seen = True

    def check(self):
        stale = time.monotonic() - self.last_cmd_t > stale_s
        if stale != self.stopping and self.seen:  # quiet until the chain is up
            if stale:
                self.get_logger().warn(
                    "drive commands stopped: commanding zero speed to vesc_driver"
                )
            else:
                self.get_logger().info("drive commands back")
        self.stopping = stale
        if stale:
            self.speed_pub.publish(self.zero)


def main(args=None):
    rclpy.init(args=args)
    node = MotorWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
