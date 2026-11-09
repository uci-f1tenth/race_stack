#!/usr/bin/env python3
"""Shut the car's computer down cleanly when the battery is nearly flat.

The VESC limits motor current at low voltage on its own, but the Jetson keeps
draining the pack. Waits for the car to stop first: with the Jetson gone the
VESC would hold the last command until its own timeout. Needs passwordless
shutdown for the f1t user, e.g. in /etc/sudoers.d/shutdown:
    f1t ALL=(root) NOPASSWD: /usr/sbin/shutdown
"""

import subprocess
import time

import rclpy
from rclpy.node import Node
from vesc_msgs.msg import VescStateStamped

cutoff_v: float = 12.0  # 3.0 V/cell on the 4S pack
hold_s: float = 5.0  # must stay below cutoff this long; acceleration sag is shorter
stopped_erpm: float = 150.0  # ~0.05 m/s at speed_to_erpm_gain 3000


class LowVoltageShutdown(Node):
    def __init__(self):
        super().__init__("low_voltage_shutdown")
        self.low_since = None
        self.waiting_to_stop = False
        self.fired = False
        self.create_subscription(VescStateStamped, "/sensors/core", self.cb, 10)
        self.get_logger().info(f"watching /sensors/core, cutoff={cutoff_v}V")

    def cb(self, msg: VescStateStamped):
        if self.fired:
            return
        v = float(msg.state.voltage_input)
        now = time.monotonic()  # immune to wall-clock steps
        if v >= cutoff_v:
            self.low_since = None
            self.waiting_to_stop = False
            return
        if self.low_since is None:
            self.low_since = now
            self.get_logger().warn(
                f"voltage {v:.2f}V < {cutoff_v}V, shutting down if it stays "
                f"low for {hold_s:.0f}s"
            )
            return
        if now - self.low_since < hold_s:
            return
        if abs(msg.state.speed) >= stopped_erpm:
            if not self.waiting_to_stop:
                self.get_logger().warn("battery flat: shutting down once the car stops")
                self.waiting_to_stop = True
            return
        self.fired = True
        self.get_logger().error(f"voltage {v:.2f}V — shutting down")
        # -n: fail instead of waiting on a password prompt nobody will see
        result = subprocess.run(
            ["sudo", "-n", "shutdown", "-h", "now"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.get_logger().error(
                "shutdown failed (needs passwordless sudo): " + result.stderr.strip()
            )


def main():
    rclpy.init()
    node = LowVoltageShutdown()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
