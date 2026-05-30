"""Shut the Nano down cleanly when the VESC battery sags below 12V.

Run on the Nano. Requires passwordless `sudo shutdown` (or run as root).
"""

import subprocess

import rclpy
from rclpy.node import Node
from vesc_msgs.msg import VescStateStamped

CUTOFF_V = 12.0
CONSECUTIVE_SAMPLES = 10  # ~0.5s at the VESC's ~20 Hz state stream — ignore brief sag


class LowVoltageShutdown(Node):
    def __init__(self):
        super().__init__("low_voltage_shutdown")
        self.low_count = 0
        self.fired = False
        self.create_subscription(VescStateStamped, "/sensors/core", self.cb, 10)
        self.get_logger().info(f"watching /sensors/core, cutoff={CUTOFF_V}V")

    def cb(self, msg: VescStateStamped):
        if self.fired:
            return
        v = float(msg.state.voltage_input)
        if v < CUTOFF_V:
            self.low_count += 1
            self.get_logger().warn(
                f"voltage {v:.2f}V < {CUTOFF_V}V ({self.low_count}/{CONSECUTIVE_SAMPLES})"
            )
            if self.low_count >= CONSECUTIVE_SAMPLES:
                self.fired = True
                self.get_logger().error(f"voltage {v:.2f}V — shutting down")
                subprocess.Popen(["sudo", "shutdown", "-h", "now"])
        else:
            self.low_count = 0


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
