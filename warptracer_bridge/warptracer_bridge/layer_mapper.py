"""Build one Cartographer map per track layer from a single scan stream.

Run with launch/layered_mapping.launch.py, which starts one cartographer_node and
one occupancy grid node per layer in namespace layer_N, each with its own map
frame layer_N_map.

The node forwards /scan to the layer the car is on (from /layer) and /odom to
every layer. Pausing a layer's scans while the car is away does not work:
Cartographer treats odometry across the gap as one long motion and the map
smears. Instead the layer's trajectory is finished on leaving, and on returning a
new trajectory starts where the layer's last tracked pose plus the odometry
since then puts the car.

Each layer's map frame is where the car stood at that layer's first scan; the
node publishes it relative to odom so Foxglove can show the maps with the replay.
"""
import math
import os
import signal
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from cartographer_ros_msgs.srv import FinishTrajectory, StartTrajectory
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32
from tf2_ros import StaticTransformBroadcaster

# Without scans for this long in wall time (sim time stops with the replay), the run
# is over: finish the active trajectory.
IDLE_SECONDS = 3.0


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def compose(a, b):
    """Planar pose b, given in a's frame, expressed in a's parent frame."""
    x, y, yaw = a
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * b[0] - s * b[1], y + s * b[0] + c * b[1], yaw + b[2]


def relative(a, b):
    """Planar pose b in the frame of pose a; both share a parent frame."""
    x, y, yaw = a
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = b[0] - x, b[1] - y
    return c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(b[2] - yaw), math.cos(b[2] - yaw))


class LayerMapper(Node):
    def __init__(self):
        super().__init__("layer_mapper")
        self.declare_parameter("layers", 2)
        self.declare_parameter("configuration_directory",
                               os.path.join(get_package_share_directory("roboracer_slam"), "config"))
        self.declare_parameter("configuration_prefix", "roboracer_cartographer_layer_")
        self.names = [f"layer_{n}" for n in range(int(self.get_parameter("layers").value))]
        self.directory = self.get_parameter("configuration_directory").value
        self.prefix = self.get_parameter("configuration_prefix").value

        self.scan_out = [self.create_publisher(LaserScan, f"/{n}/scan", 100) for n in self.names]
        self.odom_out = [self.create_publisher(Odometry, f"/{n}/odom", 100) for n in self.names]
        self.finish_clients = [self.create_client(FinishTrajectory, f"/{n}/finish_trajectory") for n in self.names]
        self.start_clients = [self.create_client(StartTrajectory, f"/{n}/start_trajectory") for n in self.names]
        self.static_broadcaster = StaticTransformBroadcaster(self)
        for layer, name in enumerate(self.names):
            self.create_subscription(PoseStamped, f"/{name}/tracked_pose",
                                     lambda message, layer=layer: self.on_tracked_pose(layer, message), 100)

        # Each layer's cartographer_node starts trajectory 0 itself and waits for scans.
        self.trajectory = [0] * len(self.names)
        self.started = [False] * len(self.names)
        self.map_pose = [None] * len(self.names)  # Last tracked pose in the layer's map frame.
        self.left_odom = [None] * len(self.names)  # Odometry pose when the car left the layer.
        self.map_origins = []
        self.odom_pose = None
        self.layer = 0
        self.active = None
        self.starting = False
        self.finished = False
        self.last_scan = None

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Int32, "/layer", self.on_layer, latched)
        self.create_subscription(Odometry, "/odom", self.on_odom, 100)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 100)
        self.create_timer(0.5, self.check_idle, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f"Mapping {len(self.names)} layers with {self.directory}/{self.prefix}N.lua")

    def on_layer(self, message):
        if not 0 <= message.data < len(self.names):
            self.get_logger().error(f"Layer {message.data} is out of range; ignoring it")
            return
        self.layer = message.data

    def on_odom(self, message):
        p = message.pose.pose.position
        self.odom_pose = (p.x, p.y, yaw_of(message.pose.pose.orientation))
        for publisher in self.odom_out:
            publisher.publish(message)

    def on_tracked_pose(self, layer, message):
        # While a layer is inactive or restarting, its pose is only an extrapolated guess.
        if layer == self.active and not self.starting:
            p = message.pose.position
            self.map_pose[layer] = (p.x, p.y, yaw_of(message.pose.orientation))

    def on_scan(self, message):
        if self.odom_pose is None or self.finished:
            return
        self.last_scan = time.monotonic()
        if self.layer != self.active:
            self.switch_to(self.layer, message.header.stamp)
        if not self.starting:
            self.scan_out[self.active].publish(message)

    def switch_to(self, layer, stamp):
        old, self.active = self.active, layer
        if old is not None:
            self.left_odom[old] = self.odom_pose
            self.finish(old)
        if not self.started[layer]:
            self.started[layer] = True
            self.publish_map_origin(layer, stamp)
            self.get_logger().info(f"{self.names[layer]}: first visit, trajectory 0")
            return
        if self.map_pose[layer] is None:
            self.get_logger().warn(f"{self.names[layer]}: no tracked pose from the last visit; "
                                   "starting without an initial pose")
            self.start(layer, None)
            return
        # Where odometry says the car moved since leaving, applied to the layer's last map pose.
        moved = relative(self.left_odom[layer], self.odom_pose)
        self.start(layer, compose(self.map_pose[layer], moved))

    def finish(self, layer):
        request = FinishTrajectory.Request(trajectory_id=self.trajectory[layer])
        self.finish_clients[layer].call_async(request)
        self.get_logger().info(f"{self.names[layer]}: finishing trajectory {self.trajectory[layer]}")

    def start(self, layer, pose):
        request = StartTrajectory.Request()
        request.configuration_directory = self.directory
        request.configuration_basename = f"{self.prefix}{layer}.lua"
        if pose is not None:
            # Cartographer places the initial pose relative to the referenced trajectory's
            # first node, not its last. Trajectory 0's first node is the map origin.
            request.use_initial_pose = True
            request.relative_to_trajectory_id = 0
            x, y, yaw = pose
            request.initial_pose.position.x, request.initial_pose.position.y = x, y
            request.initial_pose.orientation.z = math.sin(yaw / 2.0)
            request.initial_pose.orientation.w = math.cos(yaw / 2.0)
        self.starting = True
        future = self.start_clients[layer].call_async(request)
        future.add_done_callback(lambda done, layer=layer: self.on_started(layer, done))

    def on_started(self, layer, future):
        response = future.result()
        self.starting = False
        if response.status.code != 0:
            self.get_logger().error(f"{self.names[layer]}: start_trajectory failed: {response.status.message}")
            return
        self.trajectory[layer] = response.trajectory_id
        self.get_logger().info(f"{self.names[layer]}: returned, trajectory {response.trajectory_id}")

    def publish_map_origin(self, layer, stamp):
        """The layer's map frame is base_link at its first scan, projected to the ground plane."""
        x, y, yaw = self.odom_pose
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = f"{self.names[layer]}_map"
        transform.transform.translation.x, transform.transform.translation.y = x, y
        transform.transform.rotation.z = math.sin(yaw / 2.0)
        transform.transform.rotation.w = math.cos(yaw / 2.0)
        self.map_origins.append(transform)
        self.static_broadcaster.sendTransform(self.map_origins)

    def check_idle(self):
        if self.finished or self.last_scan is None or self.starting:
            return
        if time.monotonic() - self.last_scan > IDLE_SECONDS:
            self.finished = True
            self.finish(self.active)
            self.get_logger().info("No scans for a while; finished the active trajectory. "
                                   "Press Ctrl+C and relaunch to map another replay.")


def main():
    rclpy.init()
    node = LayerMapper()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Ctrl+C reaches this process twice under ros2 launch (terminal and launch), and the
        # first can shut the context down; ignore the second, and let the node end with the process.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if rclpy.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
