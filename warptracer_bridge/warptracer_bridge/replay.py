"""Publish a WarpTracer .npz recording as ROS 2 topics.

Run: ros2 run warptracer_bridge replay --ros-args -p recording:=/data/bridge.npz

Set the maps parameter to a file from tools/export_track_maps.py to also publish
the track's road grid for each layer on /ground_truth_map/layer_N, and the layer
the car is on, swapped as it changes level, on /ground_truth_map/current.
"""
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from warptracer_bridge.layer_switching import layer_schedule, pitch_degrees

# ROS treats time zero as "latest available", so the first scan cannot use it.
STAMP_OFFSET = 1.0
MOUNT_TOLERANCE = 1e-3  # meters
FREE, OCCUPIED = 0, 100  # OccupancyGrid cell values


def quat_conjugate(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def quat_multiply(a, b):
    """Hamilton product of XYZW quaternions."""
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    return np.concatenate([aw * bv + bw * av + np.cross(av, bv),
                           aw * bw - np.sum(av * bv, axis=-1, keepdims=True)], axis=-1)


def quat_rotate(q, v):
    t = 2.0 * np.cross(q[..., :3], v)
    return v + q[..., 3:] * t + np.cross(q[..., :3], t)


class Recording:
    """Scans with the car's pose and body-frame velocity at each scan time."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        self.metadata = json.loads(str(arrays["metadata_json"]))
        self.times = arrays["lidar_times"].astype(np.float64)
        self.ranges = arrays["lidar_ranges"].astype(np.float64)
        self.valid = arrays["lidar_valid"].astype(bool)
        self._load_angles(arrays["lidar_directions"].astype(np.float64))
        self._load_motion(arrays)

    def _load_angles(self, directions):
        if np.abs(directions[:, 2]).max() > 1e-6:
            raise ValueError("LaserScan needs a planar LiDAR; rays leave the sensor XY plane")
        angles = np.arctan2(directions[:, 1], directions[:, 0])
        steps = np.diff(angles)
        if len(steps) == 0 or np.ptp(steps) > 1e-5:
            raise ValueError("LaserScan needs evenly spaced beams")
        self.angle_min, self.angle_max = float(angles[0]), float(angles[-1])
        self.angle_increment = float(steps.mean())

    def _load_motion(self, arrays):
        pose_times = arrays["times"].astype(np.float64)
        poses = arrays["poses"].astype(np.float64)
        sensors = arrays["lidar_poses"].astype(np.float64)
        if pose_times[0] != self.times[0]:
            raise ValueError("Car poses and scans must start at the same time")

        # The first car pose and first scan share a time, which gives the mount.
        inverse = quat_conjugate(poses[0, 3:])
        self.mount_position = quat_rotate(inverse, sensors[0, :3] - poses[0, :3])
        self.mount_rotation = quat_multiply(inverse, sensors[0, 3:])

        self.base_rotations = quat_multiply(sensors[:, 3:], quat_conjugate(self.mount_rotation))
        self.base_positions = sensors[:, :3] - quat_rotate(self.base_rotations, self.mount_position)
        self._check_mount(pose_times, poses)

        # Recorded velocities are world-frame [linear, angular]; odometry twist is body-frame.
        world = np.stack([np.interp(self.times, pose_times, arrays["velocities"][:, axis])
                          for axis in range(6)], axis=1)
        inverse = quat_conjugate(self.base_rotations)
        self.linear = quat_rotate(inverse, world[:, :3])
        self.angular = quat_rotate(inverse, world[:, 3:])

    def _check_mount(self, pose_times, poses):
        indices = np.searchsorted(self.times, pose_times)
        indices = np.minimum(indices, len(self.times) - 1)
        shared = np.abs(self.times[indices] - pose_times) < 1e-9
        error = np.linalg.norm(self.base_positions[indices[shared]] - poses[shared, :3], axis=1).max()
        if error > MOUNT_TOLERANCE:
            raise ValueError(f"Car pose from scans is {error:.4f} m from the recorded pose")


class ReplayNode(Node):
    def __init__(self):
        super().__init__("warptracer_replay")
        self.declare_parameter("recording", "")
        self.declare_parameter("rate", 1.0)
        self.declare_parameter("maps", "")
        path = self.get_parameter("recording").value
        self.rate = float(self.get_parameter("rate").value)
        if not path:
            raise ValueError("Set the recording parameter to a WarpTracer .npz file")
        if self.rate <= 0.0:
            raise ValueError("rate must be positive")
        self.recording = Recording(path)

        self.clock_publisher = self.create_publisher(Clock, "/clock", 10)
        self.scan_publisher = self.create_publisher(LaserScan, "/scan", 10)
        self.odom_publisher = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.map_publishers = []

    def _publish_maps(self, path, stamp):
        """Publish each layer's road grid once; late subscribers still receive it.

        Returns the messages so the current layer can be swapped during the replay.
        """
        with np.load(path, allow_pickle=False) as data:
            free = data["free"].astype(bool)
            resolution = float(data["resolution"])
            origin = data["origin"].astype(np.float64)
            layer_heights = data["layer_heights"].astype(np.float64)
        if len(free) != 2:
            raise ValueError("Layer switching supports tracks with exactly two layers")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.current_map_publisher = self.create_publisher(OccupancyGrid, "/ground_truth_map/current", latched)
        messages = []
        for layer, road in enumerate(free):
            message = OccupancyGrid()
            message.header.stamp = stamp
            message.header.frame_id = "odom"
            message.info.map_load_time = stamp
            message.info.resolution = resolution
            message.info.width, message.info.height = road.shape
            message.info.origin.position.x = float(origin[0])
            message.info.origin.position.y = float(origin[1])
            # Flat sheets, lifted so the layers can be told apart.
            message.info.origin.position.z = float(layer_heights[layer])
            message.info.origin.orientation.w = 1.0
            # WarpTracer grids index [x, y]; OccupancyGrid is row-major from low y.
            cells = np.where(road.T, FREE, OCCUPIED).astype(np.int8).ravel()
            message.data = cells.tolist()
            publisher = self.create_publisher(OccupancyGrid, f"/ground_truth_map/layer_{layer}", latched)
            publisher.publish(message)
            self.map_publishers.append(publisher)
            messages.append(message)
        self.get_logger().info(f"Published {len(free)} ground truth map layers")
        return messages

    def _publish_current_map(self, message, stamp):
        """Show one layer below the car, so the sheet cannot hide it on a ramp."""
        message.header.stamp = stamp
        current = OccupancyGrid(header=message.header, info=message.info, data=message.data)
        current.info.origin.position.z = 0.0
        self.current_map_publisher.publish(current)

    def _transform(self, stamp, parent, child, position, rotation):
        message = TransformStamped()
        message.header.stamp = stamp
        message.header.frame_id = parent
        message.child_frame_id = child
        t, r = message.transform.translation, message.transform.rotation
        t.x, t.y, t.z = (float(value) for value in position)
        r.x, r.y, r.z, r.w = (float(value) for value in rotation)
        return message

    def _scan(self, stamp, index, period):
        recording = self.recording
        lidar = recording.metadata["lidar"]
        message = LaserScan()
        message.header.stamp = stamp
        message.header.frame_id = "laser"
        message.angle_min = recording.angle_min
        message.angle_max = recording.angle_max
        message.angle_increment = recording.angle_increment
        message.time_increment = 0.0  # WarpTracer scans are instantaneous.
        message.scan_time = period
        message.range_min = float(lidar["near"])
        message.range_max = float(lidar["far"])
        message.ranges = np.where(recording.valid[index], recording.ranges[index], np.inf).astype(float).tolist()
        return message

    def _odometry(self, stamp, index):
        recording = self.recording
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = "odom"
        message.child_frame_id = "base_link"
        p, q = message.pose.pose.position, message.pose.pose.orientation
        p.x, p.y, p.z = (float(value) for value in recording.base_positions[index])
        q.x, q.y, q.z, q.w = (float(value) for value in recording.base_rotations[index])
        v, w = message.twist.twist.linear, message.twist.twist.angular
        v.x, v.y, v.z = (float(value) for value in recording.linear[index])
        w.x, w.y, w.z = (float(value) for value in recording.angular[index])
        return message

    def run(self):
        recording = self.recording
        times = recording.times
        period = float(np.median(np.diff(times)))
        first = Time(seconds=float(times[0]) + STAMP_OFFSET).to_msg()
        scans_per_second = max(1, round(1.0 / period))
        maps = self.get_parameter("maps").value
        map_messages = self._publish_maps(maps, first) if maps else None
        layers = layer_schedule(times, pitch_degrees(recording.base_rotations))
        self.get_logger().info(
            f"Replaying {len(times)} scans over {times[-1] - times[0]:.1f} s at {self.rate:g}x; "
            f"mount {np.round(recording.mount_position, 4).tolist()}")

        start = time.monotonic()
        for index, scan_time in enumerate(times):
            if not rclpy.ok():
                return
            delay = start + (scan_time - times[0]) / self.rate - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            stamp = Time(seconds=float(scan_time) + STAMP_OFFSET).to_msg()
            clock = Clock()
            clock.clock = stamp
            self.clock_publisher.publish(clock)
            # A new replay moves the clock backwards, which can clear a viewer's
            # transforms, so this follows the clock and repeats instead of latching once.
            if index % scans_per_second == 0:
                self.static_broadcaster.sendTransform(self._transform(
                    stamp, "base_link", "laser", recording.mount_position, recording.mount_rotation))
            if map_messages and (index == 0 or layers[index] != layers[index - 1]):
                self._publish_current_map(map_messages[layers[index]], stamp)
                self.get_logger().info(f"t={scan_time:.2f} s: current map is layer {layers[index]}")
            # Transform first, so a scan's pose is available when the scan arrives.
            self.tf_broadcaster.sendTransform(self._transform(
                stamp, "odom", "base_link", recording.base_positions[index], recording.base_rotations[index]))
            self.odom_publisher.publish(self._odometry(stamp, index))
            self.scan_publisher.publish(self._scan(stamp, index, period))
        self.get_logger().info("Replay finished")


def main():
    rclpy.init()
    node = ReplayNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
