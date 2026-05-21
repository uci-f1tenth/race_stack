"""Bring up disparity_extender against the AutoDRIVE Roboracer simulator.

Remaps:
  /scan          <- /autodrive/roboracer_1/lidar
  /sensors/imu   <- /autodrive/roboracer_1/imu

The bridge node converts the AckermannDriveStamped on /drive into the two
Float32 command topics the AutoDRIVE sim subscribes to.
"""

from launch import LaunchDescription
from launch_ros.actions import Node

VEHICLE = "roboracer_1"


def generate_launch_description():
    base = f"/autodrive/{VEHICLE}"
    return LaunchDescription(
        [
            Node(
                package="autodrive_bridge",
                executable="autodrive_bridge",
                parameters=[{"vehicle": VEHICLE, "max_speed": 3.0}],
            ),
            Node(
                package="disparity_extender",
                executable="disparity_extender",
                remappings=[
                    ("/scan", f"{base}/lidar"),
                    ("/sensors/imu", f"{base}/imu"),
                ],
            ),
        ]
    )
