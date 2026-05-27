"""Cartographer brought up against a recorded bag (use_sim_time + topic remap).

Run with `ros2 bag play <bag> --clock` in another terminal. The `--clock` flag
is what makes use_sim_time meaningful; without it, Cartographer falls back to
wall time and rejects scans whose stamps look ancient.

If your bag uses a different odom topic (e.g. /ego_racecar/odom), edit the
remap below to point at it. Cartographer subscribes to the relative topic
`odom` when use_odometry=true (see roboracer_cartographer.lua).
"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(os.path.dirname(current_dir), "config")
    lua_file = "roboracer_cartographer.lua"

    lidar_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="roboracer_lidar_tf",
        arguments=["0", "0", "0.15", "0", "0", "0", "base_link", "lidar"],
        parameters=[{"use_sim_time": True}],
    )

    cartographer_node = Node(
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=[
            "-configuration_directory", config_dir,
            "-configuration_basename", lua_file,
        ],
        remappings=[("odom", "/odom")],
    )

    occupancy_grid_node = Node(
        package="cartographer_ros",
        executable="cartographer_occupancy_grid_node",
        name="cartographer_occupancy_grid_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=["-resolution", "0.05"],
    )

    return LaunchDescription([lidar_tf_node, cartographer_node, occupancy_grid_node])
