"""Fresh per-layer Cartographer mappers plus one replay, as a single repeatable run.

Cartographer cannot delete trajectories, so every replay needs new mapper nodes.
This starts them, waits for them to come up, then starts the replay:
    ros2 launch warptracer_bridge layered_replay.launch.py recording:=/data/bridge.npz maps:=/data/bridge.maps.npz

The maps stay up after the replay ends. Press Ctrl+C to stop, then run it again.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mappers = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        PathJoinSubstitution([FindPackageShare("warptracer_bridge"), "launch", "layered_mapping.launch.py"])))
    replay = Node(
        package="warptracer_bridge", executable="replay", output="screen",
        parameters=[{"recording": LaunchConfiguration("recording"),
                     "maps": LaunchConfiguration("maps"),
                     "rate": LaunchConfiguration("rate")}])
    return LaunchDescription([
        DeclareLaunchArgument("recording", default_value="/data/bridge.npz"),
        DeclareLaunchArgument("maps", default_value="/data/bridge.maps.npz"),
        DeclareLaunchArgument("rate", default_value="1.0"),
        DeclareLaunchArgument("startup_delay", default_value="3.0",
                              description="Seconds to let the mappers come up before the replay starts"),
        mappers,
        TimerAction(period=LaunchConfiguration("startup_delay"), actions=[replay]),
    ])
