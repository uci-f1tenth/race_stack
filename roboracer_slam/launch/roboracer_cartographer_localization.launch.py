from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("roboracer_slam")
    config_dir = PathJoinSubstitution([pkg_share, "config"])

    use_sim_time = LaunchConfiguration("use_sim_time")
    load_state_filename = LaunchConfiguration("load_state_filename")

    declare_args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("load_state_filename"),
    ]

    cartographer_node = Node(
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[
            "-configuration_directory",
            config_dir,
            "-configuration_basename",
            "roboracer_cartographer_localization.lua",
            "-load_state_filename",
            load_state_filename,
        ],
    )

    occupancy_grid_node = Node(
        package="cartographer_ros",
        executable="cartographer_occupancy_grid_node",
        name="cartographer_occupancy_grid_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=["-resolution", "0.05"],
    )

    return LaunchDescription(
        declare_args
        + [
            cartographer_node,
            occupancy_grid_node,
        ]
    )
