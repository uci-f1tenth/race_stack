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
    scan_topic = LaunchConfiguration("scan_topic")
    odom_topic = LaunchConfiguration("odom_topic")

    declare_args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("load_state_filename"),
        DeclareLaunchArgument("scan_topic", default_value="/scan"),
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
    ]

    lidar_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_to_laser",
        output="screen",
        arguments=[
            "--x",
            "0.2733",
            "--y",
            "0.0",
            "--z",
            "0.096",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "base_link",
            "--child-frame-id",
            "laser",
        ],
    )

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
        remappings=[
            ("scan", scan_topic),
            ("odom", odom_topic),
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
            lidar_static_tf,
            cartographer_node,
            occupancy_grid_node,
        ]
    )
