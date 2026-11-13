"""Car bringup: replaces f1tenth_stack's bringup_launch.py.

Same driver nodes, but with race_stack's configs, no joystick (joy /
joy_teleop), the websocket deadman switch gating /drive, imu_odom in place of
vesc_to_odom, a motor watchdog, and a low-battery shutdown. The car can only
move while the deadman page is held.
"""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_dir = PathJoinSubstitution([FindPackageShare("race_bringup"), "config"])
    vesc_config = PathJoinSubstitution([config_dir, "vesc.yaml"])
    sensors_config = PathJoinSubstitution([config_dir, "sensors.yaml"])
    mux_config = PathJoinSubstitution([config_dir, "mux.yaml"])

    return LaunchDescription(
        [
            Node(
                package="vesc_driver",
                executable="vesc_driver_node",
                name="vesc_driver_node",
                parameters=[vesc_config],
            ),
            Node(
                package="vesc_ackermann",
                executable="ackermann_to_vesc_node",
                name="ackermann_to_vesc_node",
                parameters=[vesc_config],
            ),
            Node(
                package="race_bringup",
                executable="imu_odom",
                name="imu_odom",
                parameters=[vesc_config],
            ),
            Node(
                package="race_bringup",
                executable="motor_watchdog",
                name="motor_watchdog",
            ),
            Node(
                package="race_bringup",
                executable="low_voltage_shutdown",
                name="low_voltage_shutdown",
            ),
            Node(
                package="urg_node",
                executable="urg_node_driver",
                name="urg_node",
                parameters=[sensors_config],
            ),
            Node(
                package="ackermann_mux",
                executable="ackermann_mux",
                name="ackermann_mux",
                parameters=[mux_config],
            ),
            Node(
                package="deadman_switch",
                executable="deadman_switch",
                name="deadman_switch",
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_baselink_to_laser",
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
            ),
        ]
    )
