import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_dir = os.path.abspath('config')
    lua_file = 'roboracer_cartographer.lua'

    # The bridge that tells Cartographer where the LiDAR is mounted
    lidar_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='roboracer_lidar_tf',
        arguments=['0', '0', '0.1', '0', '0', '0', 'roboracer_1', 'lidar']
    )

    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': False}],
        arguments=[
            '-configuration_directory', config_dir,
            '-configuration_basename', lua_file
        ],
        remappings=[
            ('scan', '/autodrive/roboracer_1/lidar')
        ]
    )

    occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': False}],
        arguments=['-resolution', '0.05']
    )

    return LaunchDescription([
        lidar_tf_node,
        cartographer_node,
        occupancy_grid_node
    ])
