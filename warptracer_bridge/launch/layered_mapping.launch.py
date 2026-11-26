"""One Cartographer per layer of a two-layer WarpTracer track, fed by layer_mapper.

Start it before the replay, and restart it before each new replay:
    ros2 launch warptracer_bridge layered_mapping.launch.py

The replay publishes base_link -> laser itself, so unlike roboracer_slam's launch
files this one does not.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

LAYERS = 2


def generate_launch_description():
    config_dir = PathJoinSubstitution([FindPackageShare("roboracer_slam"), "config"])
    sim_time = {"use_sim_time": True}
    nodes = []
    for layer in range(LAYERS):
        namespace = f"layer_{layer}"
        nodes.append(Node(
            package="cartographer_ros", executable="cartographer_node", namespace=namespace,
            name="cartographer_node", output="screen", parameters=[sim_time],
            arguments=["-configuration_directory", config_dir,
                       "-configuration_basename", f"roboracer_cartographer_layer_{layer}.lua"]))
        nodes.append(Node(
            package="cartographer_ros", executable="cartographer_occupancy_grid_node", namespace=namespace,
            name="cartographer_occupancy_grid_node", output="screen", parameters=[sim_time],
            arguments=["-resolution", "0.05"]))
    nodes.append(Node(package="warptracer_bridge", executable="layer_mapper", output="screen",
                      parameters=[sim_time, {"layers": LAYERS}]))
    return LaunchDescription(nodes)
