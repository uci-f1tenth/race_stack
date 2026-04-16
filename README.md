# autodrive devkit
```bash
ros2 launch autodrive_roboracer bringup_headless.launch.py
```
# Foxglove
```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```
# slam_toolbox mapping
```bash
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=autodrive_online_async_roboracer.yaml
```
# Disparity extender:
```bash
cd disparity_extender && colcon build && source install/setup.bash
ros2 launch disparity_extender disparity_extender.launch.py
```
# google cartographer
```bash
ros2 launch launch/cartographer_launch.py
```
# Saving SLAM Map
```bash
cd /workspaces/race_stack/maps
ros2 run nav2_map_server map_saver_cli -f my_track_map
```