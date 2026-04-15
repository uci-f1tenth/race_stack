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
# google cartographer
```bash
ros2 launch launch/cartographer_launch.py
```
# google cartographer
```bash
ros2 launch launch/cartographer_launch.py
```