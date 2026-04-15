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
# AI
```bash
python3 racing_agent.py maps/my_map.yaml checkpoints/agent_final.pt
```
# Disparity extender
```bash
cd disparity_extender && colcon build && source install/setup.bash
ros2 launch disparity_extender disparity_extender.launch.py
```
