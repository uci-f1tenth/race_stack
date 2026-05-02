# Foxglove
```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
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
# F1tenth stack
```bash
ros2 launch f1tenth_stack bringup_launch.py
```
# AI
```bash
python3 racing_agent.py maps/my_map.yaml checkpoints/agent_final.pt
```
