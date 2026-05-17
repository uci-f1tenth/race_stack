```bash
# F1tenth stack
ros2 launch f1tenth_stack bringup_launch.py
# Foxglove
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
# Disparity extender:
ros2 launch disparity_extender disparity_extender.launch.py
# warpSLAM
python3 warpSLAM/slam_node.py
```

<!--Please fix, this doesn't work:
``` bash
# google cartographer
ros2 launch launch/cartographer_launch.py
# Saving SLAM Map
cd /workspaces/race_stack/maps
ros2 run nav2_map_server map_saver_cli -f my_track_map
```-->
