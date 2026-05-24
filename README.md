```bash
# F1tenth stack
colcon build && source install/setup.bash && ros2 launch f1tenth_stack bringup_launch.py
# publish one commoand
ros2 topic pub --once /drive ackermann_msgs/msg/AckermannDriveStamped "{header: {frame_id: 'laser'}, drive: {steering_angle: 0.0s, speed: 0.0}}"
# run cartographer
ros2 launch cartographer/launch/cartographer_launch.py
# Foxglove
colcon build && source install/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml
# Disparity extender:
colcon build && source install/setup.bash && ros2 launch disparity_extender disparity_extender.launch.py
# warpSLAM
colcon build && source install/setup.bash && python3 warpSLAM/slam_node.py
```
```
steering_angle_to_servo_gain: -0.58
steering_angle_to_servo_offset: 0.45
```

<!--Please fix, this doesn't work:
``` bash
# google cartographer
ros2 launch launch/cartographer_launch.py
# Saving SLAM Map
cd /workspaces/race_stack/maps
ros2 run nav2_map_server map_saver_cli -f my_track_map
```-->
