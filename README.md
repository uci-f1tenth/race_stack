```bash
# F1tenth stack
colcon build && source install/setup.bash && ros2 launch f1tenth_stack bringup_launch.py
# publish one commoand
ros2 topic pub --once /drive ackermann_msgs/msg/AckermannDriveStamped "{header: {frame_id: 'laser'}, drive: {steering_angle: 0.0, speed: 0.0}}"
# run cartographer
colcon build && source install/setup.bash && ros2 launch roboracer_slam roboracer_cartographer.launch.py
# save map
ros2 run nav2_map_server map_saver_cli -f my_map --ros-args -p save_map_timeout:=10000.0
# Foxglove
colcon build && source install/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml
# Disparity extender:
colcon build && source install/setup.bash && ros2 launch disparity_extender disparity_extender.launch.py
# Warporacer:
colcon build && source install/setup.bash && ros2 launch warporacer_node warporacer_node.launch.py
# RC teleop (car side):
colcon build && source install/setup.bash && ros2 launch rc_teleop rc_teleop.launch.py
# RC teleop (laptop side, WASD / arrow keys):
python3 rc_teleop.py
# warpSLAM
colcon build && source install/setup.bash && python3 warpSLAM/slam_node.py
```
```
steering_angle_to_servo_gain: -0.58
steering_angle_to_servo_offset: 0.45
```
```
ssh f1t@10.42.0.1 # over AP
ssh f1t@192.168.55.1 # over usb
ssh f1t@100.111.6.29 # tailscale
```
