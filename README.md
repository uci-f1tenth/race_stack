```bash
# F1tenth stack
colcon build && source install/setup.bash && ros2 launch f1tenth_stack bringup_launch.py
# publish one commoand
ros2 topic pub --once /drive ackermann_msgs/msg/AckermannDriveStamped "{header: {frame_id: 'laser'}, drive: {steering_angle: 0.0, speed: 0.0}}"
# run cartographer
colcon build && source install/setup.bash && ros2 launch roboracer_slam roboracer_cartographer.launch.py
# save map
ros2 run nav2_map_server map_saver_cli -f my_map --ros-args -p save_map_timeout:=10000.0
# Deadman switch (car won't move without it), then open http://<car ip>:5005 on the laptop and hold
colcon build && source install/setup.bash && ros2 launch deadman_switch deadman_switch.launch.py
# Foxglove
colcon build && source install/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml
# Disparity extender:
colcon build && source install/setup.bash && ros2 launch disparity_extender disparity_extender.launch.py
# Warporacer:
colcon build && source install/setup.bash && ros2 launch warporacer_node warporacer_node.launch.py
# warpSLAM
colcon build && source install/setup.bash && python3 /home/f1t/f1tenth_ws/src/race_stack/warpSLAM/slam_node.py
```
```
f1tenth_stack changes for the deadman switch (replaces the F710):
config/mux.yaml          navigation topic: drive -> drive_gated, delete the joystick block
launch/bringup_launch.py delete ld.add_action(joy_node) and ld.add_action(joy_teleop_node)
```
```
steering_angle_to_servo_gain: -0.58
steering_angle_to_servo_offset: 0.45
speed_to_erpm: 3000
```
```
ssh f1t@10.42.0.1 # over AP
ssh f1t@192.168.55.1 # over usb
ssh f1t@100.111.6.29 # tailscale
```
WARPORACER_STEER_GAIN
WARPORACER_SPEED_SCALE
