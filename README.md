```bash
# Always build and source from ~/f1tenth_ws: building anywhere else makes a second workspace
# whose stale packages can shadow these.
# Car bringup (replaces f1tenth_stack bringup_launch.py: no joystick; deadman switch, IMU odom, low-battery shutdown)
# then open http://<car ip>:5005 on the laptop: MANUAL = drag or W/A/S/D to drive (for mapping),
# AUTO = hold the pad or space bar to let a running controller drive. Letting go stops the car.
cd ~/f1tenth_ws && colcon build && source install/setup.bash && ros2 launch race_bringup bringup_launch.py
# publish one command (needs the page held in AUTO)
ros2 topic pub --once /drive ackermann_msgs/msg/AckermannDriveStamped "{header: {frame_id: 'laser'}, drive: {steering_angle: 0.0, speed: 0.0}}"
# run cartographer
cd ~/f1tenth_ws && colcon build && source install/setup.bash && ros2 launch roboracer_slam roboracer_cartographer.launch.py
# save map: finish the trajectory first so the last submaps and pending loop closures get optimized in,
# wait a few seconds for /map to republish, then save
ros2 service call /finish_trajectory cartographer_ros_msgs/srv/FinishTrajectory "{trajectory_id: 0}"
ros2 run nav2_map_server map_saver_cli -f my_map --ros-args -p save_map_timeout:=10000.0
# save the cartographer state too; roboracer_cartographer_localization.launch.py needs it (load_state_filename:=...)
ros2 service call /write_state cartographer_ros_msgs/srv/WriteState "{filename: '/home/f1t/my_map.pbstream', include_unfinished_submaps: true}"
# Foxglove
cd ~/f1tenth_ws && colcon build && source install/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml
# Disparity extender:
cd ~/f1tenth_ws && colcon build && source install/setup.bash && ros2 launch disparity_extender disparity_extender.launch.py
# Warporacer (model: copy it to race_stack/warporacer_node/agent_final.pt, or set WARPORACER_CHECKPOINT)
cd ~/f1tenth_ws && colcon build && source install/setup.bash && ros2 launch warporacer_node warporacer_node.launch.py
# check speed_to_erpm_gain: drives 1 m by odometry (hold the deadman), then measure the real distance
cd ~/f1tenth_ws && source install/setup.bash && python3 src/race_stack/calibrate.py
```
```
Car configs live in race_bringup/config (copied from the car's f1tenth_stack, changes marked "race_stack:")
One-time car setup:
- VESC Tool > App Settings > General: Timeout ~200 ms with a non-zero Timeout Brake Current. If the
  stack or Jetson dies the VESC otherwise keeps the last command for 1 s, then coasts. Test on a stand:
  arm with the wheels spinning, kill race_bringup, and the wheels should stop.
- ~/.bashrc: export ROS_DOMAIN_ID=<unique number>, so other ROS machines on a shared network can't
  publish motor commands to this car
- low-battery shutdown needs (sudo visudo -f /etc/sudoers.d/shutdown):
  f1t ALL=(root) NOPASSWD: /usr/sbin/shutdown
```
```
ssh f1t@10.42.0.1 # over AP
ssh f1t@192.168.55.1 # over usb
ssh f1t@100.111.6.29 # tailscale
```
WARPORACER_STEER_GAIN
WARPORACER_SPEED_SCALE
