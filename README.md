```bash
# Always build and source from ~/f1tenth_ws: building anywhere else makes a second workspace
# whose stale packages can shadow these.
# Car bringup: VESC + lidar drivers, deadman switch, IMU odometry, motor watchdog, low-battery shutdown
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
# Foxglove: connect to ws://<car ip>:8765 from Chrome or the Foxglove desktop app (Safari blocks ws://)
cd ~/f1tenth_ws && colcon build && source install/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml
# Disparity extender:
cd ~/f1tenth_ws && colcon build && source install/setup.bash && ros2 launch disparity_extender disparity_extender.launch.py
# Warporacer (model: copy it to race_stack/warporacer_node/agent_final.pt, or set WARPORACER_CHECKPOINT)
cd ~/f1tenth_ws && colcon build && source install/setup.bash && ros2 launch warporacer_node warporacer_node.launch.py
# Warporacer with a pose prior (the policy also sees where cartographer puts it on the map):
# 1. map the track and save the map and state (commands above); don't Ctrl-C cartographer before saving
# 2. on a laptop, copy my_map.yaml/.pgm into warporacer/maps and train from warporacer/ (cars drive the
#    loop the way the car faced when mapping started; --reverse for the other way):
#      uv run python main.py maps/my_map.yaml --pose --no-use-wandb
# 3. copy logs/agent_final.pt to race_stack/warporacer_node/ and colcon build; start localization with the
#    car anywhere on the track (it finds itself on the map in ~10 s, standing still), then warporacer
#    (it won't drive until cartographer has matched the car to the map) and hold AUTO
cd ~/f1tenth_ws && source install/setup.bash && ros2 launch roboracer_slam roboracer_cartographer_localization.launch.py load_state_filename:=/home/f1t/my_map.pbstream
cd ~/f1tenth_ws && source install/setup.bash && WARPORACER_SPEED_SCALE=0.5 ros2 launch warporacer_node warporacer_node.launch.py
# check speed_to_erpm_gain: drives 1 m by odometry (hold the deadman), waits until the car is at rest, prints
# the odometry distance; compare with a tape (set gain = gain * odometry / measured)
cd ~/f1tenth_ws && source install/setup.bash && python3 src/race_stack/calibrate.py
```
```bash
# Tuning cartographer offline: record a drive with bringup running (drive slowly, include turns and revisits)
ros2 bag record -o ~/mapping_bag /scan /odom /sensors/core /sensors/imu/raw /tf_static
# copy the bag to a laptop with Docker, then from race_stack (each run prints drift/residual/crispness):
docker build -t race-tune tuning
docker run --rm -v ~/mapping_bag:/bag:ro -v $PWD:/race_stack:ro -v /tmp/runs:/runs -w /race_stack/tuning race-tune \
  bash -c 'source /opt/ros/humble/setup.bash && python3 run.py baseline && python3 run.py try --lua "POSE_GRAPH.optimize_every_n_nodes = 30"'
# odometry calibration from the same bag: lidar-only run, then compare odometry against it
#   python3 run.py lidar_only --lua "options.use_odometry = false" && python3 odom_check.py /runs/lidar_only
```
```
Setting up a car (Jetson Orin Nano, JetPack 6 / Ubuntu 22.04, ROS 2 Humble installed from apt):
git clone https://github.com/uci-f1tenth/race_stack ~/f1tenth_ws/src/race_stack
CLIENT_PASSWORD=... HOTSPOT_PASSWORD=... ~/f1tenth_ws/src/race_stack/car_setup/setup_car.sh
It imports the pinned vesc driver and ackermann_mux (car_setup/race_stack.repos; f1tenth_system is no
longer needed), installs apt deps via rosdep plus torch, the /dev/sensors/vesc udev rule, the sudoers
line the low-battery shutdown needs, the Hokuyo link, wifi (joins anetwork-guest when in range, else
its own f1t-nano hotspot; the LTE dongle is only the backup route) and ROS_DOMAIN_ID, then
builds. Safe to re-run. Then copy the policy to race_stack/warporacer_node/agent_final.pt and rebuild.
Car configs live in race_bringup/config (copied from f1tenth_stack's, changes marked "race_stack:").
If the deadman, mux or ackermann_to_vesc dies, motor_watchdog commands zero speed straight to
vesc_driver. If vesc_driver, the USB link or the Jetson dies, the VESC keeps the last command for its
own timeout (1 s by default, then coasts), so let go of the page before stopping the stack.
```
```
ssh f1t@10.42.0.1 # over AP
ssh f1t@192.168.55.1 # over usb
ssh f1t@100.111.6.29 # tailscale
```
WARPORACER_STEER_GAIN
WARPORACER_SPEED_SCALE
