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
# warpSLAM
colcon build && source install/setup.bash && python3 /home/f1t/f1tenth_ws/src/race_stack/warpSLAM/slam_node.py
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
WARPORACER_STEER_GAIN
WARPORACER_SPEED_SCALE

## Replay WarpTracer in Foxglove

Record the bridge track and export its maps. Run from the WarpTracer repository root,
with race_stack next to it:
```bash
uv run warptracer demo bridge
uv run python ../race_stack/warptracer_bridge/tools/export_track_maps.py outputs/bridge.npz
```

Build the image from the race_stack root. Rebuild after changing `roboracer_slam` or
`warptracer_bridge`, then recreate the container:
```bash
docker build -f docker/mapping/Dockerfile -t race-stack-mapping .
```

Start the Foxglove bridge, replacing `<WarpTracer>` with the path to that repository:
```bash
docker run -d --rm --name replay -p 8765:8765 -v "<WarpTracer>/outputs:/data:ro" race-stack-mapping ros2 launch foxglove_bridge foxglove_bridge_launch.xml use_sim_time:=true
```

Connect Foxglove to `ws://localhost:8765` and open a 3D panel with display frame `odom`.
Turn on `/ground_truth_map/current` and `/scan`. For `/scan`, use a flat bright color and
point size 5, because the default is too dark and small to see.

Start the replay. It takes 120 s, and you can run it again once it finishes, but not
two at once:
```bash
docker exec replay bash -c "source /ws/install/setup.bash; ros2 run warptracer_bridge replay --ros-args -p recording:=/data/bridge.npz -p maps:=/data/bridge.maps.npz"
```

Stop the bridge when you are done:
```bash
docker rm -f replay
```
