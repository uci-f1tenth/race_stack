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
with race_stack next to it. The car is capped at 1.0 m/s, below the demo's 1.5 m/s
default, because slower driving gives Cartographer more overlap between scans; 180 s
still covers two laps:
```bash
uv run warptracer demo bridge --max-speed 1.0 --seconds 180
uv run python ../race_stack/warptracer_bridge/tools/export_track_maps.py outputs/bridge.npz
```

Build the image from the race_stack root. Rebuild after changing `roboracer_slam` or
`warptracer_bridge`, then recreate the container:
```bash
docker build -f docker/mapping/Dockerfile -t race-stack-mapping .
```

Start the Foxglove bridge, replacing `<WarpTracer>` with the path to that repository.
Recordings are read from `/data`; exported maps are written to `/maps`:
```bash
docker run -d --rm --name replay -p 8765:8765 -v "<WarpTracer>/outputs:/data:ro" -v "<WarpTracer>/outputs/cartographer:/maps" race-stack-mapping ros2 launch foxglove_bridge foxglove_bridge_launch.xml use_sim_time:=true
```

Connect Foxglove to `ws://localhost:8765` and open a 3D panel with display frame `odom`.
Turn on `/scan`, `/layer_0/map` and `/layer_1/map`. For `/scan`, use a flat bright color
and point size 5, because the default is too dark and small to see.

### Map each layer with Cartographer

Start fresh mappers and the replay together. It takes as long as the recording (180 s
above):
```bash
docker exec -it replay bash -c "source /ws/install/setup.bash; ros2 launch warptracer_bridge layered_replay.launch.py"
```

It replays `/data/bridge.npz` with `/data/bridge.maps.npz` by default; pass
`recording:=...`, `maps:=...` or `rate:=...` to change them. Keep `rate` at 1: at 4x,
Cartographer falls behind and the maps degrade. `layer_mapper` sends each scan to the
layer the car is on, from `/layer`, and finishes a layer's trajectory when the car leaves
it. When the car returns, it starts a new trajectory placed by odometry. Each live map has
its own frame, `layer_N_map`, placed where the car first reached that layer.

The replay puts `base_link` at the rear axle, so `base_link -> laser` is the real car's
0.2733 m forward and 0.096 m up. It warns if a recording's mount differs; re-record those
with current WarpTracer.

The layer configs weight odometry 100x Cartographer's default in scan matching, because
on ramps the level LiDAR shoots over the walls climbing and hits the road descending.
The replay reports beams with no return at `range_max`, not `inf`, so Cartographer clears
3 m of free space along them (`missing_data_ray_length`); otherwise the car's own path on
the bridge deck stays low-confidence free, which a trinary map shows as unknown.

The maps stay up after the replay ends. To save them, run this in another terminal
before stopping the launch:
```bash
docker exec replay bash -c "source /ws/install/setup.bash; ros2 run warptracer_bridge export_maps --ros-args -p output:=/maps/bridge -p maps:=/data/bridge.maps.npz"
```

It writes `bridge_layer_N.pgm` and `.yaml` per layer (nav2 map_server format) to
`<WarpTracer>/outputs/cartographer`. Every layer is resampled into `odom` on one shared
grid, so the maps stack cell for cell and line up with the ground truth. With `maps` set,
it also writes `bridge_layer_N_check.ppm`, the map over the ground truth (black: wall on
a true wall, red: wall off one, green: mapped road), and prints how well each layer
matches. Without `maps`, the grid just covers all layers' maps.

Press Ctrl+C and run the launch again to map another replay; Cartographer cannot delete
trajectories, so each run starts new mappers.
`layered_mapping.launch.py` starts the mappers alone, for a scan source other than the
replay.

### Replay without mapping (optional)

To only watch a recording, run the replay on its own. Turn on `/ground_truth_map/current`
to see the road under the car. You can run it again once it finishes, but not two at
once, and not alongside the mapping launch:
```bash
docker exec replay bash -c "source /ws/install/setup.bash; ros2 run warptracer_bridge replay --ros-args -p recording:=/data/bridge.npz -p maps:=/data/bridge.maps.npz"
```

Stop the bridge when you are done:
```bash
docker rm -f replay
```
