```bash
# F1tenth stack
source install/setup.bash && ros2 launch f1tenth_stack bringup_launch.py
# Foxglove
source install/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml
# Disparity extender:
source install/setup.bash && ros2 launch disparity_extender disparity_extender.launch.py
# warpSLAM
source install/setup.bash && python3 warpSLAM/slam_node.py
```

## Running in the AutoDRIVE Roboracer simulator

The `.devcontainer/AutoDRIVE` container is based on `autodriveecosystem/autodrive_roboracer_api:2026-icra-practice`
and includes the sim. AutoDRIVE uses different topic names and a Float32 drive
interface, so we layer a small bridge (`autodrive_bridge`) on top.

```bash
# 1. From host: VS Code "Dev Containers: Reopen in Container" -> AutoDRIVE.

# 2. Inside container, build the workspace.
colcon build --symlink-install
source install/setup.bash

# 3. Start the sim (headless avoids needing X11 from Windows).
ros2 launch autodrive_roboracer bringup_headless.launch.py &

# 4. Start foxglove for visualization (browse to ws://localhost:8765 on host).
ros2 launch foxglove_bridge foxglove_bridge_launch.xml &

# 5. Launch the bridge + driver with the right topic remaps.
ros2 launch autodrive_bridge sim.launch.py

# 6. Arm the car: on the Windows host, run `python deadman_switch.py`
#    and update NANO_HOST in the script to "host.docker.internal" or your
#    container IP so the UDP packets reach the driver node.
```

Topic mapping handled by the launch file / bridge:

| Driver code | AutoDRIVE topic |
| --- | --- |
| `/scan` | `/autodrive/roboracer_1/lidar` |
| `/sensors/imu` | `/autodrive/roboracer_1/imu` |
| `/drive` (AckermannDriveStamped) | `/autodrive/roboracer_1/{steering,throttle}_command` (Float32, via bridge) |

The bridge maps `drive.speed` (m/s) to normalized throttle linearly via the
`max_speed` parameter — tune in `sim.launch.py` if the car saturates or stalls.

## Recording and replaying bags for SLAM evaluation

Drive the car once manually, record the sensors + reference odom, then replay
that bag through each SLAM stack and compare against the recorded odom in
Foxglove. The reference odom is the car's wheel odometry (from
`vesc_to_odom_node` in `f1tenth_stack`) — not ground truth, but a useful
baseline for spotting SLAM drift.

### 1. Confirm the reference odom topic (on the car)

```bash
source install/setup.bash && ros2 launch f1tenth_stack bringup_launch.py &
ros2 topic list | grep -i odom
ros2 topic info /odom        # confirm nav_msgs/msg/Odometry
ros2 topic hz   /odom        # confirm ~50 Hz
```

### 2. Record (on the car, deadman armed for manual drive)

```bash
mkdir -p bags && cd bags
ros2 bag record /scan /sensors/imu /odom /tf /tf_static \
  -o run_$(date +%Y%m%d_%H%M%S)
```

`/tf_static` is mandatory — without it Foxglove can't place the scan in `base_link`.

### 3. Replay through each SLAM (separately)

Open three terminals. **Terminal A** plays the bag with simulated clock:

```bash
ros2 bag play bags/run_<timestamp> --clock
```

**Terminal B** runs one SLAM at a time (never both — they collide on `/map`):

```bash
# Cartographer pass
source install/setup.bash
ros2 launch cartographer cartographer_replay.launch.py

# warpSLAM pass (in a fresh shell after stopping cartographer)
source install/setup.bash
cd warpSLAM && python3 slam_node.py --ros-args -p use_sim_time:=true
```

**Terminal C** exposes everything to Foxglove:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

### 4. View in Foxglove

On the host, open Foxglove Studio → connect to `ws://localhost:8765`. In a 3D
panel, add:

- `/map` (OccupancyGrid) — the SLAM-built map
- `/odom` as a line/trail with long history — the reference trajectory
- TF `map → base_link` (auto, from Cartographer) or `/slam_pose` (from warpSLAM)
- `/scan` — sanity check the live scan hugs the map walls

The map is good if the `/odom` trail stays inside the corridors of the
occupancy grid. Save the layout to `foxglove/slam_eval.json` so reruns are one
click ("Import/Export layout" in the top-right of the Foxglove UI).

### 5. Quantitative drift (optional)

[eval/slam_eval.py](eval/slam_eval.py) subscribes to `/odom` (reference) and
the SLAM's pose output, time-aligns them, and prints mean/max/RMS position +
heading error at exit. Run it in a fourth terminal alongside the bag replay:

```bash
# warpSLAM (publishes /slam_pose)
python3 eval/slam_eval.py --slam-pose-topic /slam_pose

# Cartographer (publishes TF map->base_link, no pose topic)
python3 eval/slam_eval.py --from-tf
```

Each run writes `slam_eval.csv` for offline plotting. Rename it between SLAMs
(`mv slam_eval.csv cartographer.csv`) so you can diff them.

### Sanity checks while replaying

```bash
ros2 bag info bags/run_<timestamp>      # non-zero counts on all topics
ros2 topic hz /map                       # > 0 within ~10s of bag start
ros2 run tf2_tools view_frames           # map -> odom -> base_link -> lidar
```
