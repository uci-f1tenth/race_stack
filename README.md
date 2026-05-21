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

<!--Please fix, this doesn't work:
``` bash
# google cartographer
ros2 launch launch/cartographer_launch.py
# Saving SLAM Map
cd /workspaces/race_stack/maps
ros2 run nav2_map_server map_saver_cli -f my_track_map
```-->
