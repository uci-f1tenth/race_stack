"""Replay a recorded drive through cartographer offline and score the map.

Inside the tuning image, with the bag at /bag, race_stack at /race_stack and an
output dir at /runs:
    python3 /race_stack/tuning/run.py NAME [--dist 0.9] [--shift 0.01] \
        [--lua 'POSE_GRAPH.optimize_every_n_nodes = 30'] [--recorded_odom]
Prints one JSON line; the map, config and log land in /runs/NAME.
"""

import argparse
import json
import os
import shutil
import subprocess

import numpy as np

import tools

CARTO = "/opt/ros/humble/lib/cartographer_ros/"
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--bag", default="/bag")
ap.add_argument(
    "--config", default="/race_stack/roboracer_slam/config/roboracer_cartographer.lua"
)
ap.add_argument("--out", default="/runs")
ap.add_argument("--shift", type=float, default=0.0, help="s added to scan stamps")
ap.add_argument("--dist", type=float, default=1.0, help="odometry distance scale")
ap.add_argument("--gyro", type=float, default=1.0, help="gyro scale")
ap.add_argument("--lidar_x", type=float, default=None, help="base_link->laser x (m)")
ap.add_argument(
    "--recorded_odom", action="store_true", help="use /odom exactly as recorded"
)
ap.add_argument(
    "--lua", action="append", default=[], help="extra config line, repeatable"
)
a = ap.parse_args()

run = os.path.join(a.out, a.name)
shutil.rmtree(run, ignore_errors=True)
os.makedirs(run)
tools.make_bag(
    a.bag,
    os.path.join(run, "bag"),
    a.shift,
    a.dist,
    a.gyro,
    a.lidar_x,
    not a.recorded_odom,
)
# The bag has no odom->base_link TF, so publish base_link (only affects TF output).
lua = open(a.config).read().replace("return options", "")
lua += (
    'options.published_frame = "base_link"\n' + "\n".join(a.lua) + "\nreturn options\n"
)
open(os.path.join(run, "cfg.lua"), "w").write(lua)

log = subprocess.run(
    [
        CARTO + "cartographer_offline_node",
        "-configuration_directory",
        run,
        "-configuration_basenames",
        "cfg.lua",
        "-bag_filenames",
        os.path.join(run, "bag"),
        "-save_state_filename",
        os.path.join(run, "out.pbstream"),
    ],
    capture_output=True,
    text=True,
)
open(os.path.join(run, "log.txt"), "w").write(log.stdout + log.stderr)
subprocess.run(
    [
        CARTO + "cartographer_pbstream_to_ros_map",
        "-pbstream_filename",
        os.path.join(run, "out.pbstream"),
        "-map_filestem",
        os.path.join(run, "map"),
        "-resolution",
        "0.05",
    ],
    capture_output=True,
)

pose_graph = tools.read_pbstream(os.path.join(run, "out.pbstream"))
res, n_nodes, n_submaps = tools.loop_closure_residuals(pose_graph)
drift = tools.drift_from_log(log.stdout + log.stderr)
entropy, amb = tools.map_crispness(os.path.join(run, "map.pgm"))


def pct(x, q):
    return round(float(np.percentile(x, q)) * 100, 1) if len(x) else None


print(
    json.dumps(
        {
            "name": a.name,
            "nodes": n_nodes,
            "submaps": n_submaps,
            "loop_closures": len(res),
            # loop closures vs the running estimate: how far local SLAM drifted
            "drift_cm_p50": pct(drift[:, 0], 50),
            "drift_cm_p90": pct(drift[:, 0], 90),
            # what's still inconsistent after the final optimization
            "residual_cm_p50": pct(res[:, 0], 50),
            "score_p50": (
                round(float(np.median(drift[:, 2])), 1) if len(drift) else None
            ),
            "entropy": round(entropy, 4),
            "ambiguous_per_wall": round(amb, 2),
        }
    )
)
