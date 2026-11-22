"""Measure odometry against a lidar-only cartographer run of the same bag.

First a lidar-only run, then the check:
    python3 /race_stack/tuning/run.py lidar_only --lua "options.use_odometry = false"
    python3 /race_stack/tuning/odom_check.py /runs/lidar_only
Prints the odometry distance scale (-> speed_to_erpm_gain), gyro scale, and how
far the lidar sits ahead of the point that tracks straight through turns.
"""

import math
import sys

import numpy as np
from rclpy.serialization import deserialize_message

import tools

L = tools.nodes(tools.read_pbstream(sys.argv[1] + "/out.pbstream"))
types, msgs = tools.read_bag(sys.argv[2] if len(sys.argv) > 2 else "/bag")
cls = tools.message_classes(types)
od = []
for topic, data, _ in msgs:
    if topic == "/odom":
        m = deserialize_message(data, cls[topic])
        p = m.pose.pose
        od.append(
            (
                tools.stamp(m.header),
                p.position.x,
                p.position.y,
                2 * math.atan2(p.orientation.z, p.orientation.w),
                m.twist.twist.linear.x,
            )
        )
OD = np.array(od)
OD[:, 3] = np.unwrap(OD[:, 3])
# signed distance along the path, so reversing subtracts
s_od = np.concatenate(
    [[0], np.cumsum(np.hypot(*np.diff(OD[:, 1:3], axis=0).T) * np.sign(OD[1:, 4]))]
)

# lidar motion over ~0.3 s windows, in the body frame at the window start
pairs = []
for i in range(len(L)):
    j = np.searchsorted(L[:, 0], L[i, 0] + 0.3)
    if j >= len(L):
        break
    d = L[j, 1:3] - L[i, 1:3]
    c, s = math.cos(L[i, 3]), math.sin(L[i, 3])
    pairs.append(
        (L[i, 0], L[j, 0], c * d[0] + s * d[1], -s * d[0] + c * d[1], L[j, 3] - L[i, 3])
    )
P = np.array(pairs)


def odom_delta(lag):
    t0, t1 = P[:, 0] + lag, P[:, 1] + lag
    return (
        np.interp(t1, OD[:, 0], s_od) - np.interp(t0, OD[:, 0], s_od),
        np.interp(t1, OD[:, 0], OD[:, 3]) - np.interp(t0, OD[:, 0], OD[:, 3]),
    )


def fit(y, x):  # least squares y = k x, with x the low-noise (odometry) side
    return (y @ x) / (x @ x)


errs = []
for lag in np.arange(-0.1, 0.1001, 0.005):
    ds, dh = odom_delta(lag)
    mv = np.abs(ds) > 0.02
    k = fit(P[mv, 2], ds[mv])
    errs.append((np.mean((P[mv, 2] - k * ds[mv]) ** 2), lag))
lag = min(errs)[1]
ds, dh = odom_delta(lag)
mv = np.abs(ds) > 0.02
k = fit(P[mv, 2], ds[mv])
print(f"{len(L)} lidar nodes, best odom time lag {lag * 1e3:+.0f} ms")
print(f"distance: lidar/odom {k:.4f} -> speed_to_erpm_gain x {1 / k:.4f}")
for name, sel in [("forward", mv & (ds > 0)), ("reverse", mv & (ds < 0))]:
    if sel.sum() > 5:
        print(f"  {name}: {fit(P[sel, 2], ds[sel]):.4f} (n={sel.sum()})")
turning = math.degrees(np.abs(dh).sum())
print(f"heading: lidar/gyro {fit(P[:, 4], dh):.4f} over {turning:.0f} deg of turning")
slip = fit(P[:, 3], dh)
print(
    f"lateral slip vs rotation: {slip:+.3f} m (0 if base_link->laser x matches the "
    "point that tracks straight through turns)"
)
