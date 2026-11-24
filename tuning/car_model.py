"""Fit how the car follows its commands, for warporacer's sim (warporacer/sim.py).

Record while warporacer drives (README), then inside the tuning image:
    python3 /race_stack/tuning/car_model.py /bags/run1 /bags/run2 ...
Uses AUTO driving clear of walls. Prints the steering delay, lag, gain and offset, the
grip, and the speed loop's lag, acceleration and braking, and how far 1 s rollouts of the
recorded commands land from where localization saw the car, for the fitted model and
for instant response with 1 g of grip. Re-run after changing tires, floor or servo.
"""

import math
import sys

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.optimize import minimize
from scipy.signal import lfilter

import tools

WHEELBASE = 0.3302  # m, as in the sim
BODY = (-0.12, 0.46, 0.21)  # car outline in base_link: rear, front, half width (m)
DT = 0.01  # s, analysis grid
H = 100  # steps per validation rollout (1 s)
TOPICS = ("/drive_gated", "/odom", "/sensors/imu/raw", "/scan", "/tf", "/tf_static")


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def load(path):
    """One bag -> 100 Hz arrays, with a mask of AUTO driving clear of walls."""
    types, msgs = tools.read_bag(path)
    cls = {k: get_message(v) for k, v in types.items() if k in TOPICS + ("/deadman_armed",)}
    gated, odom, imu, armed, m2o, o2b, scans = [], [], [], [], [], [], []
    laser_x, ang = 0.2733, None
    for topic, data, t in msgs:
        if topic not in cls:
            continue
        m, t = deserialize_message(data, cls[topic]), t * 1e-9
        if topic == "/drive_gated":
            gated.append((t, m.drive.steering_angle, m.drive.speed))
        elif topic == "/odom":
            odom.append((t, m.twist.twist.linear.x))
        elif topic == "/sensors/imu/raw":  # this VESC reports deg/s and g
            a = m.linear_acceleration
            imu.append((t, math.radians(m.angular_velocity.z), math.hypot(a.x, a.y)))
        elif topic == "/deadman_armed":
            armed.append((t, float(m.data)))
        elif topic == "/scan":
            scans.append((tools.stamp(m.header), t, np.asarray(m.ranges, np.float32)))
            ang = m.angle_min + np.arange(len(m.ranges)) * m.angle_increment
        else:
            for tf in m.transforms:
                row = (
                    tools.stamp(tf.header),
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    yaw(tf.transform.rotation),
                )
                if (tf.header.frame_id, tf.child_frame_id) == ("map", "odom"):
                    m2o.append(row)
                elif (tf.header.frame_id, tf.child_frame_id) == ("odom", "base_link"):
                    o2b.append(row)
                elif tf.child_frame_id == "laser":
                    laser_x = tf.transform.translation.x
    gated, odom, imu, armed = (np.array(x) for x in (gated, odom, imu, armed))
    m2o, o2b = np.array(sorted(m2o)), np.array(sorted(o2b))
    o2b[:, 3] = np.unwrap(o2b[:, 3])
    st, sbt = np.array([s[0] for s in scans]), np.array([s[1] for s in scans])
    # map -> base_link at each scan: latest map -> odom, odometry interpolated
    ox, oy, oth = (np.interp(st, o2b[:, 0], o2b[:, k]) for k in (1, 2, 3))
    j = np.clip(np.searchsorted(m2o[:, 0], st, side="right") - 1, 0, None)
    mx, my, mth = m2o[j, 1], m2o[j, 2], m2o[j, 3]
    px, py = mx + np.cos(mth) * ox - np.sin(mth) * oy, my + np.sin(mth) * ox + np.cos(mth) * oy
    # wall contacts: scan points within 2 cm of the car's outline
    bad = []
    for _, t, rg in scans:
        ok = (rg > 0.05) & (rg < 1.0) & ~((ang > math.radians(115)) & (rg < 0.3))  # own body
        x, y = laser_x + rg[ok] * np.cos(ang[ok]), rg[ok] * np.sin(ang[ok])
        gap = np.hypot(
            np.maximum(np.maximum(BODY[0] - x, x - BODY[1]), 0), np.maximum(np.abs(y) - BODY[2], 0)
        )
        if len(gap) and gap.min() < 0.02:
            bad.append(t)
    bad += list(imu[imu[:, 2] > 1.3, 0])  # impacts
    g = np.arange(
        max(gated[0, 0], odom[0, 0], imu[0, 0], sbt[0]),
        min(gated[-1, 0], odom[-1, 0], imu[-1, 0], sbt[-1]),
        DT,
    )
    i = np.clip(np.searchsorted(gated[:, 0], g, side="right") - 1, 0, None)
    v, w = np.interp(g, odom[:, 0], odom[:, 1]), np.interp(g, imu[:, 0], imu[:, 1])
    still = np.abs(v) < 0.01
    if still.sum() > 100:
        w = w - np.median(w[still])  # gyro bias
    clean = np.interp(g, armed[:, 0], armed[:, 1]) > 0.5
    for t in bad:
        clean &= np.abs(g - t) > 0.5
    return dict(
        cs=gated[i, 1],
        cv=gated[i, 2],
        v=v,
        w=w,
        clean=clean,
        px=np.interp(g, sbt, px),
        py=np.interp(g, sbt, py),
        pth=np.interp(g, sbt, np.unwrap(mth + oth)),
    )


def shift(x, k):
    return np.concatenate([np.full(k, x[0]), x[: len(x) - k]]) if k > 0 else x


def lag(x, tau):
    a = 1 - math.exp(-DT / max(tau, 1e-4))
    return lfilter([a], [1, a - 1], x, zi=[x[0] * (1 - a)])[0]


def yaw_rate(v, wheel, gain, offset, grip):
    """Kinematic yaw rate, lateral acceleration saturating as grip * tanh(a / grip)."""
    wk = v * np.tan(gain * wheel + offset) / WHEELBASE
    z = v * wk / grip
    return wk * np.where(np.abs(z) > 1e-6, np.tanh(z) / np.where(np.abs(z) > 1e-6, z, 1), 1)


def fit_steering(runs):
    """Delay (whole 10 ms steps), lag, gain, offset, grip: least squares on yaw rate."""
    best = None
    for delay in range(0, 13):

        def err(p):
            tau, gain, offset, grip = p
            if not (0.001 < tau < 0.3 and 0.5 < gain < 1.5 and grip > 1):
                return 1e3
            e = [
                yaw_rate(d["v"], lag(shift(d["cs"], delay), tau), gain, offset, grip) - d["w"]
                for d in runs
            ]
            m = [d["clean"] & (np.abs(d["v"]) > 0.4) for d in runs]
            return np.sqrt(np.mean(np.concatenate([x[k] for x, k in zip(e, m, strict=True)]) ** 2))

        r = minimize(
            err,
            [0.03, 1.0, 0.0, 8.0],
            method="Nelder-Mead",
            options=dict(maxiter=800, xatol=1e-4, fatol=1e-6),
        )
        if best is None or r.fun < best[0]:
            best = (r.fun, delay * DT, *r.x)
    return best


def speed_step(v, cmd, tau, accel, brake):
    """First-order approach to the command (exact over DT, so any lag is stable), at most
    accel / brake m/s^2."""
    dv = (cmd - v) * (1 - math.exp(-DT / max(tau, 1e-4)))
    return v + np.clip(dv, -brake * DT, accel * DT)


def speed_rollout(d, s, delay, tau, accel, brake):
    """Measured speed after 1 s of the commands from each start s."""
    cv, v = shift(d["cv"], delay), d["v"][s].copy()
    for j in range(H):
        v = speed_step(v, cv[s + j], tau, accel, brake)
    return v


def fit_speed(runs):
    """The VESC speed loop on warporacer's own commands (>= 0.9 m/s: lower ones make
    the VESC coast)."""
    data = []
    for d in runs:
        ok = d["clean"] & (d["cv"] >= 0.9)
        s = np.array([k for k in range(0, len(ok) - H, 5) if ok[k : k + H + 1].all()], int)
        if len(s):
            data.append((d, s))
    best = None
    for delay in range(0, 7):

        def err(p):
            if min(p) <= 0.01:
                return 1e3
            return np.mean(
                np.concatenate(
                    [np.abs(speed_rollout(d, s, delay, *p) - d["v"][s + H]) for d, s in data]
                )
            )

        r = minimize(err, [0.1, 6.0, 6.0], method="Nelder-Mead", options=dict(maxiter=600))
        if best is None or r.fun < best[0]:
            best = (r.fun, delay * DT, *r.x)
    return best


def pose_rollouts(d, model):
    """Position and heading errors of 1 s rollouts of the recorded commands."""
    sd, tau_s, gain, offset, grip, vd, tau_v, accel, brake = model
    wheel, cv = lag(shift(d["cs"], round(sd / DT)), tau_s), shift(d["cv"], round(vd / DT))
    ok = d["clean"] & (np.abs(d["v"]) > 0.3)
    s = np.array([k for k in range(0, len(ok) - H, 5) if ok[k : k + H + 1].all()], int)
    if not len(s):
        return np.zeros((0, 2))
    x, y, th, v = d["px"][s], d["py"][s], d["pth"][s], d["v"][s].copy()
    for j in range(H):
        v = speed_step(v, cv[s + j], tau_v, accel, brake)
        th = th + yaw_rate(v, wheel[s + j], gain, offset, grip) * DT
        x, y = x + v * np.cos(th) * DT, y + v * np.sin(th) * DT
    e = s + H
    head = np.abs((th - d["pth"][e] + math.pi) % (2 * math.pi) - math.pi)
    return np.column_stack([np.hypot(x - d["px"][e], y - d["py"][e]), np.degrees(head)])


def main():
    runs = [load(p) for p in sys.argv[1:]]
    clean = sum(d["clean"].sum() for d in runs) * DT
    print(f"{len(runs)} bags, {clean:.0f} s of AUTO driving clear of walls")
    rs, sd, tau_s, gain, offset, grip = fit_steering(runs)
    rv, vd, tau_v, accel, brake = fit_speed(runs)
    print(
        f"steering: {sd * 1e3:.0f} ms delay + {tau_s * 1e3:.0f} ms lag, gain {gain:.3f}, "
        f"offset {math.degrees(offset):+.2f} deg; grip {grip:.1f} m/s^2 "
        f"(yaw rate rms {math.degrees(rs):.1f} deg/s)"
    )
    print(
        f"speed loop: {vd * 1e3:.0f} ms delay + {tau_v * 1e3:.0f} ms lag, accel {accel:.1f}, "
        f"brake {brake:.1f} m/s^2 (1 s speed error {rv:.3f} m/s)"
    )
    fitted = (sd, tau_s, gain, offset, grip, vd, tau_v, accel, brake)
    instant = (0.0, 1e-4, 1.0, 0.0, 10.3, 0.0, 1e-4, 1e3, 1e3)
    for name, m in (("fitted model", fitted), ("instant, 1 g", instant)):
        e = np.concatenate([pose_rollouts(d, m) for d in runs])
        print(
            f"1 s rollouts, {name:12s}: position {np.median(e[:, 0]) * 100:.1f} cm median, "
            f"{np.percentile(e[:, 0], 90) * 100:.0f} p90; heading {np.median(e[:, 1]):.1f} "
            f"deg median, {np.percentile(e[:, 1], 90):.0f} p90 (n={len(e)})"
        )


if __name__ == "__main__":
    main()
