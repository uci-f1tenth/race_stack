"""Shared helpers for offline cartographer tuning (run inside the tuning image)."""

import math
import re
import struct
import zlib

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message

UNIX_FROM_CARTO = 62135596800.0  # cartographer time: 100 ns ticks since year 1


def read_bag(path):
    r = rosbag2_py.SequentialReader()
    r.open(
        rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    msgs = []
    while r.has_next():
        msgs.append(r.read_next())  # (topic, data, receive time ns)
    return types, msgs


def message_classes(types):
    # /sensors/core (vesc_msgs) isn't installed here and isn't needed
    return {
        k: get_message(v) for k, v in types.items() if not v.startswith("vesc_msgs/")
    }


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def set_stamp(h, t):
    ns = int(round(t * 1e9))
    h.stamp.sec, h.stamp.nanosec = ns // 1_000_000_000, ns % 1_000_000_000


class ImuOdom:
    """Offline copy of race_bringup/imu_odom.py (keep the two in step).

    Rebuilds /odom from the recorded speed (already erpm / gain, dead-banded) and
    the VESC gyro, with an optional distance and gyro scale.
    """

    def __init__(self, dist_scale=1.0, gyro_scale=1.0):
        self.k, self.g = dist_scale, gyro_scale
        self.yaw, self.yaw_t, self.yaw_rate = 0.0, None, 0.0
        self.gyro_t, self.bias, self.n_bias = None, 0.0, 0
        self.stationary = True
        self.x = self.y = 0.0
        self.last_t, self.last_heading = None, 0.0

    def advance(self, t, rate, trapezoid):
        if self.yaw_t is not None and t <= self.yaw_t:
            self.yaw_rate = rate
            return
        if self.yaw_t is not None and t - self.yaw_t < 0.5:
            self.yaw += (0.5 * (rate + self.yaw_rate) if trapezoid else rate) * (
                t - self.yaw_t
            )
        self.yaw_t, self.yaw_rate = t, rate

    def imu(self, t, gz_deg):
        dev = math.radians(gz_deg) - self.bias
        if self.stationary and abs(dev) < math.radians(5.0):
            self.n_bias += 1
            self.bias += max(0.01, 1.0 / self.n_bias) * dev
            rate = 0.0
        else:
            rate = self.g * dev
        self.gyro_t = t
        self.advance(t, rate, True)

    def odom(self, t, v_recorded):
        v = self.k * v_recorded
        self.stationary = v_recorded == 0.0
        if not (self.gyro_t is not None and t - self.gyro_t < 0.1):
            self.advance(t, 0.0, False)
        heading = self.yaw + self.yaw_rate * (t - self.yaw_t)
        if self.last_t is not None and 0.0 < t - self.last_t < 0.5:
            dt = t - self.last_t
            mid = 0.5 * (self.last_heading + heading)
            self.x += v * math.cos(mid) * dt
            self.y += v * math.sin(mid) * dt
        self.last_t, self.last_heading = t, heading
        return self.x, self.y, heading, v


def make_bag(
    src,
    dst,
    scan_shift=0.0,
    dist_scale=1.0,
    gyro_scale=1.0,
    lidar_x=None,
    recompute_odom=True,
):
    """Write /scan, /odom, /tf_static from src with the given input corrections."""
    types, msgs = read_bag(src)
    cls = message_classes(types)
    w = rosbag2_py.SequentialWriter()
    w.open(
        rosbag2_py.StorageOptions(uri=dst, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for name in ("/scan", "/odom", "/tf_static"):
        w.create_topic(
            rosbag2_py.TopicMetadata(
                name=name, type=types[name], serialization_format="cdr"
            )
        )
    io = ImuOdom(dist_scale, gyro_scale)
    for topic, data, t in msgs:
        if topic == "/sensors/imu/raw":
            if recompute_odom:
                m = deserialize_message(data, cls[topic])
                io.imu(stamp(m.header), m.angular_velocity.z)
            continue
        if topic not in ("/scan", "/odom", "/tf_static"):
            continue
        m = deserialize_message(data, cls[topic])
        if topic == "/scan" and scan_shift:
            set_stamp(m.header, stamp(m.header) + scan_shift)
        elif topic == "/tf_static" and lidar_x is not None:
            for tf in m.transforms:
                if tf.child_frame_id == "laser":
                    tf.transform.translation.x = lidar_x
        elif topic == "/odom" and recompute_odom:
            x, y, h, v = io.odom(stamp(m.header), m.twist.twist.linear.x)
            m.pose.pose.position.x, m.pose.pose.position.y = x, y
            m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(
                h / 2
            ), math.cos(h / 2)
            m.twist.twist.linear.x = v
        w.write(topic, serialize_message(m), t)
    del w


def read_pbstream(path):
    """-> PoseGraph proto (optimized node and submap poses, constraints)."""
    from cartographer.mapping.proto import serialization_pb2

    data = open(path, "rb").read()
    assert struct.unpack("<Q", data[:8])[0] == 0x7B1D1F7B5BF501DB, "not a pbstream"
    i = 8
    while i < len(data):
        n = struct.unpack("<Q", data[i : i + 8])[0]
        raw = zlib.decompress(data[i + 8 : i + 8 + n], 16 + zlib.MAX_WBITS)
        i += 8 + n
        sd = serialization_pb2.SerializedData()
        if sd.ParseFromString(raw) and sd.WhichOneof("data") == "pose_graph":
            return sd.pose_graph
    raise ValueError("no pose graph in " + path)


def pose2d(p):
    q = p.rotation
    return np.array(
        [
            p.translation.x,
            p.translation.y,
            math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
        ]
    )


def nodes(pg):
    """(unix time, x, y, yaw) of every optimized trajectory node, time-sorted."""
    rows = sorted(
        (n.timestamp * 1e-7 - UNIX_FROM_CARTO, *pose2d(n.pose))
        for t in pg.trajectory
        for n in t.node
    )
    out = np.array(rows)
    out[:, 3] = np.unwrap(out[:, 3])
    return out


def loop_closure_residuals(pg):
    """Translation (m) and rotation (rad) by which the optimized poses miss each
    loop closure: how self-consistent the final map is."""
    node_pose, submap_pose = {}, {}
    for t in pg.trajectory:
        node_pose.update({n.node_index: pose2d(n.pose) for n in t.node})
        submap_pose.update({s.submap_index: pose2d(s.pose) for s in t.submap})
    res = []
    for c in pg.constraint:
        sid, nid = c.submap_id.submap_index, c.node_id.node_index
        if c.tag != c.INTER_SUBMAP or sid not in submap_pose or nid not in node_pose:
            continue
        s, n, m = submap_pose[sid], node_pose[nid], pose2d(c.relative_pose)
        cs, sn = math.cos(s[2]), math.sin(s[2])
        d = n[:2] - s[:2]
        rel = np.array([cs * d[0] + sn * d[1], -sn * d[0] + cs * d[1], n[2] - s[2]])
        res.append(
            (
                math.hypot(*(rel[:2] - m[:2])),
                abs((rel[2] - m[2] + math.pi) % (2 * math.pi) - math.pi),
            )
        )
    return np.array(res).reshape(-1, 2), len(node_pose), len(submap_pose)


def read_pgm(path):
    data = open(path, "rb").read()
    tokens, i = [], 0
    while len(tokens) < 4:
        while data[i : i + 1].isspace():
            i += 1
        if data[i : i + 1] == b"#":
            while data[i : i + 1] != b"\n":
                i += 1
            continue
        j = i
        while not data[j : j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    w, h = int(tokens[1]), int(tokens[2])
    return np.frombuffer(data[i + 1 : i + 1 + w * h], dtype=np.uint8).reshape(h, w)


def map_crispness(pgm_path):
    """Mean cell entropy (bits, lower = crisper) and ambiguous cells per wall cell."""
    img = read_pgm(pgm_path).astype(float)
    p = np.clip(1.0 - img[img != 128] / 255.0, 1e-3, 1 - 1e-3)  # 128 = unknown
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    walls, ambiguous = (p > 0.65).sum(), ((p > 0.25) & (p <= 0.65)).sum()
    return float(entropy.mean()), float(ambiguous / max(walls, 1))


def drift_from_log(log_text):
    """Loop-closure match vs current estimate, as cartographer logs it ('differs
    by'): translation (m), rotation (rad), score (%)."""
    rows = re.findall(
        r"differs by translation ([\d.-]+) rotation ([\d.-]+) with score ([\d.]+)",
        log_text,
    )
    return np.array([[float(x) for x in r] for r in rows]).reshape(-1, 3)
