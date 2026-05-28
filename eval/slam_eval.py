import argparse
import csv
import math
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import Buffer, TransformException, TransformListener


def yaw_from_quat(z: float, w: float) -> float:
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SlamEvaluator(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("slam_evaluator")
        self.set_parameters([Parameter("use_sim_time", value=True)])

        self.args = args
        self.odom_buf: deque = deque(maxlen=4000)
        self.pairs: list = []

        self.create_subscription(
            Odometry, args.odom_topic, self.on_odom, qos_profile_sensor_data
        )

        if args.from_tf:
            self.tf_buf = Buffer()
            self.tf_listener = TransformListener(self.tf_buf, self)
            self.create_timer(0.05, self.poll_tf)
        else:
            self.create_subscription(
                PoseStamped,
                args.slam_pose_topic,
                self.on_pose,
                qos_profile_sensor_data,
            )

        self.create_timer(5.0, self.report_progress)
        self.get_logger().info(
            f"slam_eval up | ref={args.odom_topic} | "
            f"slam={'TF ' + args.map_frame + '->' + args.body_frame if args.from_tf else args.slam_pose_topic} | "
            f"max_dt={args.max_dt}s"
        )

    def on_odom(self, msg: Odometry) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose
        self.odom_buf.append(
            (t, p.position.x, p.position.y,
             yaw_from_quat(p.orientation.z, p.orientation.w))
        )

    def on_pose(self, msg: PoseStamped) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose
        self.try_match(
            t, p.position.x, p.position.y,
            yaw_from_quat(p.orientation.z, p.orientation.w)
        )

    def poll_tf(self) -> None:
        try:
            tr = self.tf_buf.lookup_transform(
                self.args.map_frame, self.args.body_frame, rclpy.time.Time()
            )
        except TransformException:
            return
        t = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
        self.try_match(
            t,
            tr.transform.translation.x,
            tr.transform.translation.y,
            yaw_from_quat(tr.transform.rotation.z, tr.transform.rotation.w),
        )

    def try_match(self, t_slam: float, sx: float, sy: float, syaw: float) -> None:
        if not self.odom_buf:
            return
        # Linear scan is fine: buffer is bounded and timestamps are monotonic.
        best_i, best_dt = 0, float("inf")
        for i, o in enumerate(self.odom_buf):
            dt = abs(o[0] - t_slam)
            if dt < best_dt:
                best_dt, best_i = dt, i
        if best_dt > self.args.max_dt:
            return
        ot, ox, oy, oyaw = self.odom_buf[best_i]
        self.pairs.append(
            {
                "t": t_slam,
                "ref_x": ox, "ref_y": oy, "ref_yaw": oyaw,
                "slam_x": sx, "slam_y": sy, "slam_yaw": syaw,
                "err_xy": math.hypot(sx - ox, sy - oy),
                "err_yaw": wrap(syaw - oyaw),
            }
        )

    def report_progress(self) -> None:
        if not self.pairs:
            self.get_logger().info(
                f"no matched pairs yet | odom buf={len(self.odom_buf)}"
            )
            return
        errs = np.array([p["err_xy"] for p in self.pairs])
        self.get_logger().info(
            f"pairs={len(self.pairs)} mean_err={errs.mean():.3f}m "
            f"max={errs.max():.3f}m"
        )

    def summarize(self) -> None:
        n = len(self.pairs)
        if n == 0:
            print("\nNo matched pose pairs. Check that:")
            print("  - the bag is playing with --clock")
            print("  - the SLAM is running and publishing the expected topic / TF")
            print(f"  - --max-dt is generous enough (current: {self.args.max_dt}s)")
            return
        errs_xy = np.array([p["err_xy"] for p in self.pairs])
        errs_yaw = np.array([abs(p["err_yaw"]) for p in self.pairs])
        bar = "=" * 60
        print(f"\n{bar}\nSLAM trajectory drift vs /odom over {n} matched pairs\n{bar}")
        print(
            f"  position error (m):  "
            f"mean={errs_xy.mean():.3f}  max={errs_xy.max():.3f}  "
            f"rms={np.sqrt((errs_xy ** 2).mean()):.3f}"
        )
        print(
            f"  heading error (rad): "
            f"mean={errs_yaw.mean():.3f}  max={errs_yaw.max():.3f}  "
            f"({math.degrees(errs_yaw.mean()):.1f} deg mean)"
        )
        if self.args.csv:
            with open(self.args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.pairs[0]))
                writer.writeheader()
                writer.writerows(self.pairs)
            print(f"  wrote per-pair data -> {self.args.csv}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--odom-topic", default="/odom",
                   help="Reference odometry topic (default: /odom)")
    p.add_argument("--slam-pose-topic", default="/slam_pose",
                   help="PoseStamped from SLAM (default: /slam_pose)")
    p.add_argument("--from-tf", action="store_true",
                   help="Use TF map->base_link instead of a pose topic (Cartographer)")
    p.add_argument("--map-frame", default="map")
    p.add_argument("--body-frame", default="base_link")
    p.add_argument("--max-dt", type=float, default=0.05,
                   help="Max seconds between paired SLAM/odom timestamps")
    p.add_argument("--csv", default="slam_eval.csv",
                   help="Write per-pair CSV (empty string to skip)")
    p.add_argument("--duration", type=float, default=0,
                   help="Auto-exit after N seconds (0 = until Ctrl+C)")
    args = p.parse_args()

    rclpy.init()
    node = SlamEvaluator(args)
    try:
        if args.duration > 0:
            end = time.monotonic() + args.duration
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summarize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
