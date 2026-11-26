"""Save each layer's Cartographer map in the odom frame, on one shared grid.

Each layer_N/map is in its own layer_N_map frame, which layer_mapper places relative
to odom. This resamples every layer onto one odom-aligned grid, so the saved maps
stack cell for cell and line up with the replay's ground truth. Run it while the
mappers from layered_replay.launch.py are still up:

    ros2 run warptracer_bridge export_maps --ros-args -p output:=/maps/bridge -p maps:=/data/bridge.maps.npz

For each layer N it writes <output>_layer_N.pgm and .yaml in nav2 map_server format
(trinary, origin in odom, no rotation). With maps set, the grid is the ground truth's,
padded by margin, and it also writes <output>_layer_N_check.ppm, an overlay of the map
on the ground truth, and prints how well each layer matches.
"""
import math
import os

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

# nav2 map_server trinary thresholds and pixel values.
OCCUPIED_THRESHOLD, FREE_THRESHOLD = 65, 25
OCCUPIED_PIXEL, FREE_PIXEL, UNKNOWN_PIXEL = 0, 254, 205
UNKNOWN, FREE, OCCUPIED = -1, 0, 1
WALL_TOLERANCE = 2  # cells: an occupied cell this close to a true wall counts as on it


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def dilate(mask, cells):
    """Grow a boolean mask by a square of the given radius in cells."""
    grown = mask.copy()
    padded = np.pad(mask, cells)
    for dx in range(-cells, cells + 1):
        for dy in range(-cells, cells + 1):
            grown |= padded[cells + dx:cells + dx + mask.shape[0], cells + dy:cells + dy + mask.shape[1]]
    return grown


def resample(message, map_in_odom, origin, shape, resolution):
    """Trinary [x, y] grid on the odom-aligned canvas, nearest cell of the layer map."""
    info = message.info
    cells = np.asarray(message.data, dtype=np.int16).reshape(info.height, info.width).T  # [x, y]
    trinary = np.full(cells.shape, UNKNOWN, dtype=np.int8)
    trinary[(cells >= 0) & (cells <= FREE_THRESHOLD)] = FREE
    trinary[cells >= OCCUPIED_THRESHOLD] = OCCUPIED

    # Canvas cell centers in odom, then in the layer map frame, then in the layer map's grid.
    xs = origin[0] + (np.arange(shape[0]) + 0.5) * resolution
    ys = origin[1] + (np.arange(shape[1]) + 0.5) * resolution
    px, py = np.meshgrid(xs, ys, indexing="ij")
    tx, ty, tyaw = map_in_odom
    c, s = math.cos(tyaw), math.sin(tyaw)
    mx, my = c * (px - tx) + s * (py - ty), -s * (px - tx) + c * (py - ty)
    gyaw = yaw_of(info.origin.orientation)
    gc, gs = math.cos(gyaw), math.sin(gyaw)
    dx, dy = mx - info.origin.position.x, my - info.origin.position.y
    ix = np.floor((gc * dx + gs * dy) / info.resolution).astype(int)
    iy = np.floor((-gs * dx + gc * dy) / info.resolution).astype(int)
    inside = (ix >= 0) & (ix < info.width) & (iy >= 0) & (iy < info.height)
    out = np.full(shape, UNKNOWN, dtype=np.int8)
    out[inside] = trinary[ix[inside], iy[inside]]
    return out


def write_pgm(path, grid):
    pixels = np.full(grid.shape, UNKNOWN_PIXEL, dtype=np.uint8)
    pixels[grid == FREE] = FREE_PIXEL
    pixels[grid == OCCUPIED] = OCCUPIED_PIXEL
    image = pixels.T[::-1]  # [x, y] -> rows from high y, as map_server reads them
    with open(path, "wb") as f:
        f.write(f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode())
        f.write(image.tobytes())


def write_yaml(path, image, origin, resolution):
    with open(path, "w") as f:
        f.write(f"image: {image}\nmode: trinary\nresolution: {resolution}\n"
                f"origin: [{origin[0]:.4f}, {origin[1]:.4f}, 0]\nnegate: 0\n"
                f"occupied_thresh: {OCCUPIED_THRESHOLD / 100}\nfree_thresh: {FREE_THRESHOLD / 100}\n")


def write_check(path, grid, road, wall_near):
    """Overlay: true road light grey, mapped free green, occupied on a true wall black, off it red."""
    image = np.full(grid.shape + (3,), 255, dtype=np.uint8)
    image[road] = (215, 215, 215)
    image[grid == FREE] = (150, 220, 150)
    image[(grid == OCCUPIED) & wall_near] = (0, 0, 0)
    image[(grid == OCCUPIED) & ~wall_near] = (230, 0, 0)
    image = image.transpose(1, 0, 2)[::-1]
    with open(path, "wb") as f:
        f.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode())
        f.write(np.ascontiguousarray(image).tobytes())


class MapExporter(Node):
    def __init__(self):
        super().__init__("map_exporter")
        self.declare_parameter("layers", 2)
        self.declare_parameter("output", "/maps/bridge")
        self.declare_parameter("maps", "")
        self.declare_parameter("margin", 1.0)  # meters around the ground truth grid
        self.declare_parameter("timeout", 10.0)
        self.names = [f"layer_{n}" for n in range(int(self.get_parameter("layers").value))]
        self.output = self.get_parameter("output").value
        self.timeout = float(self.get_parameter("timeout").value)
        self.messages = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for name in self.names:
            self.create_subscription(OccupancyGrid, f"/{name}/map",
                                     lambda message, name=name: self.messages.__setitem__(name, message), latched)

    def wait(self):
        deadline = self.get_clock().now() + Duration(seconds=self.timeout)
        while rclpy.ok() and self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self.messages) == len(self.names) and all(
                    self.tf_buffer.can_transform("odom", m.header.frame_id, Time()) for m in self.messages.values()):
                return
        missing = [n for n in self.names if n not in self.messages]
        raise RuntimeError(f"No map from {missing}" if missing else "No odom -> layer map transform; "
                           "are the mappers from layered_replay.launch.py still running?")

    def map_pose(self, frame):
        t = self.tf_buffer.lookup_transform("odom", frame, Time()).transform
        return t.translation.x, t.translation.y, yaw_of(t.rotation)

    def canvas(self, poses):
        """Ground truth grid padded by margin, or the bounds of all layer maps in odom."""
        maps = self.get_parameter("maps").value
        if maps:
            with np.load(maps, allow_pickle=False) as data:
                road = data["free"].astype(bool)
                resolution = float(data["resolution"])
                origin = data["origin"].astype(np.float64)[:2]
            pad = int(round(float(self.get_parameter("margin").value) / resolution))
            road = np.pad(road, ((0, 0), (pad, pad), (pad, pad)))
            return origin - pad * resolution, road.shape[1:], resolution, road
        corners = []
        for message in self.messages.values():
            info = message.info
            tx, ty, tyaw = poses[message.header.frame_id]
            for cx in (0.0, info.width * info.resolution):
                for cy in (0.0, info.height * info.resolution):
                    x, y = info.origin.position.x + cx, info.origin.position.y + cy
                    corners.append((tx + math.cos(tyaw) * x - math.sin(tyaw) * y,
                                    ty + math.sin(tyaw) * x + math.cos(tyaw) * y))
        corners = np.array(corners)
        resolution = min(m.info.resolution for m in self.messages.values())
        low = np.floor(corners.min(axis=0) / resolution) * resolution
        shape = tuple(np.ceil((corners.max(axis=0) - low) / resolution).astype(int))
        return low, shape, resolution, None

    def export(self):
        self.wait()
        poses = {m.header.frame_id: self.map_pose(m.header.frame_id) for m in self.messages.values()}
        origin, shape, resolution, roads = self.canvas(poses)
        os.makedirs(os.path.dirname(self.output) or ".", exist_ok=True)
        stem = os.path.basename(self.output)
        for layer, name in enumerate(self.names):
            message = self.messages[name]
            grid = resample(message, poses[message.header.frame_id], origin, shape, resolution)
            write_pgm(f"{self.output}_{name}.pgm", grid)
            write_yaml(f"{self.output}_{name}.yaml", f"{stem}_{name}.pgm", origin, resolution)
            line = f"{self.output}_{name}.pgm: {shape[0]} x {shape[1]} cells at {resolution} m, origin {np.round(origin, 3).tolist()}"
            if roads is not None:
                road = roads[layer]
                wall = dilate(road, 1) & ~road
                wall_near = dilate(wall, WALL_TOLERANCE)
                occupied, free = grid == OCCUPIED, grid == FREE
                on_wall = (occupied & wall_near).sum() / max(occupied.sum(), 1)
                coverage = (free & road).sum() / max(road.sum(), 1)
                stray = (free & ~dilate(road, WALL_TOLERANCE)).sum() / max(free.sum(), 1)
                line += (f"; {100 * on_wall:.1f}% of occupied cells on a true wall, "
                         f"{100 * coverage:.1f}% of the road mapped free, {100 * stray:.1f}% of free cells off the road")
                write_check(f"{self.output}_{name}_check.ppm", grid, road, wall_near)
            self.get_logger().info(line)


def main():
    rclpy.init()
    node = MapExporter()
    try:
        node.export()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
