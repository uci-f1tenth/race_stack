#!/usr/bin/env python3
"""Deadman switch: the only path from /drive to the motors.

Controllers publish /drive as usual. This node publishes to /drive_gated (the
ackermann_mux input) only while the laptop page is held: in AUTO mode it
forwards /drive, in MANUAL mode it sends the page's own drive command and
ignores /drive. Otherwise it publishes zero-speed commands, so the car stops if
the page is released, the laptop drops off wifi, or the controller goes quiet.
If this node, the mux or the Jetson dies, commands just stop and the VESC holds
the last one until its own timeout, so keep that short in VESC Tool (App
Settings > General).

The laptop app is deadman_switch.html, served by this node: open
http://<car ip>:5005 in a browser. Every 50 ms it sends "0" (released), "1"
(held, AUTO) or "1 <speed m/s> <steer rad>" (held, MANUAL), and the node
replies "<armed 0/1> <pages connected>". Every connected page must be held, so
any one of them can stop the car. /deadman_armed is true while controllers'
/drive reaches the motors, so they can reset integrators the rest of the time.
"""

import base64
import hashlib
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Bool

port: int = 5005  # HTTP page + websocket; 8765 is taken by foxglove_bridge
deadman_timeout: float = 0.3  # s since last "held" heartbeat before disarming
drive_timeout: float = 0.15  # s, brake if armed but /drive goes quiet (< mux 0.2)
socket_timeout: float = 2.0  # s, drop a client that stops sending entirely
publish_hz: float = 50.0  # zero/manual command rate
# Manual driving is clamped here, whatever the page sends.
manual_max_speed: float = 3.0  # m/s
manual_max_reverse: float = 1.0  # m/s
manual_max_steer: float = 0.34  # rad, the old F710 teleop's steering range

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def parse_heartbeat(payload: bytes):
    """-> (held, manual (speed, steer) or None) from a page's heartbeat."""
    parts = payload.decode().split()
    held = parts[:1] == ["1"]
    if not held or len(parts) != 3:
        return held, None
    speed, steer = float(parts[1]), float(parts[2])
    if not (math.isfinite(speed) and math.isfinite(steer)):
        raise ValueError("non-finite manual command")
    speed = min(max(speed, -manual_max_reverse), manual_max_speed)
    steer = min(max(steer, -manual_max_steer), manual_max_steer)
    return held, (speed, steer)


class Heartbeats:
    """What each connected page last reported (thread-safe).

    Armed only while every connected page holds, so any one of them can stop the
    car (and a stray second page can't keep it armed after you let go).
    """

    def __init__(self):
        self.lock = threading.Lock()
        # client -> (monotonic time of last "held", -inf if released; manual
        # command or None). -inf because the monotonic clock's zero is
        # arbitrary, so 0.0 could still look fresh.
        self.clients = {}

    def update(self, client, held: bool, manual):
        with self.lock:
            self.clients[client] = (time.monotonic() if held else -math.inf, manual)

    def remove(self, client):
        with self.lock:
            self.clients.pop(client, None)

    def state(self):
        """(armed, pages connected, manual (speed, steer) while armed or None)"""
        now = time.monotonic()
        with self.lock:
            fresh = [now - t < deadman_timeout for t, _ in self.clients.values()]
            armed = bool(fresh) and all(fresh)
            manual = next((m for _, m in self.clients.values() if m), None)
            return armed, len(fresh), manual if armed else None


class DeadmanHandler(BaseHTTPRequestHandler):
    """Plain GET serves the laptop page; a websocket upgrade carries heartbeats."""

    protocol_version = "HTTP/1.1"  # browsers reject a 101 over HTTP/1.0
    timeout = socket_timeout

    def do_GET(self):
        if self.headers.get("Upgrade", "").lower() != "websocket":
            page = self.server.page
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)
            return

        # Only the page this node serves may arm the car, not some other site open
        # in the operator's browser (browsers always send Origin).
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + self.headers.get("Host", ""):
            self.send_error(403, "cross-origin websocket refused")
            return

        key = self.headers.get("Sec-WebSocket-Key", "")
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest())
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept.decode())
        self.end_headers()
        self.close_connection = True
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        beats = self.server.heartbeats
        client = object()
        self.server.logger.info(f"deadman client connected: {self.client_address[0]}")
        try:
            while True:
                opcode, payload = self.recv_frame()
                if opcode == 0x8:  # close
                    self.send_frame(0x8, payload[:2])
                    break
                if opcode == 0x9:  # ping
                    self.send_frame(0xA, payload)
                elif opcode == 0x1:  # text heartbeat
                    beats.update(client, *parse_heartbeat(payload))
                    armed, pages, _ = beats.state()
                    self.send_frame(0x1, f"{int(armed)} {pages}".encode())
        except (OSError, ValueError):
            pass  # timeout, reset, or a frame we don't speak: treat as disconnect
        finally:
            beats.remove(client)
            self.server.logger.info(
                f"deadman client disconnected: {self.client_address[0]}"
            )

    def read_exact(self, n: int) -> bytes:
        data = self.rfile.read(n)
        if len(data) < n:
            raise ConnectionError("websocket closed")
        return data

    def recv_frame(self):
        # Our client only sends 1-byte text frames, so anything unmasked or
        # long enough to need an extended length field is not ours.
        b0, b1 = self.read_exact(2)
        n = b1 & 0x7F
        if not b1 & 0x80 or n > 125:
            raise ValueError("unexpected websocket frame")
        mask = self.read_exact(4)
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(self.read_exact(n)))
        return b0 & 0x0F, payload

    def send_frame(self, opcode: int, payload: bytes):
        self.wfile.write(bytes([0x80 | opcode, len(payload)]) + payload)

    def log_message(self, format, *args):
        pass  # connects/disconnects are logged above; skip per-request noise


class DeadmanServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, page: bytes, heartbeats: Heartbeats, logger):
        super().__init__(("0.0.0.0", port), DeadmanHandler)
        self.page = page
        self.heartbeats = heartbeats
        self.logger = logger


class DeadmanSwitch(Node):
    def __init__(self):
        super().__init__("deadman_switch")
        self.heartbeats = Heartbeats()
        self.last_drive_t = -math.inf
        self.mode = None
        self.zero = AckermannDriveStamped()
        self.zero.header.frame_id = "base_link"
        self.manual_cmd = AckermannDriveStamped()
        self.manual_cmd.header.frame_id = "base_link"

        self.create_subscription(AckermannDriveStamped, "/drive", self.drive_cb, 10)
        self.gated_pub = self.create_publisher(
            AckermannDriveStamped, "/drive_gated", 10
        )
        self.armed_pub = self.create_publisher(Bool, "/deadman_armed", 10)
        self.armed_msg = Bool()
        self.create_timer(1.0 / publish_hz, self.watchdog)

        share = Path(get_package_share_directory("deadman_switch"))
        page = (share / "deadman_switch.html").read_bytes()
        self.server = DeadmanServer(page, self.heartbeats, self.get_logger())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.get_logger().info(
            f"deadman switch up: open http://<car ip>:{port} on the laptop"
        )

    def destroy_node(self):
        self.server.shutdown()
        self.server.server_close()
        super().destroy_node()

    def drive_cb(self, msg: AckermannDriveStamped):
        self.last_drive_t = time.monotonic()
        armed, _, manual = self.heartbeats.state()
        if armed and manual is None:
            self.gated_pub.publish(msg)

    def watchdog(self):
        armed, _, manual = self.heartbeats.state()
        auto = armed and manual is None
        mode = "ARMED: autonomous" if auto else "ARMED: manual" if armed else "DISARMED"
        if mode != self.mode:
            self.get_logger().info(mode)
            self.mode = mode
        self.armed_msg.data = auto
        self.armed_pub.publish(self.armed_msg)
        if manual is not None:
            cmd = self.manual_cmd
            cmd.drive.speed, cmd.drive.steering_angle = manual
        elif not auto or time.monotonic() - self.last_drive_t > drive_timeout:
            cmd = self.zero
        else:
            return  # drive_cb is forwarding the controller
        cmd.header.stamp = self.get_clock().now().to_msg()
        self.gated_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DeadmanSwitch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
