#!/usr/bin/env python3
"""Deadman switch: the only path from /drive to the motors.

Controllers publish /drive as usual. This node republishes it on /drive_gated
(the ackermann_mux input) only while a laptop is holding the arm button over a
websocket; otherwise it publishes zero-speed commands. So the car stops if the
button is released, the laptop drops off wifi, or the controller goes quiet. If
this node dies, nothing reaches /drive_gated and the mux stops forwarding.

The laptop app is deadman_switch.html, served by this node: open
http://<car ip>:5005 in a browser. It sends "1" (held) or "0" every 50 ms and
the node replies with the car's armed state.
"""

import base64
import hashlib
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

port: int = 5005  # HTTP page + websocket; 8765 is taken by foxglove_bridge
deadman_timeout: float = 0.3  # s since last "held" heartbeat before disarming
drive_timeout: float = 0.15  # s, brake if armed but /drive goes quiet (< mux 0.2)
socket_timeout: float = 2.0  # s, drop a client that stops sending entirely
publish_hz: float = 50.0  # zero-command rate while disarmed

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class Heartbeats:
    """When each connected client last reported the button held (thread-safe)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.held = {}

    def update(self, client, held: bool):
        with self.lock:
            if held:
                self.held[client] = time.monotonic()
            else:
                self.held.pop(client, None)

    def armed(self) -> bool:
        now = time.monotonic()
        with self.lock:
            return any(now - t < deadman_timeout for t in self.held.values())


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
                    beats.update(client, payload == b"1")
                    self.send_frame(0x1, b"1" if beats.armed() else b"0")
        except (OSError, ValueError):
            pass  # timeout, reset, or a frame we don't speak: treat as disconnect
        finally:
            beats.update(client, False)
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
        self.last_drive_t = 0.0
        self.was_armed = False
        self.zero = AckermannDriveStamped()
        self.zero.header.frame_id = "base_link"

        self.create_subscription(AckermannDriveStamped, "/drive", self.drive_cb, 10)
        self.gated_pub = self.create_publisher(
            AckermannDriveStamped, "/drive_gated", 10
        )
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
        if self.heartbeats.armed():
            self.gated_pub.publish(msg)

    def watchdog(self):
        armed = self.heartbeats.armed()
        if armed != self.was_armed:
            self.get_logger().info("ARMED" if armed else "DISARMED")
            self.was_armed = armed
        if not armed or time.monotonic() - self.last_drive_t > drive_timeout:
            self.zero.header.stamp = self.get_clock().now().to_msg()
            self.gated_pub.publish(self.zero)


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
