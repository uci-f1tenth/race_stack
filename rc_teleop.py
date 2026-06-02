"""RC teleop. Run on the laptop. WASD or arrow keys to drive."""

import socket
import time
import tkinter as tk

NANO_HOST = "100.111.6.29"
NANO_PORT = 5006
TICK_MS = 50
ACK_TIMEOUT = 0.5
MAX_SPEED = 1.5  # m/s
MAX_STEER = 0.4  # rad
RED = "#b00020"
GREEN = "#1b8a3a"
GRAY = "#555555"


class RcTeleop:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.keys = set()
        self.last_ack = 0.0

        self.root = tk.Tk()
        self.root.title("F1Tenth RC Teleop")
        self.root.geometry("700x500")

        self.label = tk.Label(
            self.root,
            text="DISCONNECTED",
            font=("Helvetica", 48, "bold"),
            bg=GRAY,
            fg="white",
            justify="center",
        )
        self.label.pack(fill=tk.BOTH, expand=True)
        self.root.bind("<KeyPress>", self._on_press)
        self.root.bind("<KeyRelease>", self._on_release)
        self.root.bind("<FocusOut>", lambda e: self.keys.clear())
        self.root.focus_force()
        self.tick()

    def _on_press(self, e):
        self.keys.add(e.keysym.lower())

    def _on_release(self, e):
        self.keys.discard(e.keysym.lower())

    def _command(self):
        steer = 0.0
        speed = 0.0
        if self.keys & {"w", "up"}:
            speed += MAX_SPEED
        if self.keys & {"s", "down"}:
            speed -= MAX_SPEED
        if self.keys & {"a", "left"}:
            steer += MAX_STEER
        if self.keys & {"d", "right"}:
            steer -= MAX_STEER
        return steer, speed

    def tick(self):
        self.root.after(TICK_MS, self.tick)
        steer, speed = self._command()

        try:
            self.sock.sendto(
                f"{steer:.3f},{speed:.3f}".encode(), (NANO_HOST, NANO_PORT)
            )
        except OSError:
            pass

        while True:
            try:
                self.sock.recvfrom(64)
                self.last_ack = time.monotonic()
            except (BlockingIOError, OSError):
                break

        connected = (time.monotonic() - self.last_ack) < ACK_TIMEOUT
        if not connected:
            self.label.config(text="DISCONNECTED", bg=GRAY)
        elif steer == 0.0 and speed == 0.0:
            self.label.config(text="IDLE\n\nWASD / arrows to drive", bg=RED)
        else:
            self.label.config(
                text=f"DRIVING\n\nsteer {steer:+.2f} rad\nspeed {speed:+.2f} m/s",
                bg=GREEN,
            )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    RcTeleop().run()
