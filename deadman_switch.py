"""Hold the button to arm the car. Run on the laptop."""

import socket
import time
import tkinter as tk

NANO_HOST = "100.111.6.29"
NANO_PORT = 5005
HEARTBEAT_MS = 50
ACK_TIMEOUT = 0.5
RED = "#b00020"
GREEN = "#1b8a3a"
GRAY = "#555555"


class Deadman:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.held = False
        self.last_ack = 0.0

        self.root = tk.Tk()
        self.root.title("F1Tenth Deadman")
        self.root.geometry("700x500")

        self.label = tk.Label(
            self.root,
            text="DISCONNECTED",
            font=("Helvetica", 80, "bold"),
            bg=GRAY,
            fg="white",
        )
        self.label.pack(fill=tk.BOTH, expand=True)
        self.label.bind("<ButtonPress-1>", lambda e: self._set(True))
        self.label.bind("<ButtonRelease-1>", lambda e: self._set(False))
        self.label.bind("<Leave>", lambda e: self._set(False))
        self.root.bind("<FocusOut>", lambda e: self._set(False))

        self.tick()

    def _set(self, held):
        self.held = held

    def tick(self):
        self.root.after(HEARTBEAT_MS, self.tick)

        # send heartbeat: "1" if held, "0" otherwise (still pings so we know we're connected)
        try:
            self.sock.sendto(b"1" if self.held else b"0", (NANO_HOST, NANO_PORT))
        except OSError:
            pass

        # drain any echo replies
        while True:
            try:
                self.sock.recvfrom(64)
                self.last_ack = time.monotonic()
            except (BlockingIOError, OSError):
                break

        connected = (time.monotonic() - self.last_ack) < ACK_TIMEOUT
        if not connected:
            self.label.config(text="DISCONNECTED", bg=GRAY)
        elif self.held:
            self.label.config(text="ARMED", bg=GREEN)
        else:
            self.label.config(text="NOT ARMED", bg=RED)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Deadman().run()
