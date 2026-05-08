import socket
import tkinter as tk

NANO_HOST = "100.111.6.29"
NANO_PORT = 5005
HEARTBEAT_MS = 50

RED = "#b00020"
GREEN = "#1b8a3a"


class Deadman:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.armed = False

        self.root = tk.Tk()
        self.root.title("F1Tenth Deadman")
        self.root.geometry("700x500")

        self.label = tk.Label(
            self.root,
            text="NOT ARMED",
            font=("Helvetica", 80, "bold"),
            bg=RED,
            fg="white",
        )
        self.label.pack(fill=tk.BOTH, expand=True)
        self.label.bind("<ButtonPress-1>", self.arm)
        self.label.bind("<ButtonRelease-1>", self.disarm)
        self.label.bind("<Leave>", self.disarm)
        self.root.bind("<FocusOut>", self.disarm)

        self.tick()

    def arm(self, _):
        self.armed = True
        self.label.config(text="ARMED", bg=GREEN)

    def disarm(self, _):
        self.armed = False
        self.label.config(text="NOT ARMED", bg=RED)

    def tick(self):
        if self.armed:
            self.sock.sendto(b"1", (NANO_HOST, NANO_PORT))
        self.root.after(HEARTBEAT_MS, self.tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Deadman().run()
