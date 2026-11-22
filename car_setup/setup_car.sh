#!/usr/bin/env bash
# Set up a car's Jetson to run race_stack. Assumes Ubuntu 22.04 (JetPack 6) with
# ROS 2 Humble already installed from apt, and race_stack cloned to
# ~/f1tenth_ws/src/race_stack. Safe to re-run: each step skips what's already
# done, and nothing is deleted.
#
#   CLIENT_PASSWORD=... HOTSPOT_PASSWORD=... ~/f1tenth_ws/src/race_stack/car_setup/setup_car.sh
#
# Wifi: joins CLIENT_SSID whenever it's in range, otherwise runs its own
# hotspot; the USB LTE dongle is only the backup internet route.
# Overridable: WS, LIDAR_IFACE, WIFI_IFACE, CLIENT_SSID, HOTSPOT_SSID,
# LTE_CONNECTION, RACE_DOMAIN_ID. Passwords stay out of git.
set -euo pipefail

WS=${WS:-$HOME/f1tenth_ws}
LIDAR_IFACE=${LIDAR_IFACE:-enP8p1s0}  # Orin Nano devkit Ethernet port
WIFI_IFACE=${WIFI_IFACE:-wlP1p1s0}
CLIENT_SSID=${CLIENT_SSID:-anetwork-guest}
HOTSPOT_SSID=${HOTSPOT_SSID:-f1t-nano}
LTE_CONNECTION=${LTE_CONNECTION:-Wired connection 1}  # the USB LTE dongle
RACE_DOMAIN_ID=${RACE_DOMAIN_ID:-37}  # any 1-101 no other car on the network uses
HERE=$(cd "$(dirname "$0")" && pwd)

step() { printf '\n== %s\n' "$*"; }
has_connection() { nmcli -t -f NAME connection show | grep -qx "$1"; }
add_line() { grep -qxF "$1" "$2" 2>/dev/null || echo "$1" >>"$2"; }

if [ -d "$WS/src/f1tenth_system" ]; then
  echo "$WS/src/f1tenth_system is still there. race_stack replaces it, and its"
  echo "vesc/ackermann_mux copies would clash with the pinned ones. Move it out"
  echo "(mv $WS/src/f1tenth_system ~/) and re-run."
  exit 1
fi

set +u  # ROS setup scripts read unset variables
# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash
set -u

step "apt: build tools, and foxglove_bridge for viewing the car from the laptop"
sudo apt-get update
sudo apt-get install -y python3-vcstool python3-colcon-common-extensions \
  python3-rosdep python3-pip ros-humble-foxglove-bridge

step "sources: pinned vesc driver and ackermann_mux (race_stack.repos)"
vcs import --skip-existing --input "$HERE/race_stack.repos" "$WS/src"

step "rosdep: every apt package the package.xml files declare"
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
rosdep update
rosdep install --from-paths "$WS/src" --ignore-src --rosdistro humble -y

step "python: torch (warporacer_node only uses it to read the checkpoint)"
python3 -c "import torch" 2>/dev/null || pip3 install --user torch

step "udev: /dev/sensors/vesc"
sudo install -m 644 "$HERE/99-vesc.rules" /etc/udev/rules.d/99-vesc.rules
sudo udevadm control --reload-rules
# The rule fires on "add", so replay that for serial devices (not the default "change").
sudo udevadm trigger --action=add --subsystem-match=tty

step "sudoers: low_voltage_shutdown may power off without a password"
sudoers=$(mktemp)
echo "$USER ALL=(root) NOPASSWD: /usr/sbin/shutdown" >"$sudoers"
sudo visudo -cf "$sudoers"
sudo install -m 440 -o root -g root "$sudoers" /etc/sudoers.d/race-stack-shutdown
rm "$sudoers"

step "network: static link to the Hokuyo (lidar at 192.168.0.10)"
if has_connection Hokuyo; then
  echo "Hokuyo connection exists, leaving it"
else
  # The /32 route keeps lidar traffic on this port even if another interface
  # (e.g. a USB LTE dongle) also lands on 192.168.0.0/24.
  sudo nmcli connection add type ethernet ifname "$LIDAR_IFACE" con-name Hokuyo \
    ipv4.method manual ipv4.addresses 192.168.0.50/24 \
    ipv4.routes "192.168.0.10/32 10 src=192.168.0.50" ipv6.method ignore
fi

# One radio can't be a client and a hotspot at once, so NetworkManager picks by
# autoconnect priority: CLIENT_SSID (200) when it's in range, else the hotspot
# (100, above any other saved client networks).
step "network: join $CLIENT_SSID when in range"
if has_connection "$CLIENT_SSID"; then
  echo "$CLIENT_SSID connection exists, leaving it"
elif [ -z "${CLIENT_PASSWORD:-}" ]; then
  echo "skipped: set CLIENT_PASSWORD to create it"
else
  # Route metric 50 puts its default route ahead of LTE (700, below).
  sudo nmcli connection add type wifi ifname "$WIFI_IFACE" con-name "$CLIENT_SSID" \
    ssid "$CLIENT_SSID" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$CLIENT_PASSWORD" \
    connection.autoconnect yes connection.autoconnect-priority 200 \
    ipv4.route-metric 50 ipv6.route-metric 50
fi

step "network: fallback wifi hotspot for the laptop (car is 10.42.0.1)"
if has_connection Hotspot; then
  echo "Hotspot connection exists, leaving it"
elif [ -z "${HOTSPOT_PASSWORD:-}" ]; then
  echo "skipped: set HOTSPOT_PASSWORD (8+ characters) to create it"
else
  sudo nmcli connection add type wifi ifname "$WIFI_IFACE" con-name Hotspot \
    autoconnect yes connection.autoconnect-priority 100 ssid "$HOTSPOT_SSID" \
    802-11-wireless.mode ap ipv4.method shared \
    wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$HOTSPOT_PASSWORD"
fi

step "network: LTE dongle only as the backup internet route"
if has_connection "$LTE_CONNECTION"; then
  # NetworkManager ranks wired (100) above wifi (600) by default, and the dongle
  # shows up as wired, so it would win over wifi without this.
  sudo nmcli connection modify "$LTE_CONNECTION" ipv4.route-metric 700 ipv6.route-metric 700
else
  echo "no '$LTE_CONNECTION' connection (set LTE_CONNECTION if the dongle's has another name)"
fi

step "bashrc: ROS environment"
add_line "source /opt/ros/humble/setup.bash" ~/.bashrc
add_line "export ROS_DOMAIN_ID=$RACE_DOMAIN_ID" ~/.bashrc

step "power mode"
# The car runs 25 W (mode 1 on an Orin Nano Super; ids differ on other Jetsons).
# Changing it can need a reboot, so this only reports.
nvpmodel -q 2>/dev/null | head -1 || true

step "build"
cd "$WS"
colcon build

cat <<EOF

Done. Still by hand:
- copy the warporacer policy to $WS/src/race_stack/warporacer_node/agent_final.pt
  and run colcon build again
- optional remote access: curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
- open a new shell (for ~/.bashrc), then follow the README
EOF
