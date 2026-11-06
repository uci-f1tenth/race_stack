"""Save a layered WarpTracer recording's per-layer road grids for the replay node.

Recordings store the track's generator settings, not its grids, and the ROS
container has no WarpTracer. Run this from the WarpTracer repository root:

    uv run python <race_stack>/warptracer_bridge/tools/export_track_maps.py outputs/bridge.npz

It writes outputs/bridge.maps.npz next to the recording.
"""
import json
import sys
from pathlib import Path

import numpy as np
from warptracer.layers import LayeredTrack
from warptracer.maps import restore_track


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    recording = Path(sys.argv[1])
    with np.load(recording, allow_pickle=False) as data:
        scene = json.loads(str(data["scene_json"]))
    track = restore_track(scene["track"])
    if not isinstance(track, LayeredTrack):
        sys.exit(f"{recording} is not a layered track recording")
    output = recording.with_suffix(".maps.npz")
    layer_heights = [float(heights[free].max()) for heights, free in zip(track.heights, track.free)]
    np.savez_compressed(output, free=track.free, resolution=track.resolution,
                        origin=np.asarray(track.origin, dtype=float),
                        layer_heights=np.asarray(layer_heights))
    print(f"{output}: {track.free.shape[0]} layers of {track.free.shape[1]} x {track.free.shape[2]} cells "
          f"at {track.resolution} m; layer heights {np.round(layer_heights, 3).tolist()}")


if __name__ == "__main__":
    main()
