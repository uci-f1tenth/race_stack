"""Decide which track layer the car is on from its pitch alone.

A climb selects the upper layer as soon as it starts. The lower layer returns only
once a descent has finished, because the car is still up top when its nose first
tips down. Height is not used: estimating it from pitch and speed drifts.
"""
import numpy as np

PITCH_THRESHOLD = 2.0  # degrees
SMOOTHING_SECONDS = 0.5


def pitch_degrees(rotations):
    """Nose-up pitch of XYZW body rotations, whose body X axis points forward."""
    x, y, z, w = rotations[..., 0], rotations[..., 1], rotations[..., 2], rotations[..., 3]
    return np.degrees(np.arcsin(np.clip(2.0 * (x * z - w * y), -1.0, 1.0)))


def trailing_average(values, count):
    """Average over the latest samples only, as a running car would have to."""
    totals = np.concatenate([[0.0], np.cumsum(values)])
    starts = np.maximum(np.arange(1, len(values) + 1) - count, 0)
    return (totals[1:] - totals[starts]) / (np.arange(1, len(values) + 1) - starts)


def layer_schedule(times, pitch, threshold=PITCH_THRESHOLD, smoothing=SMOOTHING_SECONDS):
    """Layer index, 0 or 1, for each sample of a run that starts on the lower layer."""
    period = float(np.median(np.diff(times)))
    smoothed = trailing_average(pitch, max(1, round(smoothing / period)))
    layers = np.zeros(len(pitch), dtype=int)
    layer, descending = 0, False
    for index, value in enumerate(smoothed):
        if value > threshold:
            layer, descending = 1, False
        elif value < -threshold:
            descending = True
        elif descending:
            layer, descending = 0, False
        layers[index] = layer
    return layers
