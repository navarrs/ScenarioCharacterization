"""Constant-velocity counterfactual probe for the AD (autonomous driving) domain."""

import numpy as np
from numpy.typing import NDArray

from characterization.probing.common import ProbeValidity


def constant_velocity_probe(
    trajectory: NDArray[np.float32],
    reference_frame_index: int,
    frequency_hz: float,
) -> tuple[NDArray[np.float32], ProbeValidity]:
    """Replace the future portion of a single-agent trajectory with a constant-velocity prediction.

    The AD trajectory format is ``(T, 10)``: ``[x, y, z, length, width, height, heading_rad, vx, vy, valid]``.
    The velocity components ``vx`` and ``vy`` at the last valid frame before ``reference_frame_index`` are used to
    extrapolate ``x`` and ``y`` for all future frames. All extrapolated frames are marked valid.

    Args:
        trajectory: Single-agent trajectory array of shape ``(T, 10)``, already in metres / m/s.
        reference_frame_index: Index of the reference frame; future starts at ``reference_frame_index + 1``.
        frequency_hz: Trajectory sampling frequency in Hz.

    Returns:
        New ``(T, 10)`` float32 array with the future portion replaced by constant-velocity extrapolation, and the
        probe validity status.
    """
    valid = trajectory[: reference_frame_index + 1, 9].astype(bool)
    valid_idxs = np.where(valid)[0]
    ref_idx = int(valid_idxs[-1]) if len(valid_idxs) > 0 else reference_frame_index

    vx = float(trajectory[ref_idx, 7])
    vy = float(trajectory[ref_idx, 8])
    dt = 1.0 / frequency_hz
    dx = vx * dt
    dy = vy * dt

    if dx == 0.0 and dy == 0.0:
        return trajectory, ProbeValidity.INVALID

    result = trajectory.copy()
    future = slice(reference_frame_index + 1, None)
    result[future] = result[ref_idx]

    steps = np.arange(reference_frame_index + 1 - ref_idx, trajectory.shape[0] - ref_idx, dtype=np.float32)
    result[future, 0] += steps * dx  # x
    result[future, 1] += steps * dy  # y
    result[future, 9] = 1.0  # mark valid

    return result, ProbeValidity.VALID
