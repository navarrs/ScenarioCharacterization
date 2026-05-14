"""Constant-velocity counterfactual probe for autonomous driving scenarios.

Replaces an agent's future trajectory (frames after ``current_time_index``) with a straight-line constant-velocity
extrapolation anchored at the last valid observed state.
"""

import numpy as np
from numpy.typing import NDArray

from characterization.probing.common import ProbeValidity
from characterization.utils.common import AgentTrajectoryMasker

_STATIONARY_SPEED_THRESHOLD = 1e-4  # m/s — below this the probe is considered invalid


def constant_velocity_probe(
    trajectory: AgentTrajectoryMasker,
    current_time_index: int,
    frequency_hz: float,
) -> tuple[NDArray[np.float32], ProbeValidity]:
    """Replace future frames with a constant-velocity extrapolation anchored at the last valid state.

    The probe is computed as follows:
    1. Find the last valid frame at or before ``current_time_index``.
    2. Compute per-step displacement from the stored (vx, vy) velocity and ``frequency_hz``.
    3. If displacement is (near) zero, return ``ProbeValidity.INVALID``.
    4. Fill all future frames by accumulating integer multiples of the displacement from the reference position. Heading
    and velocity are held constant; all future frames are marked valid.

    Args:
        trajectory: Single-agent trajectory wrapped in ``AgentTrajectoryMasker`` (underlying shape ``(T, 10)``).
            Modified on a copy — the original is not mutated.
        current_time_index: Index of the last observed frame (inclusive). Future frames start at
            ``current_time_index + 1``.
        frequency_hz: Sampling frequency in Hz, used to convert velocity to per-step displacement.

    Returns:
        ``(modified_trajectory, ProbeValidity.VALID)`` if the agent is moving, or
        ``(original_trajectory, ProbeValidity.INVALID)`` if the agent is stationary.
    """
    raw = trajectory.agent_trajectories
    num_timesteps = raw.shape[0]

    # Find the last valid observed frame at or before current_time_index.
    valid_flags = trajectory.agent_valid[: current_time_index + 1, 0]
    valid_indices = np.where(valid_flags > 0)[0]
    if len(valid_indices) == 0:
        return raw, ProbeValidity.INVALID
    ref_idx = int(valid_indices[-1])

    ref_vel = trajectory.agent_xy_vel[ref_idx]  # shape (2,): [vx, vy]
    vx, vy = float(ref_vel[0]), float(ref_vel[1])

    dt = 1.0 / frequency_hz
    dx = vx * dt
    dy = vy * dt

    if abs(dx) < _STATIONARY_SPEED_THRESHOLD * dt and abs(dy) < _STATIONARY_SPEED_THRESHOLD * dt:
        return raw, ProbeValidity.INVALID

    perturbed = raw.copy()
    ref_xyz = trajectory.agent_xyz_pos[ref_idx]  # shape (3,): [x, y, z]
    ref_heading = trajectory.agent_headings[ref_idx]  # shape (1,)

    future_start = current_time_index + 1
    for t in range(future_start, num_timesteps):
        steps = t - ref_idx
        perturbed[t][trajectory.xyz_pos_mask] = [ref_xyz[0] + steps * dx, ref_xyz[1] + steps * dy, ref_xyz[2]]
        perturbed[t][trajectory.heading_mask] = ref_heading
        perturbed[t][trajectory.xy_vel_mask] = ref_vel
        perturbed[t][trajectory.valid_mask] = 1.0

    return perturbed, ProbeValidity.VALID
