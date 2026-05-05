"""Constant-velocity counterfactual probe."""

import numpy as np
from numpy.typing import NDArray

from safeair.scenario_characterization.common import ProbeValidity
from safeair.utils.constants import SCALE_FACTOR_TO_M, SPEED_TO_MS, SpeedUnits, XYZScale
from safeair.utils.scenario_types import AgentTrajectory


def constant_velocity_probe(
    agent_trajectory: AgentTrajectory,
    reference_frame_index: int,
    frequency_hz: float,
    xyz_scale: XYZScale = XYZScale.M,
    speed_units: SpeedUnits = SpeedUnits.MS,
) -> tuple[NDArray[np.float32], ProbeValidity]:
    """Replace the future portion of a single-agent trajectory with a constant-velocity prediction.

    The last valid observation at or before ``reference_frame_index`` is used as the reference state.  Future frames
    (``reference_frame_index + 1`` onward) are extrapolated using the agent's heading and speed at that reference frame.
    Altitude is extrapolated at the rate estimated from the two last valid frames. Lat/lon and range/bearing are filled
    from the reference frame; geodesic updates are left as a TODO. All future frames are marked valid.

    Args:
        agent_trajectory: Single-agent trajectory accessor wrapping a ``(T, 10)`` array.
        reference_frame_index: Index of the reference frame; future starts at ``reference_frame_index + 1``.
        frequency_hz: Trajectory sampling frequency in Hz.
        xyz_scale: Coordinate unit of the x/y position fields.
        speed_units: Unit of the speed field.

    Returns:
        New ``(T, 10)`` float32 array with the future portion replaced by constant-velocity extrapolation.
    """
    trajectory = agent_trajectory.agent_trajectory.copy()
    num_timesteps = trajectory.shape[0]

    # Find the last valid observation at or before reference_frame_index
    valid = agent_trajectory.valid.squeeze(-1)[: reference_frame_index + 1].astype(bool)
    valid_idxs = np.where(valid)[0]
    ref_idx = int(valid_idxs[-1]) if len(valid_idxs) > 0 else reference_frame_index
    prev_idx = int(valid_idxs[-2]) if len(valid_idxs) > 1 else ref_idx

    # Single conversion factor: speed_units → position_units per second (e.g. knots → m/s if speed is in knots and
    # position is in meters)
    speed_raw = float(agent_trajectory.speed.squeeze(-1)[ref_idx])
    speed_factor = speed_raw * (SPEED_TO_MS[speed_units] / SCALE_FACTOR_TO_M[xyz_scale])

    # Calculate the x/y displacement per timestep from the heading and speed at the reference frame
    heading = float(agent_trajectory.heading.squeeze(-1)[ref_idx])
    dt = 1.0 / frequency_hz
    dx = speed_factor * np.cos(heading) * dt
    dy = speed_factor * np.sin(heading) * dt

    # Altitude rate estimated from the two last valid frames
    idx_diff = ref_idx - prev_idx
    z_ref = agent_trajectory.altitude.squeeze(-1)[ref_idx]
    z_prev = agent_trajectory.altitude.squeeze(-1)[prev_idx]
    dz = (z_ref - z_prev) / idx_diff if idx_diff > 0 else np.zeros_like(z_ref)

    if dx == 0.0 and dy == 0.0 and float(dz) == 0.0:
        return trajectory, ProbeValidity.INVALID

    # Fill all future frames from the reference state
    future = slice(reference_frame_index + 1, None)
    trajectory[future] = trajectory[ref_idx]

    steps = np.arange(reference_frame_index + 1 - ref_idx, num_timesteps - ref_idx)  # (n_future, )
    xyz_mask = agent_trajectory.xyz_position_mask
    trajectory[future, xyz_mask] += steps[:, np.newaxis] * np.array([dx, dy, dz])
    trajectory[future, agent_trajectory.heading_mask] = heading
    trajectory[future, agent_trajectory.speed_mask] = speed_raw

    # TODO: update lat/lon and range/bearing using geodesic calculations based on the new x/y/z position.

    # Make all extrapolated future frames valid
    trajectory[future, agent_trajectory.valid_mask] = 1.0
    return trajectory, ProbeValidity.VALID
