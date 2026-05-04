"""Interaction-based criticality assessment for AD counterfactual probing."""

import numpy as np
from numpy.typing import NDArray

from characterization.features.interaction_utils import (
    compute_drac,
    compute_pair_gate,
    compute_ttc,
    get_joint_valid_mask,
)
from characterization.schemas.critical_probe import CriticalityMetric, CriticalityResult
from characterization.utils.common import ReturnCriterion


def find_criticality_timestamp(
    traj_a: NDArray[np.float32],
    traj_b: NDArray[np.float32],
    max_distance: float,
    max_deceleration: float,
    current_time_index: int = 0,
) -> CriticalityResult | None:
    """Return the frame index of peak criticality between two AD agents, restricted to the future segment.

    The AD trajectory format is ``(T, 10)``: ``[x, y, z, length, width, height, heading_rad, vx, vy, valid]``.
    Speed is derived as ``sqrt(vx² + vy²)``. Positions and velocities are assumed to already be in metres / m/s.

    Only timesteps at or after ``current_time_index`` are considered. Uses minimum per-timestep TTC as the primary
    metric, falling back to maximum per-timestep DRAC if agents are not converging at any future timestep.

    Args:
        traj_a: Single-agent trajectory of shape ``(T, 10)`` for agent A.
        traj_b: Single-agent trajectory of shape ``(T, 10)`` for agent B.
        max_distance: Horizontal distance gate in metres (from ``scenario.metadata.agent_to_agent_max_distance``).
        max_deceleration: Maximum feasible deceleration in m/s² (from ``scenario.metadata.agent_max_deceleration``).
        current_time_index: First future timestep index (inclusive). Defaults to 0 (full trajectory).

    Returns:
        :class:`CriticalityResult` with the absolute frame index and the metric used, or ``None`` if no finite
        criticality measure was found in the future segment.
    """
    valid_a = traj_a[:, 9].astype(np.float32)
    valid_b = traj_b[:, 9].astype(np.float32)

    joint_valid = get_joint_valid_mask(valid_a[:, np.newaxis], valid_b[:, np.newaxis])
    future_mask = np.zeros(len(joint_valid), dtype=bool)
    future_mask[current_time_index:] = True
    joint_valid = joint_valid & future_mask

    if not np.any(joint_valid):
        return None

    pos_a = traj_a[:, :3].astype(np.float32)
    pos_b = traj_b[:, :3].astype(np.float32)
    speeds_a = np.sqrt(traj_a[:, 7] ** 2 + traj_a[:, 8] ** 2).astype(np.float32)
    speeds_b = np.sqrt(traj_b[:, 7] ** 2 + traj_b[:, 8] ** 2).astype(np.float32)
    headings_a = traj_a[:, 6].astype(np.float32)
    headings_b = traj_b[:, 6].astype(np.float32)

    joint_valid_gated = compute_pair_gate(
        pos_a,
        pos_b,
        speeds_a,
        speeds_b,
        headings_a,
        headings_b,
        joint_valid,
        max_cpa_dist=max_distance,
        max_vertical_separation=np.inf,
    )

    ttc_per_t = compute_ttc(
        pos_a,
        pos_b,
        speeds_a,
        speeds_b,
        headings_a,
        headings_b,
        joint_valid_gated,
        ReturnCriterion.ALL,
    )
    assert isinstance(ttc_per_t, np.ndarray)
    if np.any(np.isfinite(ttc_per_t)):
        return CriticalityResult(int(np.nanargmin(ttc_per_t)), CriticalityMetric.TTC)

    drac_per_t = compute_drac(
        pos_a,
        pos_b,
        speeds_a,
        speeds_b,
        headings_a,
        headings_b,
        joint_valid_gated,
        max_deceleration,
        ReturnCriterion.ALL,
    )
    assert isinstance(drac_per_t, np.ndarray)
    if np.any(np.isfinite(drac_per_t) & (drac_per_t > 0)):
        return CriticalityResult(int(np.nanargmax(drac_per_t)), CriticalityMetric.DRAC)

    return None
