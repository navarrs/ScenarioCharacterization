"""Interaction-based criticality assessment for aviation counterfactual probing."""

import numpy as np

from characterization.domains.aviation.scenario_types import AgentTrajectory
from characterization.features.interaction_utils import (
    compute_drac,
    compute_pair_gate,
    compute_ttc,
    get_joint_valid_mask,
)
from characterization.schemas.critical_probe import CriticalityMetric, CriticalityResult
from characterization.utils.common import ReturnCriterion


def find_criticality_timestamp(
    traj_a: AgentTrajectory,
    traj_b: AgentTrajectory,
    horizontal_separation_breach: float,
    vertical_separation_breach: float,
    agent_max_deceleration: float,
    scale_to_m: float,
    speed_to_ms: float,
    current_time_index: int = 0,
) -> CriticalityResult | None:
    """Return the frame index of peak criticality between two agents, restricted to the future segment.

    Only timesteps at or after ``current_time_index`` are considered. Uses minimum per-timestep TTC as the primary
    metric, falling back to maximum per-timestep DRAC if agents are not converging at any future timestep.
    Returns ``None`` if neither metric produces a finite value.

    Args:
        traj_a: Single-agent trajectory accessor for agent A, shape ``(T, 10)``.
        traj_b: Single-agent trajectory accessor for agent B, shape ``(T, 10)``.
        horizontal_separation_breach: Horizontal separation threshold in metres.
        vertical_separation_breach: Vertical separation threshold in metres.
        agent_max_deceleration: Maximum feasible deceleration in m/s².
        scale_to_m: Conversion factor from trajectory coordinate units to metres.
        speed_to_ms: Conversion factor from stored speed units to m/s.
        current_time_index: First future timestep index (inclusive). Defaults to 0 (full trajectory).

    Returns:
        :class:`CriticalityResult` with the absolute frame index and the metric used, or ``None`` if no finite
        criticality measure was found in the future segment.
    """
    joint_valid = get_joint_valid_mask(traj_a.valid, traj_b.valid)
    future_mask = np.zeros(len(joint_valid), dtype=bool)
    future_mask[current_time_index:] = True
    joint_valid = joint_valid & future_mask

    if not np.any(joint_valid):
        return None

    pos_a = (traj_a.xyz_position * scale_to_m).astype(np.float32)
    pos_b = (traj_b.xyz_position * scale_to_m).astype(np.float32)
    speeds_a = (traj_a.speed.squeeze(-1) * speed_to_ms).astype(np.float32)
    speeds_b = (traj_b.speed.squeeze(-1) * speed_to_ms).astype(np.float32)
    headings_a = traj_a.heading.squeeze(-1).astype(np.float32)
    headings_b = traj_b.heading.squeeze(-1).astype(np.float32)

    joint_valid_gated = compute_pair_gate(
        pos_a,
        pos_b,
        speeds_a,
        speeds_b,
        headings_a,
        headings_b,
        joint_valid,
        max_cpa_dist=horizontal_separation_breach,
        max_vertical_separation=vertical_separation_breach,
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
        agent_max_deceleration,
        ReturnCriterion.ALL,
    )
    assert isinstance(drac_per_t, np.ndarray)
    if np.any(np.isfinite(drac_per_t) & (drac_per_t > 0)):
        return CriticalityResult(int(np.nanargmax(drac_per_t)), CriticalityMetric.DRAC)

    return None
