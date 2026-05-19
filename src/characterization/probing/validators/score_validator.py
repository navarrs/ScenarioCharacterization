"""Probe selection and interaction-based criticality scoring for counterfactual probing."""

import numpy as np

from characterization.features import interaction_utils
from characterization.probing.common import CandidateProbeResult, CriticalityMetric
from characterization.schemas.critical_probe import CriticalityResult
from characterization.schemas.scenario import ScenarioMetadata
from characterization.utils.common import EPSILON, MIN_VALID_POINTS, AgentTrajectoryMasker, InteractionAgent


def max_score_delta_validator(candidates: list[CandidateProbeResult]) -> int | None:
    """Return the index of the candidate with the highest ``pair_delta``.

    All candidates are assumed to have non-empty ``affected_ids`` (the prober filters before building the candidate
    list). Ties are broken in favour of the last-seen candidate, matching the original greedy-max loop semantics.

    Args:
        candidates: All valid per-agent probe candidates for a scenario.

    Returns:
        Index of the selected candidate, or ``None`` if the list is empty.
    """
    best_idx: int | None = None
    best_delta = -float("inf")
    for i, c in enumerate(candidates):
        if c.pair_delta >= best_delta:
            best_delta = c.pair_delta
            best_idx = i
    return best_idx


def find_criticality_timestamp(
    traj_i: AgentTrajectoryMasker,
    traj_j: AgentTrajectoryMasker,
    metadata: ScenarioMetadata,
    current_time_index: int,
) -> CriticalityResult | None:
    """Return the frame of peak criticality between two agents in the counterfactual future.

    Only frames at or after ``current_time_index`` where both agents are valid are considered. Uses minimum per-timestep
    TTC as the primary criticality metric, falling back to maximum per-timestep DRAC. Returns ``None`` if neither metric
    produces a finite result.

    Args:
        traj_i: Trajectory for agent i (probed/counterfactual) wrapped in ``AgentTrajectoryMasker``.
        traj_j: Trajectory for agent j (observed) wrapped in ``AgentTrajectoryMasker``.
        metadata: Scenario metadata containing distance and heading thresholds.
        current_time_index: First future timestep (inclusive); past frames are excluded.

    Returns:
        :class:`CriticalityResult` with the absolute frame index and the metric used, or ``None``.
    """
    num_timesteps = traj_i.agent_trajectories.shape[0]

    future_mask = np.zeros(num_timesteps, dtype=bool)
    future_mask[current_time_index:] = True
    joint_valid = (traj_i.agent_valid[:, 0] > 0) & (traj_j.agent_valid[:, 0] > 0) & future_mask
    joint_valid_indices = np.where(joint_valid)[0]

    if len(joint_valid_indices) < MIN_VALID_POINTS:
        return None

    # Build InteractionAgent objects from the jointly valid future frames.
    agent_i = InteractionAgent()
    agent_j = InteractionAgent()

    agent_i.position = traj_i.agent_xy_pos[joint_valid_indices].astype(np.float32)
    agent_j.position = traj_j.agent_xy_pos[joint_valid_indices].astype(np.float32)

    vel_i = traj_i.agent_xy_vel[joint_valid_indices]  # (T', 2)
    agent_i.speed = (np.linalg.norm(vel_i, axis=-1) + EPSILON).astype(np.float32)

    vel_j = traj_j.agent_xy_vel[joint_valid_indices]  # (T', 2)
    agent_j.speed = (np.linalg.norm(vel_j, axis=-1) + EPSILON).astype(np.float32)

    # Headings are stored in radians in the trajectory; interaction_utils expects degrees.
    agent_i.heading = np.rad2deg(traj_i.agent_headings[joint_valid_indices].squeeze(-1)).astype(np.float32)
    agent_j.heading = np.rad2deg(traj_j.agent_headings[joint_valid_indices].squeeze(-1)).astype(np.float32)

    # Length is needed for gap computation inside compute_ttc / compute_drac.
    agent_i.length = traj_i.agent_lengths[joint_valid_indices].squeeze(-1).astype(np.float32)
    agent_j.length = traj_j.agent_lengths[joint_valid_indices].squeeze(-1).astype(np.float32)

    # Spatial gate: skip if the agents are always too far apart.
    separations = interaction_utils.compute_separation(agent_i, agent_j)
    if not np.any(separations <= metadata.agent_to_agent_max_distance):
        return None

    # Primary metric: TTC (requires heading alignment to identify leader/follower).
    valid_headings = interaction_utils.find_valid_headings(agent_i, agent_j, metadata.heading_threshold)
    if len(valid_headings) >= MIN_VALID_POINTS:
        leading_agent = interaction_utils.find_leading_agent(agent_i, agent_j, valid_headings)
        ttc_per_t = interaction_utils.compute_ttc(agent_i, agent_j, leading_agent, valid_headings)
        finite_mask = np.isfinite(ttc_per_t)
        if np.any(finite_mask):
            # argmin within ttc → index in valid_headings → index in joint_valid_indices → absolute frame
            local_idx = int(np.nanargmin(ttc_per_t))
            abs_frame = int(joint_valid_indices[valid_headings[local_idx]])
            return CriticalityResult(timestamp=abs_frame, metric=CriticalityMetric.TTC)

        # Fallback: DRAC on the same heading-aligned timesteps.
        drac_per_t = interaction_utils.compute_drac(
            agent_i, agent_j, leading_agent, valid_headings, metadata.agent_max_deceleration
        )
        positive_finite = np.isfinite(drac_per_t) & (drac_per_t > 0)
        if np.any(positive_finite):
            local_idx = int(np.nanargmax(drac_per_t))
            abs_frame = int(joint_valid_indices[valid_headings[local_idx]])
            return CriticalityResult(timestamp=abs_frame, metric=CriticalityMetric.DRAC)

    return None
