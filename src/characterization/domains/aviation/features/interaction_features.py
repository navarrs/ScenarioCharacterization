"""Pairwise interaction feature extractor for aviation scenarios."""

import itertools

import numpy as np
from joblib import Parallel, delayed

from characterization.domains.aviation.features import interaction_utils
from characterization.domains.aviation.features.base_feature import AviationBaseFeature
from characterization.domains.aviation.scenario_types import AgentTrajectory
from characterization.domains.aviation.scenario_types.agent_types import (
    AgentPairType,
    get_agent_pair_type,
    raw_to_agent_type,
)
from characterization.domains.aviation.schemas.scenario import Scenario
from characterization.domains.aviation.schemas.scenario_features import (
    CharacterizationParameters,
    InteractionPairFeatures,
    ScenarioFeatures,
)
from characterization.domains.aviation.utils.scenario_characterization_utils import get_conflict_points_from_scenario
from characterization.utils.common import ReturnCriterion
from characterization.utils.constants import EPSILON

_MIN_AGENTS_FOR_PAIRS = 2
_DEFAULT_CHARACTERIZATION = CharacterizationParameters()


class InteractionFeatures(AviationBaseFeature):
    """Computes pairwise interaction features for every candidate agent pair in a scenario.

    Candidate pairs are pre-filtered through a six-stage filter pipeline. Only pairs that pass all stages are
    processed, avoiding expensive computations for distant or non-interacting pairs.

    Features computed per pair:
    - ``loss_of_separation``: 1.0 if trajectory paths intersect (with vertical separation check); otherwise the breach
      fraction.
    - ``mttcp``: minimum time-to-conflict-point difference in seconds.
    - ``thw``: minimum time headway in seconds (co-directional pairs only).
    - ``ttc``: minimum time-to-collision in seconds.
    - ``drac``: maximum deceleration rate to avoid collision in m/s².

    Unit-conversion factors are derived automatically from ``scenario.metadata`` at compute time.

    Args:
        return_criterion: Aggregation strategy.
        characterization: Thresholds controlling pair filtering and feature computation. Defaults to the standard
            aviation values.
        n_jobs: Number of parallel jobs for ``joblib.Parallel``. ``-1`` uses all CPUs.
    """

    def __init__(
        self,
        return_criterion: ReturnCriterion,
        *,
        characterization: CharacterizationParameters = _DEFAULT_CHARACTERIZATION,
        n_jobs: int = -1,
    ) -> None:
        """Intitialize with the aggregation criterion, characterization parameters, and parallelism degree."""
        super().__init__(return_criterion, characterization=characterization)
        self._n_jobs = n_jobs

    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Compute interaction features for all candidate agent pairs.

        Args:
            scenario: The scenario to characterize.

        Returns:
            ScenarioFeatures with ``interaction_features`` populated and ``individual_features`` empty.
        """
        self._set_unit_factors(scenario)
        pairs = self._get_candidate_pairs(scenario)
        results = self.compute_pairs(scenario, pairs)
        return ScenarioFeatures(
            scenario_id=scenario.metadata.scenario_id,
            individual_features=[],
            interaction_features=results,
        )

    def compute_pairs(self, scenario: Scenario, pairs: list[tuple[int, int]]) -> list[InteractionPairFeatures]:
        """Compute interaction features for a specific list of ``(index_a, index_b)`` agent pairs.

        Unit-conversion factors must already be set via a prior call to ``compute()``. The scenario's metadata must
        match the one used in that call (only trajectories may differ).

        Args:
            scenario: The scenario containing agent trajectories.
            pairs: Agent index pairs to compute.

        Returns:
            List of InteractionPairFeatures in the same order as ``pairs``.
        """
        if not pairs:
            return []
        results: list[InteractionPairFeatures] = Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(self._compute_pair)(scenario, a, b) for a, b in pairs
        )  # pyright: ignore[reportAssignmentType]
        return results

    def _get_candidate_pairs(self, scenario: Scenario) -> list[tuple[int, int]]:
        """Return index pairs whose trajectories pass a six-stage filter.

        Filters applied in order:

        1. **Distance filter**: pairs whose minimum XY distance exceeds ``agent_to_agent_max_distance`` are dropped.
        2. **FOV filter**: dropped if neither agent's heading points toward the other within ``agent_to_agent_fov_deg``
           at any joint-valid timestep. All subsequent filters operate only on FOV-valid timesteps.
        3. **Close-threshold override**: pairs where the minimum distance at any FOV-valid timestep is within
           ``horizontal_separation_breach`` are always included (remaining gates skipped).
        4. **Closing-rate filter**: dropped if the closing rate is non-positive at every FOV-valid timestep.
        5. **Reachability filter**: dropped if no FOV-valid approaching timestep satisfies
           ``dist <= closing_rate * remaining_seconds`` (i.e. TTC > remaining scenario horizon everywhere).
        6. **CPA filter**: dropped if the minimum projected closest-point-of-approach distance across all FOV-valid
           approaching timesteps exceeds ``horizontal_separation_breach``.

        Args:
            scenario: The scenario to filter.

        Returns:
            List of ``(index_a, index_b)`` tuples.
        """
        n = scenario.agent_data.num_agents
        if n < _MIN_AGENTS_FOR_PAIRS:
            return []

        agent_trajectories = AgentTrajectory(scenario.agent_data.agent_trajectories)

        xy_positions = agent_trajectories.xyz_position[..., :2] * self._scale_to_m  # (N, T, 2)
        valid = agent_trajectories.valid.astype(bool).squeeze(-1)  # (N, T)
        headings = agent_trajectories.heading.squeeze(-1)  # (N, T), radians clockwise-from-north
        speeds = agent_trajectories.speed.squeeze(-1) * self._speed_to_ms  # (N, T), m/s

        n_timesteps = valid.shape[1]
        frequency_hz = scenario.metadata.frequency_hz
        cos_fov = np.cos(np.radians(self._characterization.agent_to_agent_fov_deg))

        candidates: list[tuple[int, int]] = []
        for i, j in itertools.combinations(range(n), 2):
            joint = valid[i] & valid[j]  # (T,)
            if not np.any(joint):
                continue

            # Filter 1: hard distance cap.
            diff = xy_positions[j][joint] - xy_positions[i][joint]  # (M, 2) — joint-filtered
            dists = np.linalg.norm(diff, axis=1)  # (M,) — joint-filtered
            if np.min(dists) > self._characterization.agent_to_agent_max_distance:
                continue

            # Filter 2: FOV gate — drop if neither agent ever sees the other (before close-threshold override).
            dir_i_to_j = diff / (dists[:, None] + EPSILON)  # (M, 2) unit vectors, i→j
            heading_i = headings[i][joint]  # (M,)
            heading_j = headings[j][joint]  # (M,)
            # Order of sin, cos gives clockwise-from-north heading vectors.
            hv_i = np.stack([np.sin(heading_i), np.cos(heading_i)], axis=1)  # (M, 2)
            hv_j = np.stack([np.sin(heading_j), np.cos(heading_j)], axis=1)  # (M, 2)
            in_fov_i = np.einsum("mi,mi->m", hv_i, dir_i_to_j) >= cos_fov  # (M,) bool
            in_fov_j = np.einsum("mi,mi->m", hv_j, -dir_i_to_j) >= cos_fov  # (M,) bool
            fov_mask = in_fov_i | in_fov_j  # (M,) bool — aligned to M joint-valid timesteps
            if not np.any(fov_mask):
                continue

            # All remaining filters operate on FOV-valid timesteps only.
            # dists has shape (M,) — already joint-filtered. fov_mask has shape (M,).
            # dists[fov_mask] gives shape (F,) where F = fov_mask.sum(), guaranteed >= 1.
            dists_fov = dists[fov_mask]  # (F,)

            # Filter 3: close-threshold override — include immediately if within breach at any FOV-valid timestep.
            if np.min(dists_fov) <= self._characterization.horizontal_separation_breach:
                candidates.append((i, j))
                continue

            # Filter 4: closing-rate gate — restricted to FOV-valid timesteps.
            xy_i = xy_positions[i][joint][fov_mask]  # (F, 2)
            xy_j = xy_positions[j][joint][fov_mask]
            vx_i, vy_i = interaction_utils.velocity_components_2d(
                speeds[i][joint][fov_mask],
                headings[i][joint][fov_mask],
            )
            vx_j, vy_j = interaction_utils.velocity_components_2d(
                speeds[j][joint][fov_mask],
                headings[j][joint][fov_mask],
            )
            closing_rate = interaction_utils.closing_rate_2d(xy_i, xy_j, vx_i, vy_i, vx_j, vy_j)  # (F,)
            closing_mask = closing_rate > 0  # (F,) bool
            if not np.any(closing_mask):
                continue

            # Filter 5: direction-aware reachability — restricted to FOV-valid approaching timesteps.
            # Check dist <= closing_rate * remaining_s (i.e. TTC <= remaining scenario horizon).
            joint_idxs = np.where(joint)[0]  # (M,) absolute frame indices
            fov_idxs = joint_idxs[fov_mask]  # (F,)
            remaining_s = (n_timesteps - 1 - fov_idxs) / frequency_hz  # (F,)
            if not np.any(dists_fov[closing_mask] <= closing_rate[closing_mask] * remaining_s[closing_mask]):
                continue

            # Filter 6: projected CPA check — skip if minimum projected approach distance always exceeds breach.
            # Reuses vx_i/vy_i/vx_j/vy_j and closing_mask; F_closing = closing_mask.sum() >= 1.
            cpa_dists = interaction_utils.projected_cpa_dist_2d(
                xy_i[closing_mask],
                xy_j[closing_mask],
                vx_i[closing_mask],
                vy_i[closing_mask],
                vx_j[closing_mask],
                vy_j[closing_mask],
            )  # (F_closing,)
            if np.min(cpa_dists) > self._characterization.horizontal_separation_breach:
                continue

            candidates.append((i, j))

        return candidates

    def _compute_pair(self, scenario: Scenario, idx_a: int, idx_b: int) -> InteractionPairFeatures:
        """Compute all interaction features for a single agent pair.

        Args:
            scenario: The scenario containing the agents.
            idx_a: Index of the first agent.
            idx_b: Index of the second agent.

        Returns:
            InteractionPairFeatures. Feature values are ``None`` when not computable.
        """
        agent_id_a = int(scenario.agent_data.agent_ids[idx_a])
        agent_id_b = int(scenario.agent_data.agent_ids[idx_b])

        agent_type_a = raw_to_agent_type(scenario.agent_data.agent_types[idx_a])
        agent_type_b = raw_to_agent_type(scenario.agent_data.agent_types[idx_b])
        agent_pair_type: AgentPairType = get_agent_pair_type(agent_type_a, agent_type_b)

        agent_traj_a = AgentTrajectory(scenario.agent_data.agent_trajectories[idx_a])
        agent_traj_b = AgentTrajectory(scenario.agent_data.agent_trajectories[idx_b])

        joint_valid = interaction_utils.get_joint_valid_mask(agent_traj_a.valid, agent_traj_b.valid)

        pos_a = (agent_traj_a.xyz_position * self._scale_to_m).astype(np.float32)  # (T, 3)
        pos_b = (agent_traj_b.xyz_position * self._scale_to_m).astype(np.float32)  # (T, 3)
        speeds_a = (agent_traj_a.speed.squeeze(-1) * self._speed_to_ms).astype(np.float32)  # (T,)
        speeds_b = (agent_traj_b.speed.squeeze(-1) * self._speed_to_ms).astype(np.float32)
        headings_a = agent_traj_a.heading.squeeze(-1).astype(np.float32)  # (T,) in radians
        headings_b = agent_traj_b.heading.squeeze(-1).astype(np.float32)

        # Conflict points
        conflict_points = get_conflict_points_from_scenario(scenario, scale=self._scale_to_m)

        # Loss of separation occurs if there is a breach in either horizontal or vertical dimension.
        los_val = interaction_utils.compute_loss_of_separation(
            pos_a,
            pos_b,
            joint_valid,
            self._characterization.horizontal_separation_breach,
            self._characterization.vertical_separation_breach,
            self.return_criterion,
        )
        assert isinstance(los_val, float)
        loss_of_separation: float | None = None if np.isnan(los_val) else los_val

        # Minimum Time to Conflict point (mTTCP) is only computed if conflict point information is available.
        mttcp = None
        if conflict_points is not None:
            mttcp = interaction_utils.compute_mttcp(
                pos_a,
                pos_b,
                speeds_a,
                speeds_b,
                conflict_points,
                joint_valid,
                self._characterization.agent_to_conflict_point_max_distance,
                self.return_criterion,
            )
            assert isinstance(mttcp, float)

        # Time Headway (THW) is only computed for co-directional pairs, which are determined by their headings.
        thw = interaction_utils.compute_thw(
            pos_a,
            pos_b,
            speeds_a,
            speeds_b,
            headings_a,
            headings_b,
            joint_valid,
            self._characterization.heading_threshold,
            self.return_criterion,
        )
        assert isinstance(thw, float)

        # Pre-compute geometry gate for TTC and DRAC: suppress timesteps where paths cannot realistically converge (CPA
        # too large) or agents are vertically separated.
        joint_valid_gated = interaction_utils.compute_pair_gate(
            pos_a,
            pos_b,
            speeds_a,
            speeds_b,
            headings_a,
            headings_b,
            joint_valid,
            max_cpa_dist=self._characterization.horizontal_separation_breach,
            max_vertical_separation=self._characterization.vertical_separation_breach,
        )

        # Time to Collision (TTC) uses velocity-vector closing rate; geometry gate applied above.
        ttc = interaction_utils.compute_ttc(
            pos_a,
            pos_b,
            speeds_a,
            speeds_b,
            headings_a,
            headings_b,
            joint_valid_gated,
            self.return_criterion,
        )
        assert isinstance(ttc, float)

        # Deceleration Rate to Avoid Collision (DRAC) is computed based on the maximum deceleration value.
        drac_val = interaction_utils.compute_drac(
            pos_a,
            pos_b,
            speeds_a,
            speeds_b,
            headings_a,
            headings_b,
            joint_valid_gated,
            self._characterization.agent_max_deceleration,
            self.return_criterion,
        )
        assert isinstance(drac_val, float)
        drac: float | None = None if np.isnan(drac_val) else drac_val

        return InteractionPairFeatures(
            agent_id_a=agent_id_a,
            agent_id_b=agent_id_b,
            pair_type=agent_pair_type.name,
            loss_of_separation=loss_of_separation,
            mttcp=mttcp,
            thw=thw,
            ttc=ttc,
            drac=drac,
        )
