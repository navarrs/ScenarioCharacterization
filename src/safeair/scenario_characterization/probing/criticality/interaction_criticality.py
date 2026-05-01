"""Interaction-based criticality assessment for counterfactual probing."""

import numpy as np

from characterization.domains.aviation.scenario_types import AgentTrajectory
from safeair.scenario_characterization.common import ReturnCriterion
from safeair.scenario_characterization.features.interaction_utils import (
    compute_drac,
    compute_pair_gate,
    compute_ttc,
    get_joint_valid_mask,
)
from safeair.scenario_characterization.features.safeair_features import SafeAirFeatures
from safeair.scenario_characterization.scores.safeair_scorer import SafeAirScorer
from safeair.schemas.critical_probe import CriticalityMetric, CriticalityResult
from safeair.schemas.scenario import Scenario
from safeair.schemas.scenario_features import CharacterizationParameters
from safeair.schemas.scenario_scores import ScenarioScores


def compute_scenario_criticality(
    scenario: Scenario,
    extractor: SafeAirFeatures,
    scorer: SafeAirScorer,
) -> ScenarioScores:
    """Compute scenario scores using pre-instantiated extractor and scorer.

    Args:
        scenario: The scenario to score.
        extractor: Pre-instantiated SafeAirFeatures extractor.
        scorer: Pre-instantiated SafeAirScorer scorer.

    Returns:
        ScenarioScores with all score fields populated.
    """
    features = extractor.compute(scenario)
    return scorer.compute(scenario, features)


def find_criticality_timestamp(
    traj_a: AgentTrajectory,
    traj_b: AgentTrajectory,
    characterization: CharacterizationParameters,
    scale_to_m: float,
    speed_to_ms: float,
    current_time_index: int = 0,
) -> CriticalityResult | None:
    """Return the frame index of peak criticality between two agents, restricted to the future segment.

    Only timesteps at or after ``current_time_index`` are considered, so past observations cannot influence the result.
    Uses minimum per-timestep TTC as the primary metric, falling back to maximum per-timestep DRAC if agents are not
    converging at any future timestep. Returns ``None`` if neither metric produces a finite value.

    Args:
        traj_a: Single-agent trajectory accessor for agent A (shape ``(T, 10)``).
        traj_b: Single-agent trajectory accessor for agent B (shape ``(T, 10)``).
        characterization: Scenario characterization config containing separation thresholds.
        scale_to_m: Conversion factor from trajectory coordinate units to meters.
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
        max_cpa_dist=characterization.horizontal_separation_breach,
        max_vertical_separation=characterization.vertical_separation_breach,
    )

    ttc_per_t = compute_ttc(
        pos_a, pos_b, speeds_a, speeds_b, headings_a, headings_b, joint_valid_gated, ReturnCriterion.ALL
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
        characterization.agent_max_deceleration,
        ReturnCriterion.ALL,
    )
    assert isinstance(drac_per_t, np.ndarray)
    if np.any(np.isfinite(drac_per_t) & (drac_per_t > 0)):
        return CriticalityResult(int(np.nanargmax(drac_per_t)), CriticalityMetric.DRAC)

    return None
