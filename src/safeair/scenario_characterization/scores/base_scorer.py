"""Abstract base class and configuration for scenario scorers."""

from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from characterization.domains.aviation.scenario_types import AgentTrajectory
from characterization.utils.constants import EPSILON, SCALE_FACTOR_TO_M
from safeair.scenario_characterization.common import ValueClipper
from safeair.schemas.scenario import Scenario
from safeair.schemas.scenario_features import FeatureDetections, FeatureWeights, ScenarioFeatures
from safeair.schemas.scenario_scores import ScenarioScores


class ScoreWeightingMethod(StrEnum):
    """How to weight per-agent contributions to the scene-level score.

    Attributes:
        UNIFORM: All agents contribute equally (weight = 1.0).
        DISTANCE_TO_EGO: Agents closer to the ego agent contribute more; weight is inversely proportional to the
            minimum distance to the ego agent over all joint-valid timesteps.
    """

    UNIFORM = "uniform"
    DISTANCE_TO_EGO = "distance_to_ego"


class ScorerConfig(BaseModel):
    """Configuration for a scenario scorer.

    Attributes:
        weights: Per-feature multipliers.
        detections: Per-feature detection (capping) thresholds.
        score_clip: Min/max bounds for the final scene-level score.
        score_weighting_method: How per-agent contributions are weighted when aggregating to a scene score.
        max_critical_distance: Minimum denominator (in meters) when computing distance-based weights, preventing
            extreme weights for agents very close to the ego.
        aggregated_score_weight: Weight ``w`` given to the individual scene score in the combined score
            ``w * individual + (1 - w) * interaction``.
        ego_pairs_only: If True, only score interaction pairs where one of the agents is the ego agent.
    """

    weights: FeatureWeights = FeatureWeights()
    detections: FeatureDetections = FeatureDetections()
    score_clip: ValueClipper = ValueClipper()
    score_weighting_method: ScoreWeightingMethod = ScoreWeightingMethod.DISTANCE_TO_EGO
    max_critical_distance: float = 0.5
    aggregated_score_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    ego_pairs_only: bool = True


class BaseScorer(ABC):
    """Abstract base class for scenario scorers.

    Args:
        config: Scorer configuration. Defaults to ``ScorerConfig()`` if not provided.
    """

    def __init__(self, config: ScorerConfig | None = None) -> None:
        """Store configuration, defaulting to ``ScorerConfig()`` if none is provided."""
        self.config = config or ScorerConfig()

    def _compute_agent_weights(self, scenario: Scenario) -> NDArray[np.float32]:
        """Compute per-agent weights based on the configured weighting method.

            For ``UNIFORM``, all agents receive weight 1.0.

            For ``DISTANCE_TO_EGO``, each agent's weight is ``min(1/d, 1/max_critical_distance)`` where ``d`` is the
            minimum Euclidean distance to the ego agent over all joint-valid timesteps. Agents with no joint-valid
            overlap with the ego receive weight 0.

        Args:
            scenario: The scenario from which to derive weights.

        Returns:
            Shape ``(N,)`` float32 array of per-agent weights.
        """
        n = scenario.agent_data.num_agents
        if self.config.score_weighting_method == ScoreWeightingMethod.UNIFORM or n == 1:
            return np.ones(n, dtype=np.float32)

        scale_to_m = SCALE_FACTOR_TO_M[scenario.metadata.xyz_scale]
        agent_trajectories = AgentTrajectory(scenario.agent_data.agent_trajectories)
        positions = agent_trajectories.xyz_position * scale_to_m  # (N, T, 3)
        valid = agent_trajectories.valid.squeeze(-1).astype(bool)  # (N, T)

        ego_idx = scenario.metadata.ego_agent_index
        ego_pos = positions[ego_idx]  # (T, 3)
        ego_valid = valid[ego_idx]  # (T,)

        critical = max(self.config.max_critical_distance, EPSILON)
        weights = np.ones(n, dtype=np.float32)
        for i in range(n):
            if i == ego_idx:
                continue
            joint = ego_valid & valid[i]
            if not np.any(joint):
                weights[i] = 0.0
                continue
            dists = np.linalg.norm(positions[i][joint] - ego_pos[joint], axis=1)
            min_dist = float(np.min(dists)) + EPSILON
            weights[i] = min(1.0 / min_dist, 1.0 / critical)
        return weights

    @abstractmethod
    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Compute scenario-level scores from pre-computed features.

        Args:
            scenario: The scenario to score.
            scenario_features: Pre-computed features for the scenario.

        Returns:
            ScenarioScores with populated score fields.
        """
        ...
