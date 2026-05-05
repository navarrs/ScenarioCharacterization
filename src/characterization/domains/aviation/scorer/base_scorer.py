"""Aviation-domain base scorer: configuration and weight computation."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from pydantic import Field

from characterization.domains.aviation.scenario_types import AgentTrajectory
from characterization.domains.aviation.schemas.scenario import Scenario
from characterization.domains.aviation.schemas.scenario_features import (
    FeatureDetections,
    FeatureWeights,
)
from characterization.domains.aviation.schemas.scenario_features import (
    ScenarioFeatures as AviationScenarioFeatures,
)
from characterization.scorer.base_scorer import BaseScorer, BaseScorerConfig, ScoreWeightingMethod
from characterization.utils.common import ValueClipper
from characterization.utils.constants import EPSILON, SCALE_FACTOR_TO_M


class AviationScorerConfig(BaseScorerConfig):
    """Configuration for aviation-domain scorers.

    Extends :class:`BaseScorerConfig` with aviation-specific weights, detections, and options.

    Attributes:
        weights: Per-feature score multipliers.
        detections: Per-feature detection (capping) thresholds.
        individual_score_function: Name of the individual scoring function (default ``"simple"``).
        interaction_score_function: Name of the interaction scoring function (default ``"simple"``).
        individual_categorization_file: Path to JSON percentile file for individual score categorization.
        interaction_categorization_file: Path to JSON percentile file for interaction score categorization.
    """

    weights: FeatureWeights = Field(default_factory=FeatureWeights)
    detections: FeatureDetections = Field(default_factory=FeatureDetections)
    score_weighting_method: ScoreWeightingMethod = ScoreWeightingMethod.DISTANCE_TO_EGO
    ego_pairs_only: bool = True
    individual_score_function: str = "simple"
    interaction_score_function: str = "simple"
    individual_categorization_file: str = ""
    interaction_categorization_file: str = ""

    @classmethod
    def from_dict_config(cls, cfg: DictConfig) -> "Self":
        """Construct from an OmegaConf DictConfig.

        Args:
            cfg: Hydra/OmegaConf configuration dict.

        Returns:
            Configured :class:`AviationScorerConfig` instance.
        """
        score_clip_raw = cfg.get("score_clip", {})
        score_clip = ValueClipper(
            min=score_clip_raw.get("min", 0.0) if score_clip_raw else 0.0,
            max=score_clip_raw.get("max", 200.0) if score_clip_raw else 200.0,
        )
        raw_method = cfg.get("score_weighting_method", ScoreWeightingMethod.DISTANCE_TO_EGO.value)
        return cls(
            weights=FeatureWeights.model_validate(dict(cfg.get("weights", {}))),
            detections=FeatureDetections.model_validate(dict(cfg.get("detections", {}))),
            score_weighting_method=ScoreWeightingMethod(raw_method),
            max_critical_distance=cfg.get("max_critical_distance", 0.5),
            aggregated_score_weight=cfg.get("aggregated_score_weight", 0.5),
            ego_pairs_only=cfg.get("ego_pairs_only", True),
            categorize_scores=cfg.get("categorize_scores", False),
            score_clip=score_clip,
            individual_score_function=cfg.get("individual_score_function", "simple"),
            interaction_score_function=cfg.get("interaction_score_function", "simple"),
            individual_categorization_file=cfg.get("individual_categorization_file", ""),
            interaction_categorization_file=cfg.get("interaction_categorization_file", ""),
        )


ScorerConfig = AviationScorerConfig


class AviationBaseScorer(BaseScorer):
    """Abstract base class for aviation-domain scorers."""

    def __init__(self, config: DictConfig | AviationScorerConfig | None = None) -> None:
        """Initialize with aviation scorer configuration.

        Args:
            config: Configuration as an :class:`AviationScorerConfig`, an OmegaConf ``DictConfig``
                (converted automatically), or ``None`` (uses defaults).
        """
        super().__init__()
        if isinstance(config, DictConfig):
            self.config: AviationScorerConfig = AviationScorerConfig.from_dict_config(config)
        elif config is None:
            self.config = AviationScorerConfig()
        else:
            self.config = config

    def parse_scenario_features(self, data: dict[str, object]) -> object:
        """Parse and validate a feature dict using the aviation ScenarioFeatures schema.

        Args:
            data: Raw dict produced by aviation ``ScenarioFeatures.model_dump()``.

        Returns:
            A validated aviation ``ScenarioFeatures`` instance.
        """
        return AviationScenarioFeatures.model_validate(data)

    def _compute_agent_weights(self, scenario: Scenario) -> NDArray[np.float32]:
        """Compute per-agent weights based on the configured weighting method.

        For ``UNIFORM``, all agents receive weight 1.0.

        For ``DISTANCE_TO_EGO``, each agent's weight is ``min(1/d, 1/max_critical_distance)`` where ``d`` is the
        minimum Euclidean distance (in metres) to the ego agent over all joint-valid timesteps.
        Agents with no joint-valid overlap with the ego receive weight 0.

        ``DISTANCE_TO_RELEVANT_AGENTS`` is not yet implemented for aviation.

        Args:
            scenario: The scenario from which to derive weights.

        Returns:
            Shape ``(N,)`` float32 array of per-agent weights.
        """
        n = scenario.agent_data.num_agents
        if self.config.score_weighting_method == ScoreWeightingMethod.UNIFORM or n == 1:
            return np.ones(n, dtype=np.float32)

        if self.config.score_weighting_method == ScoreWeightingMethod.DISTANCE_TO_RELEVANT_AGENTS:
            error_message = "DISTANCE_TO_RELEVANT_AGENTS is not yet implemented for aviation."
            raise NotImplementedError(error_message)

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
