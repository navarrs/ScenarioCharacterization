"""AD-domain base scorer: configuration and weight computation methods."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from pydantic import Field

from characterization.domains.ad.scenario_types import AgentTrajectoryMasker, AgentType
from characterization.domains.ad.schemas import FeatureDetections, FeatureWeights, Scenario, ScenarioFeatures
from characterization.scorer.base_scorer import BaseScorer, BaseScorerConfig, ScoreWeightingMethod
from characterization.utils.common import ValueClipper
from characterization.utils.constants import EPSILON, SCALE_FACTOR_TO_M
from characterization.utils.geometric_utils import compute_agent_to_agent_closest_dists
from characterization.utils.logging_utils import get_pylogger

logger = get_pylogger(__name__)


class ADScorerConfig(BaseScorerConfig):
    """Configuration for AD-domain scorers.

    Extends :class:`BaseScorerConfig` with AD-specific weights, detections, and options.

    Attributes:
        weights: Per-feature score multipliers.
        detections: Per-feature detection (capping) thresholds.
        vru_priority_weight: Multiplier applied to VRU (cyclist/pedestrian) agent weights.
        reduce_distance_penalty: If True, take square root of distance before inverting (softer penalty).
        individual_score_function: Name of the individual scoring function (default ``"simple"``).
        interaction_score_function: Name of the interaction scoring function (default ``"simple"``).
        individual_categorization_file: Path to JSON percentile file for individual score categorization.
        interaction_categorization_file: Path to JSON percentile file for interaction score categorization.
    """

    weights: FeatureWeights = Field(default_factory=FeatureWeights)
    detections: FeatureDetections = Field(default_factory=FeatureDetections)
    vru_priority_weight: float = 1.0
    reduce_distance_penalty: bool = False
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
            Configured :class:`ADScorerConfig` instance.
        """
        score_clip_raw = cfg.get("score_clip", {})
        score_clip = ValueClipper(
            min=score_clip_raw.get("min", 0.0) if score_clip_raw else 0.0,
            max=score_clip_raw.get("max", 200.0) if score_clip_raw else 200.0,
        )
        raw_method = cfg.get("score_weighting_method", ScoreWeightingMethod.UNIFORM.value)
        return cls(
            weights=FeatureWeights.from_dict(cfg.get("weights", None)),
            detections=FeatureDetections.from_dict(cfg.get("detections", None)),
            score_weighting_method=ScoreWeightingMethod(raw_method),
            max_critical_distance=cfg.get("max_critical_distance", 0.5),
            aggregated_score_weight=cfg.get("aggregated_score_weight", 0.5),
            ego_pairs_only=cfg.get("ego_pairs_only", False),
            categorize_scores=cfg.get("categorize_scores", False),
            score_clip=score_clip,
            vru_priority_weight=cfg.get("vru_priority_weight", 1.0),
            reduce_distance_penalty=cfg.get("reduce_distance_penalty", False),
            individual_score_function=cfg.get("individual_score_function", "simple"),
            interaction_score_function=cfg.get("interaction_score_function", "simple"),
            individual_categorization_file=cfg.get("individual_categorization_file", ""),
            interaction_categorization_file=cfg.get("interaction_categorization_file", ""),
        )


class ADBaseScorer(BaseScorer):
    """Abstract base class for AD-domain scorers."""

    def __init__(self, config: DictConfig | ADScorerConfig | None = None) -> None:
        """Initialize with AD scorer configuration.

        Args:
            config: Configuration as an :class:`ADScorerConfig`, an OmegaConf ``DictConfig``
                (converted automatically), or ``None`` (uses defaults).
        """
        super().__init__()
        if isinstance(config, DictConfig):
            self.config: ADScorerConfig = ADScorerConfig.from_dict_config(config)
        elif config is None:
            self.config = ADScorerConfig()
        else:
            self.config = config

        logger.info(
            "class [%s] initialized with detection thresholds: %s | weights: %s",
            self.__class__.__name__,
            self.config.detections,
            self.config.weights,
        )

    @staticmethod
    def _get_agent_to_agent_closest_dists(
        scenario: Scenario,
        scenario_features: ScenarioFeatures,
    ) -> NDArray[np.float32]:
        """Retrieves or computes the agent-to-agent closest distances.

        Args:
            scenario: Scenario object containing agent information.
            scenario_features: ScenarioFeatures object containing precomputed distances.

        Returns:
            NDArray[np.float32]: The agent-to-agent closest distances.
        """
        agent_to_agent_dists = scenario_features.agent_to_agent_closest_dists
        if agent_to_agent_dists is None:
            agent_data = scenario.agent_data
            agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
            agent_positions = agent_trajectories.agent_xyz_pos
            agent_to_agent_dists = compute_agent_to_agent_closest_dists(agent_positions)

        return np.nan_to_num(agent_to_agent_dists, nan=np.inf)

    @staticmethod
    def _get_weights_wrt_ego(
        scenario: Scenario,
        scenario_features: ScenarioFeatures,
        max_critical_distance: float = 0.5,
        vru_priority_weight: float = 1.0,
        *,
        reduce_distance_penalty: bool = False,
    ) -> NDArray[np.float32]:
        """Computes weights inversely proportional to distance from the ego agent.

        Args:
            scenario: Scenario object containing agent information.
            scenario_features: ScenarioFeatures with precomputed distances.
            max_critical_distance: Minimum denominator to cap the weight.
            vru_priority_weight: Weight multiplier for vulnerable road users.
            reduce_distance_penalty: If True, take the square root of distance before inverting.

        Returns:
            NDArray[np.float32]: The computed weights for each agent.
        """
        agent_to_agent_dists = ADBaseScorer._get_agent_to_agent_closest_dists(scenario, scenario_features)

        ego_agent_index = scenario.metadata.ego_agent_index
        scale_to_m = SCALE_FACTOR_TO_M[scenario.metadata.xyz_scale]

        min_dist = (agent_to_agent_dists[:, ego_agent_index] * scale_to_m) + EPSILON
        if reduce_distance_penalty:
            min_dist = np.sqrt(min_dist)

        critical_distance = max(max_critical_distance, EPSILON)
        weights = np.minimum(1.0 / min_dist, 1.0 / critical_distance)

        agent_types = np.asarray(scenario.agent_data.agent_types)
        vru_idxs = np.where((agent_types == AgentType.TYPE_CYCLIST) | (agent_types == AgentType.TYPE_PEDESTRIAN))[0]
        weights[vru_idxs] *= vru_priority_weight
        weights[ego_agent_index] = 1.0
        return weights

    @staticmethod
    def _get_weights_wrt_relevant_agents(
        scenario: Scenario,
        scenario_features: ScenarioFeatures,
        max_critical_distance: float = 0.5,
        vru_priority_weight: float = 1.0,
        *,
        reduce_distance_penalty: bool = False,
    ) -> NDArray[np.float32]:
        """Computes weights inversely proportional to distance from the closest relevant agent.

        Args:
            scenario: Scenario object containing agent information.
            scenario_features: ScenarioFeatures with precomputed distances.
            max_critical_distance: Minimum denominator to cap the weight.
            vru_priority_weight: Weight multiplier for vulnerable road users.
            reduce_distance_penalty: If True, take the square root of distance before inverting.

        Returns:
            NDArray[np.float32]: The computed weights for each agent.
        """
        agent_to_agent_dists = ADBaseScorer._get_agent_to_agent_closest_dists(scenario, scenario_features)
        num_agents = scenario.agent_data.num_agents

        agent_relevance = scenario.agent_data.agent_relevance
        if agent_relevance is None:
            return np.ones(num_agents, dtype=np.float32)
        relevant_agents = np.where(agent_relevance > 0.0)[0]
        if len(relevant_agents) == 0:
            return np.ones(num_agents, dtype=np.float32)
        relevant_agents_values = agent_relevance[relevant_agents]

        relevant_agents_dists = agent_to_agent_dists[:, relevant_agents]
        scale_to_m = SCALE_FACTOR_TO_M[scenario.metadata.xyz_scale]
        min_dist = (relevant_agents_dists.min(axis=1) * scale_to_m) + EPSILON
        if reduce_distance_penalty:
            min_dist = np.sqrt(min_dist)

        argmin_dist = relevant_agents_dists.argmin(axis=1)
        critical_distance = max(max_critical_distance, EPSILON)
        weights = relevant_agents_values[argmin_dist] * np.minimum(1.0 / min_dist, 1.0 / critical_distance)

        agent_types = np.asarray(scenario.agent_data.agent_types)
        vru_idxs = np.where((agent_types == AgentType.TYPE_CYCLIST) | (agent_types == AgentType.TYPE_PEDESTRIAN))[0]
        weights[vru_idxs] *= vru_priority_weight
        weights[scenario.metadata.ego_agent_index] = 1.0
        return weights

    def get_weights(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> NDArray[np.float32]:
        """Dispatches to the configured weighting method.

        Args:
            scenario: Scenario object containing agent information.
            scenario_features: ScenarioFeatures with precomputed distances.

        Returns:
            NDArray[np.float32]: Per-agent weights.
        """
        num_agents = scenario.agent_data.num_agents
        if num_agents == 1:
            agent_relevance = scenario.agent_data.agent_relevance
            if agent_relevance is None:
                return self._get_weights_uniform(num_agents)
            return agent_relevance

        match self.config.score_weighting_method:
            case ScoreWeightingMethod.UNIFORM:
                return self._get_weights_uniform(num_agents)
            case ScoreWeightingMethod.DISTANCE_TO_EGO:
                return ADBaseScorer._get_weights_wrt_ego(
                    scenario,
                    scenario_features,
                    max_critical_distance=self.config.max_critical_distance,
                    vru_priority_weight=self.config.vru_priority_weight,
                    reduce_distance_penalty=self.config.reduce_distance_penalty,
                )
            case ScoreWeightingMethod.DISTANCE_TO_RELEVANT_AGENTS:
                return ADBaseScorer._get_weights_wrt_relevant_agents(
                    scenario,
                    scenario_features,
                    max_critical_distance=self.config.max_critical_distance,
                    vru_priority_weight=self.config.vru_priority_weight,
                    reduce_distance_penalty=self.config.reduce_distance_penalty,
                )
            case _:
                error_message = f"Unknown score weighting method: {self.config.score_weighting_method}"
                raise ValueError(error_message)
