import re
from abc import ABC, abstractmethod
from enum import Enum
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from characterization.schemas import FeatureDetections, FeatureWeights, Scenario, ScenarioFeatures, ScenarioScores
from characterization.utils.common import SMALL_EPS, AgentTrajectoryMasker
from characterization.utils.geometric_utils import compute_agent_to_agent_closest_dists
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


class ScoreWeightingMethod(Enum):
    """Enumeration of score weighting methods."""

    UNIFORM = "uniform"
    DISTANCE_TO_EGO_AGENT = "distance_to_ego_agent"
    DISTANCE_TO_RELEVANT_AGENTS = "distance_to_relevant_agents"


class BaseScorer(ABC):
    """Abstract base class for scenario scorers."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer, including features, detections,
                weights, and score clipping parameters.
        """
        super().__init__()
        self.config = config
        self.characterizer_type = "score"
        self.aggregated_score_weight = self.config.get("aggregated_score_weight", 0.5)
        self.features = self.config.get("features", None)
        self.detections = FeatureDetections.from_dict(config.get("detections", None))
        self.reduce_distance_penalty = self.config.get("reduce_distance_penalty", False)

        logger.info("Feature detections set to: %s", self.detections)
        self.weights = FeatureWeights.from_dict(config.get("weights", None))

        logger.info("Feature weights set to: %s", self.weights)
        self.score_clip = self.config.score_clip
        self.score_weighting_method = ScoreWeightingMethod(self.config.get("score_weighting_method", "uniform"))

        self.categorize_scores = self.config.get("categorize_scores", False)
        self.categories = None

    @property
    def name(self) -> str:
        """Returns the class name formatted as a lowercase string with underscores.

        Returns:
            str: The formatted class name.
        """
        # Get the class name and add a space before each capital letter (except the first)
        return re.sub(r"(?<!^)([A-Z])", r"_\1", self.__class__.__name__).lower()

    @staticmethod
    def _get_agent_to_agent_closest_dists(
        scenario: Scenario, scenario_features: ScenarioFeatures
    ) -> NDArray[np.float32]:
        """Retrieves or computes the agent-to-agent closest distances.

        Args:
            scenario (Scenario): Scenario object containing agent information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing precomputed distances.

        Returns:
            NDArray[np.float32]: The agent-to-agent closest distances.
        """
        agent_to_agent_dists = scenario_features.agent_to_agent_closest_dists  # Shape (num_agents, num_agents)
        if agent_to_agent_dists is None:
            agent_data = scenario.agent_data
            agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
            agent_positions = agent_trajectories.agent_xyz_pos
            agent_to_agent_dists = compute_agent_to_agent_closest_dists(agent_positions)

        return np.nan_to_num(agent_to_agent_dists, nan=np.inf)

    @staticmethod
    def _get_weights_uniform(scenario: Scenario) -> NDArray[np.float32]:
        """Returns uniform weights for all agents.

        Args:
            scenario (Scenario): Scenario object containing agent information.

        Returns:
            NDArray[np.float32]: Uniform weights for each agent.
        """
        return np.ones(scenario.agent_data.num_agents, dtype=np.float32)

    @staticmethod
    def _get_weights_wrt_ego(
        scenario: Scenario, scenario_features: ScenarioFeatures, *, reduce_distance_penalty: bool = False
    ) -> NDArray[np.float32]:
        """Computes the weights for scoring based on the scenario and features, with respect to the ego agent.

        The agent's contribution (weight) to the score is inversely proportional to the closest
        distance between the agent and the ego agent.

        Args:
            scenario (Scenario): Scenario object containing agent relevance information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing agent-to-agent closest distances.
            reduce_distance_penalty (bool): Whether to reduce the distance penalty by taking the square root.

        Returns:
            NDArray[np.float32]: The computed weights for each agent.
        """
        agent_to_agent_dists = BaseScorer._get_agent_to_agent_closest_dists(
            scenario, scenario_features
        )  # Shape (num_agents, num_agents)

        ego_agent_index = scenario.metadata.ego_vehicle_index

        # Get distance between each agent and the ego agent
        min_dist = agent_to_agent_dists[:, ego_agent_index] + SMALL_EPS  # Shape (num_agents, 1)
        if reduce_distance_penalty:
            min_dist = np.sqrt(min_dist)
        return np.minimum(1.0 / min_dist, 1.0)  # Shape (num_agents, )

    @staticmethod
    def _get_weights_wrt_relevant_agents(
        scenario: Scenario, scenario_features: ScenarioFeatures, *, reduce_distance_penalty: bool = False
    ) -> NDArray[np.float32]:
        """Computes the weights for scoring based on the scenario and features, with respect to relevant agents.

        The agent's contribution (weight) to the score is inversely proportional to the closest distance between the
        agent and the relevant agents.

        Args:
            scenario (Scenario): Scenario object containing agent relevance information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing agent-to-agent closest distances.
            reduce_distance_penalty (bool): Whether to reduce the distance penalty by taking the square root

        Returns:
            NDArray[np.float32]: The computed weights for each agent.
        """
        agent_to_agent_dists = BaseScorer._get_agent_to_agent_closest_dists(
            scenario, scenario_features
        )  # Shape (num_agents, num_agents)

        # Determine the weights of the relevant agents, if the agent relevance is not provided, use uniform weights.
        agent_relevance = scenario.agent_data.agent_relevance
        if agent_relevance is None:
            return BaseScorer._get_weights_uniform(scenario)
        relevant_agents = np.where(agent_relevance > 0.0)[0]
        relevant_agents_values = agent_relevance[relevant_agents]  # Shape (num_relevant_agents)

        # Get distance between each agent and the closest relevant agent
        relevant_agents_dists = agent_to_agent_dists[:, relevant_agents]  # Shape (num_agents, num_relevant_agents)
        min_dist = relevant_agents_dists.min(axis=1) + SMALL_EPS
        if reduce_distance_penalty:
            min_dist = np.sqrt(min_dist)

        # The final weight is the relevance of the closest agent scaled by the inverse of the distance between them.
        argmin_dist = relevant_agents_dists.argmin(axis=1)
        return relevant_agents_values[argmin_dist] * np.minimum(1.0 / min_dist, 1.0)  # Shape (num_agents, )

    def get_weights(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> NDArray[np.float32]:
        """Computes the weights for scoring based on the scenario and features.

        The agent's contribution (weight) to the score is inversely proportional to the closest
        distance between the agent and the relevant agents.

        Args:
            scenario (Scenario): Scenario object containing agent relevance information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing agent-to-agent closest distances.

        Returns:
            NDArray[np.float32]: The computed weights for each agent.
        """
        if scenario.agent_data.num_agents == 1:
            agent_relevance = scenario.agent_data.agent_relevance
            if agent_relevance is None:
                return BaseScorer._get_weights_uniform(scenario)
            return agent_relevance

        match self.score_weighting_method:
            case ScoreWeightingMethod.UNIFORM:
                return BaseScorer._get_weights_uniform(scenario)
            case ScoreWeightingMethod.DISTANCE_TO_EGO_AGENT:
                return BaseScorer._get_weights_wrt_ego(
                    scenario, scenario_features, reduce_distance_penalty=self.reduce_distance_penalty
                )
            case ScoreWeightingMethod.DISTANCE_TO_RELEVANT_AGENTS:
                return BaseScorer._get_weights_wrt_relevant_agents(
                    scenario, scenario_features, reduce_distance_penalty=self.reduce_distance_penalty
                )
            case _:
                error_message = f"Unknown score weighting method: {self.score_weighting_method}"
                raise ValueError(error_message)

    def categorize(self, score: NDArray[np.float32]) -> float:
        """Categorizes a score based on predefined percentiles.

        Args:
            score (float): The score to categorize.

        Returns:
            float: The categorized score.
        """
        if self.categories is None:
            error_message = "Categories not loaded. Cannot categorize scores."
            raise ValueError(error_message)

        ranges = list(self.categories.values())

        # If there is only one category, return 1.0 or 2.0 based on the value
        if len(ranges) < 2:  # noqa: PLR2004
            return 1.0 if score < ranges[0] else 2.0

        # If value is below the lowest range, return 1.0
        if score < ranges[0]:
            return 1.0

        # Categorize based on ranges
        for category, (lower_bound, upper_bound) in enumerate(pairwise(ranges)):
            if lower_bound <= score < upper_bound:
                return float(category + 2)  # Categories start from 2.0

        # If value is above the highest range
        return float(len(self.categories) + 1)

    @abstractmethod
    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Computes scenario-level scores from features.

        This method should be overridden by subclasses to compute actual scores.

        Args:
            scenario (Scenario): Scenario object containing scenario information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing computed features.

        Returns:
            ScenarioScores: An object containing computed scenario scores.
        """
