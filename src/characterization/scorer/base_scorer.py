import re
from abc import ABC, abstractmethod
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from characterization.schemas import FeatureDetections, FeatureWeights, Scenario, ScenarioFeatures, ScenarioScores
from characterization.utils.common import SMALL_EPS, AgentTrajectoryMasker
from characterization.utils.geometric_utils import compute_agent_to_agent_closest_dists
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


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
        self.score_wrt_ego_only = self.config.get("score_wrt_ego_only", False)

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
                return np.ones(scenario.agent_data.num_agents, dtype=np.float32)
            return agent_relevance

        agent_to_agent_dists = scenario_features.agent_to_agent_closest_dists  # Shape (num_agents, num_agents)
        if agent_to_agent_dists is None:
            agent_data = scenario.agent_data
            agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
            agent_positions = agent_trajectories.agent_xyz_pos
            agent_to_agent_dists = compute_agent_to_agent_closest_dists(agent_positions)

        if self.score_wrt_ego_only:
            relevant_agents = np.array([scenario.metadata.ego_vehicle_index])
            relevant_agents_values = np.array([1.0])  # Only the ego agent is relevant
        else:
            agent_relevance = scenario.agent_data.agent_relevance
            if agent_relevance is None:
                agent_relevance = np.ones(scenario.agent_data.num_agents, dtype=np.float32)
            relevant_agents = np.where(agent_relevance > 0.0)[0]
            relevant_agents_values = agent_relevance[relevant_agents]  # Shape (num_relevant_agents)

        # An agent's contribution (weight) to the score is inversely proportional to the closest distance between the
        # agent and the relevant agents
        agent_to_agent_dists = np.nan_to_num(agent_to_agent_dists, nan=np.inf)
        relevant_agents_dists = agent_to_agent_dists[:, relevant_agents]  # Shape (num_agents, num_relevant_agents)
        min_dist = relevant_agents_dists.min(axis=1) + SMALL_EPS  # Avoid division by zero
        if self.reduce_distance_penalty:
            min_dist = np.sqrt(min_dist)
        argmin_dist = relevant_agents_dists.argmin(axis=1)

        # weights
        return relevant_agents_values[argmin_dist] * np.minimum(1.0 / min_dist, 1.0)  # Shape (num_agents, )

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
