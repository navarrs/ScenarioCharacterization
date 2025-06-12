import re
from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig

from scenchar.utils.common import EPS
from scenchar.utils.schemas import Scenario, ScenarioFeatures, ScenarioScores


class BaseScorer(ABC):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super(BaseScorer, self).__init__()
        self.config = config
        self.characterizer_type = "score"
        self.features = self.config.features
        self.detections = self.config.detections
        self.weights = self.config.weights
        self.score_clip = self.config.score_clip

    @property
    def name(self) -> str:
        """Gets the class name formatted as a lowercase string with spaces.

        Returns:
            str: The formatted class name.
        """
        # Get the class name and add a space before each capital letter (except the first)
        return re.sub(r"(?<!^)([A-Z])", r"_\1", self.__class__.__name__).lower()

    def get_weights(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> np.ndarray:
        """Computes the weights for the scoring based on the scenario and features.

        Args:
            scenario (Dict): A dictionary containing the scenario information defined in schemas.Scenario.
            scenario_features (Dict): A dictionary containing scenario feature data.

        Returns:
            float: The computed weights.
        """
        if scenario_features.agent_to_agent_closest_dists is None:
            raise ValueError("agent_to_agent_closest_dists must not be None")

        # An agent's contribution (weight) to the score is inversely proportional to the closest distance
        # between the agent and the relevant agents
        agent_to_agent_dists = scenario_features.agent_to_agent_closest_dists  # Shape (num_agents, num_agents)
        relevant_agents = np.where(scenario.agent_relevance > 0.0)[0]
        relevant_agents_values = scenario.agent_relevance[relevant_agents]  # Shape (num_relevant_agents)
        relevant_agents_dists = agent_to_agent_dists[:, relevant_agents]  # Shape (num_agents, num_relevant_agents)

        min_dist = relevant_agents_dists.min(axis=1) + EPS  # Avoid division by zero
        argmin_dist = relevant_agents_dists.argmin(axis=1)

        weights = relevant_agents_values[argmin_dist] * np.minimum(1.0 / min_dist, 1.0)  # Shape (num_agents, )
        return weights

    @abstractmethod
    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Produces a dummy output for the feature computation.

        This method should be overridden by subclasses to compute actual features.

        Args:
            scenario_features (Dict): A dictionary containing scenario feature data.

        Returns:
            Dict: A dictionary with computed scores.
        """
        pass
