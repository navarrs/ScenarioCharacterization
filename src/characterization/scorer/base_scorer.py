import re
from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig

from characterization.utils.common import EPS
from characterization.utils.schemas import Scenario, ScenarioFeatures, ScenarioScores


class BaseScorer(ABC):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer, including features, detections,
                weights, and score clipping parameters.
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
        """Returns the class name formatted as a lowercase string with underscores.

        Returns:
            str: The formatted class name.
        """
        # Get the class name and add a space before each capital letter (except the first)
        return re.sub(r"(?<!^)([A-Z])", r"_\1", self.__class__.__name__).lower()

    def get_weights(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> np.ndarray:
        """Computes the weights for scoring based on the scenario and features.

        The agent's contribution (weight) to the score is inversely proportional to the closest
        distance between the agent and the relevant agents.

        Args:
            scenario (Scenario): Scenario object containing agent relevance information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing agent-to-agent closest distances.

        Returns:
            np.ndarray: The computed weights for each agent.

        Raises:
            ValueError: If agent_to_agent_closest_dists is None in scenario_features.
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
        """Computes scenario-level scores from features.

        This method should be overridden by subclasses to compute actual scores.

        Args:
            scenario (Scenario): Scenario object containing scenario information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing computed features.

        Returns:
            ScenarioScores: An object containing computed scenario scores.
        """
        pass
