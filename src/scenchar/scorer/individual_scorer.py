import numpy as np
from omegaconf import DictConfig

from scenchar.scorer.base_scorer import BaseScorer
from scenchar.utils.common import EPS, get_logger
from scenchar.utils.schemas import Scenario, ScenarioFeatures

logger = get_logger(__name__)


class IndividualScorer(BaseScorer):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super(IndividualScorer, self).__init__(config)

    def aggregate_simple_score(self, **kwargs) -> np.ndarray:
        # Detection values are roughly obtained from: https://arxiv.org/abs/2202.07438
        speed = kwargs.get("speed", 0.0)
        acceleration = kwargs.get("acceleration", 0.0)
        deceleration = kwargs.get("deceleration", 0.0)
        jerk = kwargs.get("jerk", 0.0)
        waiting_period = kwargs.get("waiting_period", 0.0)
        return (
            min(self.detections.speed, self.weights.speed * speed)
            + min(self.detections.acceleration, self.weights.acceleration * acceleration)
            + min(self.detections.deceleration, self.weights.deceleration * deceleration)
            + min(self.detections.jerk, self.weights.jerk * jerk)
            + min(self.detections.waiting_period, self.weights.waiting_period * waiting_period)
        )

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> dict:
        """Produces a dummy output for the feature computation.

        This method should be overridden by subclasses to compute actual features.

        Args:
            scenario_features (Dict): A dictionary containing scenario feature data.

        Returns:
            Dict: A dictionary with computed scores.
        """
        # TODO: avoid these checks.
        if scenario_features.agent_to_agent_closest_dists is None:
            raise ValueError("agent_to_agent_closest_dists must not be None")
        if scenario_features.valid_idxs is None:
            raise ValueError("valid_idxs must not be None")
        if scenario_features.speed is None:
            raise ValueError("speed must not be None")
        if scenario_features.acceleration is None:
            raise ValueError("acceleration must not be None")
        if scenario_features.deceleration is None:
            raise ValueError("deceleration must not be None")
        if scenario_features.jerk is None:
            raise ValueError("jerk must not be None")
        if scenario_features.waiting_period is None:
            raise ValueError("waiting_period must not be None")

        agent_to_agent_dists = scenario_features.agent_to_agent_closest_dists

        valid_idxs = scenario_features.valid_idxs
        relevant_agents = np.where(scenario.agent_relevance > 0.0)[0]
        relevant_agents_values = scenario.agent_relevance[relevant_agents]  # Shape (num_relevant_agents, )
        relevant_agents_dists = agent_to_agent_dists[:, relevant_agents]  # Shape (num_agents, num_relevant_agents)

        # TODO: paralellize this
        N = valid_idxs.shape[0]
        scores = np.zeros(shape=(N,), dtype=np.float32)

        # An agent's contribution (weight) to the score is inversely proportional to the closest distance
        # between the agent and the relevant agents
        valid_relevant_dists = relevant_agents_dists[valid_idxs]  # Shape (num_valid_agents, num_relevant_agents)
        min_dist = valid_relevant_dists.min(axis=1)  # Shape (num_valid_agents, )
        argmin_dist = valid_relevant_dists.argmin(axis=1)

        # Broadcast relevant_agents_values to shape (num_valid_agents, num_relevant_agents)
        relevant_agents_values_broadcasted = np.broadcast_to(relevant_agents_values, valid_relevant_dists.shape)
        selected_relevant_values = relevant_agents_values_broadcasted[np.arange(N), argmin_dist]
        weights = selected_relevant_values * np.clip(1.0 / (min_dist + EPS), 0.0, 1.0)

        for n in range(N):
            scores[n] = weights[n] * self.aggregate_simple_score(
                speed=scenario_features.speed[n],
                acceleration=scenario_features.acceleration[n],
                deceleration=scenario_features.deceleration[n],
                jerk=scenario_features.jerk[n],
                waiting_period=scenario_features.waiting_period[n],
            )

        return {
            self.name: {
                "agent_scores": scores,
                "scene_score": np.clip(scores.mean(), a_min=self.score_clip.min, a_max=self.score_clip.max),
            }
        }
