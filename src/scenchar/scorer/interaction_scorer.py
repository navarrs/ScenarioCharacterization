import numpy as np
from omegaconf import DictConfig

from scenchar.features.interaction_features import InteractionStatus
from scenchar.scorer.base_scorer import BaseScorer
from scenchar.utils.common import EPS, get_logger
from scenchar.utils.schemas import Scenario, ScenarioFeatures

logger = get_logger(__name__)


class InteractionScorer(BaseScorer):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super(InteractionScorer, self).__init__(config)

    def aggregate_simple_score(self, **kwargs) -> np.ndarray:
        # Detection values are roughly obtained from: https://arxiv.org/abs/2202.07438
        collision = kwargs.get("collision", 0.0)
        mttcp = kwargs.get("mttcp", 0.0)
        return self.weights.collision * collision + min(self.detections.mttcp, self.weights.mttcp * mttcp)

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> dict:
        """
        Computes the interaction scores for agents in a scenario based on their features.

        Args:
            scenario (Dict): A dictionary containing the scenario information defined in schemas.Scenario.
            scenario_features (Dict): A dictionary containing scenario feature data.

        Returns:
            Dict: A dictionary with computed scores.
        """
        # TODO: need to avoid a lot of recomputations from the two types of features
        # TODO: avoid these checks.
        if scenario_features.agent_to_agent_closest_dists is None:
            raise ValueError("agent_to_agent_closest_dists must not be None")
        if scenario_features.interaction_agent_indices is None:
            raise ValueError("interaction_agent_indices must not be None")
        if scenario_features.interaction_status is None:
            raise ValueError("interaction_status must not be None")
        if scenario_features.collision is None:
            raise ValueError("collision must not be None")
        if scenario_features.mttcp is None:
            raise ValueError("mttcp must not be None")

        # An agent's contribution (weight) to the score is inversely proportional to the closest distance
        # between the agent and the relevant agents
        agent_to_agent_dists = scenario_features.agent_to_agent_closest_dists  # Shape (num_agents, num_agents)
        relevant_agents = np.where(scenario.agent_relevance > 0.0)[0]
        relevant_agents_values = scenario.agent_relevance[relevant_agents]  # Shape (num_relevant_agents)
        relevant_agents_dists = agent_to_agent_dists[:, relevant_agents]  # Shape (num_agents, num_relevant_agents)

        min_dist = relevant_agents_dists.min(axis=1) + EPS  # Avoid division by zero
        argmin_dist = relevant_agents_dists.argmin(axis=1)

        weights = relevant_agents_values[argmin_dist] * np.minimum(1.0 / min_dist, 1.0)  # Shape (num_agents, )

        scores = np.zeros(shape=(scenario.num_agents,), dtype=np.float32)
        interaction_agent_indices = scenario_features.interaction_agent_indices
        for n, (i, j) in enumerate(interaction_agent_indices):
            status = scenario_features.interaction_status[n]
            if status != InteractionStatus.COMPUTED_OK:
                continue

            # Compute the agent-pair scores
            agent_pair_score = self.aggregate_simple_score(
                collision=scenario_features.collision[n],
                mttcp=scenario_features.mttcp[n],
            )
            scores[i] += weights[i] * agent_pair_score
            scores[j] += weights[j] * agent_pair_score

        # Normalize the scores
        denom = max(np.where(scores > 0.0)[0].shape[0], 1)
        scene_score = np.clip(scores.sum() / denom, a_min=self.score_clip.min, a_max=self.score_clip.max)
        return {
            self.name: {
                "agent_scores": scores,
                "scene_score": scene_score,
            }
        }
