import numpy as np
from omegaconf import DictConfig

from features.interaction_features import InteractionStatus
from scorer.base_scorer import BaseScorer
from utils.common import EPS, get_logger
from utils.schemas import Scenario

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

    def compute_agent_weights(self, scenario: Scenario, scenario_features: dict) -> np.ndarray:
        """
        Computes the weights for each agent based on their relevance and distances to other agents.
        Args:
            scenario (dict): The scenario containing agent information.
            scenario_features (dict): The features of the scenario.
        Returns:
            np.ndarray: An array of weights for each agent.
        """
        agent_to_agent_dists = scenario_features["agent_to_agent_closest_dists"]
        relevant_agents = np.where(scenario.agent_relevance > 0.0)[0]
        relevant_agents_values = scenario.agent_relevance[relevant_agents]
        relevant_agents_dists = agent_to_agent_dists[:, relevant_agents]

        min_dist = relevant_agents_dists.min(axis=1) + EPS  # Avoid division by zero
        argmin_dist = relevant_agents_dists.argmin(axis=1)

        weights = relevant_agents_values[argmin_dist] * np.minimum(1.0 / min_dist, 1.0)
        return weights

    def compute(self, scenario: Scenario, scenario_features: dict) -> dict:
        """
        Computes the interaction scores for agents in a scenario based on their features.

        Args:
            scenario (Dict): A dictionary containing the scenario information defined in schemas.Scenario.
            scenario_features (Dict): A dictionary containing scenario feature data.

        Returns:
            Dict: A dictionary with computed scores.
        """
        # NOTE: should we avoid this overhead?
        missing_features = [feature for feature in self.features if feature not in scenario_features]
        if missing_features:
            raise ValueError(f"Missing features in scenario_features: {missing_features}")

        # TODO: make this configurable/controllable
        weights = self.compute_agent_weights(scenario, scenario_features)

        scores = np.zeros(shape=(scenario.num_agents,), dtype=np.float32)
        for n, (i, j) in enumerate(scenario_features["agent_pair_indeces"]):
            status = scenario_features["interaction_status"][n]
            if status != InteractionStatus.COMPUTED_OK:
                continue

            # Compute the agent-pair scores
            agent_pair_score = self.aggregate_simple_score(
                collision=scenario_features["collision"][n],
                mttcp=scenario_features["mttcp"][n],
            )
            scores[i] += weights[i] * agent_pair_score
            scores[j] += weights[j] * agent_pair_score

        # Normalize the scores
        # TODO: address the case where all scores are zero
        denom = max(np.where(scores > 0.0)[0].shape[0], 1)
        scene_score = np.clip(scores.sum() / denom, a_min=self.score_clip.min, a_max=self.score_clip.max)
        return {
            self.name: {
                "agent_scores": scores,
                "scene_score": scene_score,
            }
        }
