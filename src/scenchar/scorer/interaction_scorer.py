import numpy as np
from omegaconf import DictConfig

from scenchar.features.interaction_features import InteractionStatus
from scenchar.scorer.base_scorer import BaseScorer
from scenchar.utils.common import get_logger
from scenchar.utils.schemas import Scenario, ScenarioFeatures, ScenarioScores

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

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
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

        # Get the agent weights
        weights = self.get_weights(scenario, scenario_features)
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
        return ScenarioScores(
            scenario_id=scenario.scenario_id,
            num_agents=scenario.num_agents,
            interaction_agent_scores=scores,
            interaction_scene_score=scene_score,
        )
