import numpy as np
from omegaconf import DictConfig

from characterization.features.interaction_features import InteractionStatus
from characterization.scorer import INTERACTION_SCORE_FUNCTIONS
from characterization.scorer.base_scorer import BaseScorer
from characterization.utils.common import get_logger
from characterization.utils.schemas import Scenario, ScenarioFeatures, ScenarioScores

logger = get_logger(__name__)


class InteractionScorer(BaseScorer):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the InteractionScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super(InteractionScorer, self).__init__(config)

        if self.config.interaction_score_function not in INTERACTION_SCORE_FUNCTIONS:
            raise ValueError(
                f"Score function {self.config.interaction_score_function} not supported. "
                f"Supported functions are: {list(INTERACTION_SCORE_FUNCTIONS.keys())}",
            )
        self.score_function = INTERACTION_SCORE_FUNCTIONS[self.config.interaction_score_function]

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Computes interaction scores for agent pairs and a scene-level score from scenario features.

        Args:
            scenario (Scenario): Scenario object containing scenario information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing computed features.

        Returns:
            ScenarioScores: An object containing computed interaction agent-pair scores and the scene-level score.

        Raises:
            ValueError: If any required feature (agent_to_agent_closest_dists, interaction_agent_indices,
                interaction_status, collision, mttcp) is missing in scenario_features.
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
        if self.score_wrt_ego_only:
            interaction_agent_indices = [
                (i, j) for i, j in interaction_agent_indices if i == scenario.ego_index or j == scenario.ego_index
            ]
        for n, (i, j) in enumerate(interaction_agent_indices):
            status = scenario_features.interaction_status[n]
            if status != InteractionStatus.COMPUTED_OK:
                continue

            # Compute the agent-pair scores
            agent_pair_score = self.score_function(
                collision=scenario_features.collision[n],
                collision_weight=self.weights.collision,
                mttcp=scenario_features.mttcp[n],
                mttcp_weight=self.weights.mttcp,
                mttcp_detection=self.detections.mttcp,
                thw=scenario_features.thw[n],
                thw_weight=self.weights.thw,
                thw_detection=self.detections.thw,
                ttc=scenario_features.ttc[n],
                ttc_weight=self.weights.ttc,
                ttc_detection=self.detections.ttc,
                drac=scenario_features.drac[n],
                drac_weight=self.weights.drac,
                drac_detection=self.detections.drac,
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
