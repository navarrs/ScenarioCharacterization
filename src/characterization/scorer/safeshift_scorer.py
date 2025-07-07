import numpy as np
from omegaconf import DictConfig

from characterization.features.interaction_features import InteractionStatus
from characterization.scorer import (
    INDIVIDUAL_SCORE_FUNCTIONS,
    INTERACTION_SCORE_FUNCTIONS,
)
from characterization.scorer.base_scorer import BaseScorer
from characterization.utils.common import get_logger
from characterization.utils.schemas import Scenario, ScenarioFeatures, ScenarioScores

logger = get_logger(__name__)


class SafeShiftScorer(BaseScorer):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the SafeShiftScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super(SafeShiftScorer, self).__init__(config)

        if self.config.individual_score_function not in INDIVIDUAL_SCORE_FUNCTIONS:
            raise ValueError(
                f"Score function {self.config.individual_score_function} not supported. "
                f"Supported functions are: {list(INDIVIDUAL_SCORE_FUNCTIONS.keys())}"
            )
        self.individual_score_function = INDIVIDUAL_SCORE_FUNCTIONS[self.config.individual_score_function]

        if self.config.interaction_score_function not in INTERACTION_SCORE_FUNCTIONS:
            raise ValueError(
                f"Score function {self.config.interaction_score_function} not supported. "
                f"Supported functions are: {list(INTERACTION_SCORE_FUNCTIONS.keys())}"
            )
        self.interaction_score_function = INTERACTION_SCORE_FUNCTIONS[self.config.interaction_score_function]

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

        # Compute the individual scores
        scores_ind = np.zeros(shape=(scenario.num_agents,), dtype=np.float32)
        valid_idxs = scenario_features.valid_idxs
        N = valid_idxs.shape[0]
        for n in range(N):
            # TODO: fix this indexing issue.
            valid_idx = valid_idxs[n]
            scores_ind[valid_idx] = weights[valid_idx] * self.individual_score_function(
                speed=scenario_features.speed[n],
                speed_weight=self.weights.speed,
                speed_detection=self.detections.speed,
                acceleration=scenario_features.acceleration[n],
                acceleration_weight=self.weights.acceleration,
                acceleration_detection=self.detections.acceleration,
                deceleration=scenario_features.deceleration[n],
                deceleration_weight=self.weights.deceleration,
                deceleration_detection=self.detections.deceleration,
                jerk=scenario_features.jerk[n],
                jerk_weight=self.weights.jerk,
                jerk_detection=self.detections.jerk,
                waiting_period=scenario_features.waiting_period[n],
                waiting_period_weight=self.weights.waiting_period,
                waiting_period_detection=self.detections.waiting_period,
            )
        denom_ind = max(np.where(scores_ind > 0.0)[0].shape[0], 1)

        # Compute the interaction scores
        scores_int = np.zeros(shape=(scenario.num_agents,), dtype=np.float32)
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
            agent_pair_score = self.interaction_score_function(
                collision=scenario_features.collision[n],
                collision_weight=self.weights.collision,
                mttcp=scenario_features.mttcp[n],
                mttcp_weight=self.weights.mttcp,
                mttcp_detection=self.detections.mttcp,
                ttc=scenario_features.ttc[n],
                ttc_weight=self.weights.ttc,
                ttc_detection=self.detections.ttc,
                drac=scenario_features.drac[n],
                drac_weight=self.weights.drac,
                drac_detection=self.detections.drac,
            )
            scores_int[i] += weights[i] * agent_pair_score
            scores_int[j] += weights[j] * agent_pair_score
        denom_int = max(np.where(scores_int > 0.0)[0].shape[0], 1)

        # Normalize the scores
        # scores = scores_ind + scores_int
        # scene_score = (scores_ind.sum() / denom_ind) + (scores_int.sum() / denom_int)
        # scene_score = np.clip(scene_score, a_min=self.score_clip.min, a_max=self.score_clip.max)
        scores = scores_ind.copy() + scores_int.copy()
        scene_score_ind = scores_ind.sum() / denom_ind
        scene_score_int = scores_int.sum() / denom_int
        scene_score = scene_score_int.copy() + scene_score_ind.copy()
        scene_score = np.clip(scene_score, a_min=self.score_clip.min, a_max=self.score_clip.max)
        return ScenarioScores(
            scenario_id=scenario.scenario_id,
            num_agents=scenario.num_agents,
            safeshift_agent_scores=scores,
            safeshift_scene_score=scene_score,
        )
