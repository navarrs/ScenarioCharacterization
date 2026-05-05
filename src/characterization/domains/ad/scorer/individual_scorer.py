"""AD-domain individual scorer."""

from warnings import warn

import numpy as np
from omegaconf import DictConfig

from characterization.domains.ad.schemas import Scenario, ScenarioFeatures
from characterization.domains.ad.schemas.scenario_scores import Score
from characterization.domains.ad.scorer.base_scorer import ADBaseScorer, ADScorerConfig
from characterization.schemas.scenario_scores import AgentScore, ScenarioScores
from characterization.utils.common import TrajectoryType
from characterization.utils.logging_utils import get_pylogger

from .score_functions import INDIVIDUAL_SCORE_FUNCTIONS

logger = get_pylogger(__name__)


class IndividualScorer(ADBaseScorer):
    """Computes individual agent scores and a scene-level score from scenario features."""

    def __init__(self, config: DictConfig | ADScorerConfig | None = None) -> None:
        """Initialize with AD scorer configuration.

        Args:
            config: Configuration as an :class:`ADScorerConfig`, an OmegaConf ``DictConfig``, or ``None``.
        """
        super().__init__(config)

        score_function_name = self.config.individual_score_function
        if score_function_name not in INDIVIDUAL_SCORE_FUNCTIONS:
            error_message = (
                f"Score function {score_function_name} not supported. "
                f"Supported functions are: {list(INDIVIDUAL_SCORE_FUNCTIONS.keys())}"
            )
            raise ValueError(error_message)
        self._score_fn = INDIVIDUAL_SCORE_FUNCTIONS[score_function_name]

        if self.config.categorize_scores:
            self.categories = self._load_categorization_file(self.config.individual_categorization_file)

    def _compute_individual_score(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> Score:
        """Compute per-agent individual scores as an internal numpy-based Score object.

        Args:
            scenario: Scenario with agent metadata.
            scenario_features: Pre-computed individual features.

        Returns:
            :class:`Score` with numpy agent_scores array and scene_score.
        """
        features = scenario_features.individual_features
        if not features:
            warning_message = f"Invalid individual features for {scenario.metadata.scenario_id}."
            warn(warning_message, UserWarning, stacklevel=3)
            return Score(agent_scores=None, agent_scores_valid=None, scene_score=None)

        scores = np.zeros(shape=(scenario.agent_data.num_agents,), dtype=np.float32)
        valid = np.zeros(shape=(scenario.agent_data.num_agents,), dtype=bool)
        weights = self.get_weights(scenario, scenario_features)

        for agent_feat in features:
            n = agent_feat.agent_id
            traj_type = (
                TrajectoryType[agent_feat.trajectory_type]
                if agent_feat.trajectory_type
                else TrajectoryType.TYPE_STATIONARY
            )
            score_value = weights[n] * self._score_fn(  # pyright: ignore[reportCallIssue]
                speed=agent_feat.speed or 0.0,
                speed_weight=self.config.weights.speed,
                speed_detection=self.config.detections.speed,
                speed_limit_diff=agent_feat.speed_limit_diff or 0.0,
                speed_limit_diff_weight=self.config.weights.speed_limit_diff,
                speed_limit_diff_detection=self.config.detections.speed_limit_diff,
                acceleration=agent_feat.acceleration or 0.0,
                acceleration_weight=self.config.weights.acceleration,
                acceleration_detection=self.config.detections.acceleration,
                deceleration=agent_feat.deceleration or 0.0,
                deceleration_weight=self.config.weights.deceleration,
                deceleration_detection=self.config.detections.deceleration,
                jerk=agent_feat.jerk or 0.0,
                jerk_weight=self.config.weights.jerk,
                jerk_detection=self.config.detections.jerk,
                waiting_period=agent_feat.waiting_period or 0.0,
                waiting_period_weight=self.config.weights.waiting_period,
                waiting_period_detection=self.config.detections.waiting_period,
                trajectory_type=traj_type,
                trajectory_type_weight=self.config.weights.trajectory_type,
                kalman_difficulty=agent_feat.kalman_difficulty or 0.0,
                kalman_difficulty_weight=self.config.weights.kalman_difficulty,
                kalman_difficulty_detection=self.config.detections.kalman_difficulty,
            )

            if self.config.categorize_scores:
                score_value = self.categorize(float(score_value))

            scores[n] = score_value
            valid[n] = True

        scores = np.nan_to_num(scores, nan=0.0)
        denom = max(len(features), 1)
        scene_score = float(
            np.clip(scores.sum() / denom, a_min=self.config.score_clip.min, a_max=self.config.score_clip.max)
        )
        return Score(agent_scores=scores, agent_scores_valid=valid, scene_score=scene_score)

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Compute individual agent scores and a scene-level score.

        Args:
            scenario: Scenario with agent metadata.
            scenario_features: Pre-computed individual features.

        Returns:
            :class:`ScenarioScores` with ``individual_scores`` and ``individual_scene_score`` populated.
        """
        internal = self._compute_individual_score(scenario, scenario_features)
        agent_scores_list: list[AgentScore] = []
        if internal.agent_scores is not None:
            agent_scores_list = [
                AgentScore(agent_id=int(i), score=float(s)) for i, s in enumerate(internal.agent_scores)
            ]
        return ScenarioScores(
            scenario_id=scenario.metadata.scenario_id,
            individual_scores=agent_scores_list,
            individual_scene_score=internal.scene_score,
        )
