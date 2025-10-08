import numpy as np
from omegaconf import DictConfig

from characterization.schemas import Scenario, ScenarioFeatures, ScenarioScores, Score
from characterization.scorer.base_scorer import BaseScorer
from characterization.utils.io_utils import get_logger

from .score_utils import INDIVIDUAL_SCORE_FUNCTIONS

logger = get_logger(__name__)


class IndividualScorer(BaseScorer):
    """Class to compute individual agent scores and a scene-level score from scenario features."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the IndividualScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super().__init__(config)

        individual_score_function = self.config.get("individual_score_function")
        if not individual_score_function:
            warning_message = (
                "No individual_score_function specified. Defaulting to 'simple'."
                f"If this is not intended, specify one of the supported functions: {INDIVIDUAL_SCORE_FUNCTIONS.keys()}"
            )
            individual_score_function = "simple"
            logger.warning(warning_message)

        if individual_score_function not in INDIVIDUAL_SCORE_FUNCTIONS:
            error_message = (
                f"Score function {individual_score_function} not supported. "
                f"Supported functions are: {list(INDIVIDUAL_SCORE_FUNCTIONS.keys())}"
            )
            raise ValueError(error_message)
        self.score_function = INDIVIDUAL_SCORE_FUNCTIONS[individual_score_function]

    def compute_individual_score(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> Score:
        """Computes individual agent scores and a scene-level score from scenario features.

        Args:
            scenario (Scenario): Scenario object containing scenario information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing computed features.

        Returns:
            ScenarioScores: An object containing computed individual agent scores and the scene-level score.

        Raises:
            ValueError: If any required feature (valid_idxs, speed, acceleration, deceleration, jerk, waiting_period)
                is missing in scenario_features.
        """
        scores = np.zeros(shape=(scenario.agent_data.num_agents,), dtype=np.float32)
        features = scenario_features.individual_features
        if features is None:
            warning_message = "individual_features is None. Cannot compute individual scores."
            logger.warning(warning_message)
            return Score(agent_scores=scores, scene_score=0.0)

        if features.valid_idxs is None:
            error_message = "valid_idxs must not be None if individual_features is not None. Cannot compute scores."
            raise ValueError(error_message)

        # Get the agent weights
        weights = self.get_weights(scenario, scenario_features)

        valid_idxs = features.valid_idxs
        for n in range(valid_idxs.shape[0]):
            valid_idx = valid_idxs[n]
            scores[valid_idx] = weights[valid_idx] * self.score_function(
                speed=features.speed[n] if features.speed is not None else 0.0,
                speed_weight=self.weights.speed,
                speed_detection=self.detections.speed,
                speed_diff=features.speed_limit_diff[n] if features.speed_limit_diff is not None else 0.0,
                speed_diff_weight=self.weights.speed_diff,
                speed_diff_detection=self.detections.speed_diff,
                acceleration=features.acceleration[n] if features.acceleration is not None else 0.0,
                acceleration_weight=self.weights.acceleration,
                acceleration_detection=self.detections.acceleration,
                deceleration=features.deceleration[n] if features.deceleration is not None else 0.0,
                deceleration_weight=self.weights.deceleration,
                deceleration_detection=self.detections.deceleration,
                jerk=features.jerk[n] if features.jerk is not None else 0.0,
                jerk_weight=self.weights.jerk,
                jerk_detection=self.detections.jerk,
                waiting_period=features.waiting_period[n] if features.waiting_period is not None else 0.0,
                waiting_period_weight=self.weights.waiting_period,
                waiting_period_detection=self.detections.waiting_period,
                trajectory_type=features.agent_trajectory_types[n],
                trajectory_type_weight=self.weights.trajectory_type,
                kalman_difficulty=features.kalman_difficulty[n] if features.kalman_difficulty is not None else 0.0,
                kalman_difficulty_weight=self.weights.kalman_difficulty,
                kalman_difficulty_detection=self.detections.kalman_difficulty,
            )
        # As a safeguard, replace NaNs with zeros
        scores = np.nan_to_num(scores, nan=0.0)

        # Normalize the scores
        denom = max(np.where(scores > 0.0)[0].shape[0], 1)
        scene_score = np.clip(scores.sum() / denom, a_min=self.score_clip.min, a_max=self.score_clip.max)
        return Score(agent_scores=scores, scene_score=scene_score)

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Computes individual agent scores and a scene-level score from scenario features.

        Args:
            scenario (Scenario): Scenario object containing scenario information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing computed features.

        Returns:
            ScenarioScores: An object containing computed individual agent scores and the scene-level score.

        Raises:
            ValueError: If any required feature (valid_idxs, speed, acceleration, deceleration, jerk, waiting_period)
                is missing in scenario_features.
        """
        return ScenarioScores(
            metadata=scenario.metadata,
            individual_scores=self.compute_individual_score(scenario, scenario_features),
        )
