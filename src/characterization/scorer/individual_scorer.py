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
        # TODO: avoid the overhead of these checks.
        individual_features = scenario_features.individual_features
        if individual_features is None:
            raise ValueError("individual_features must not be None")  # noqa: EM101, TRY003
        if individual_features.valid_idxs is None:
            raise ValueError("valid_idxs must not be None")  # noqa: EM101, TRY003
        if individual_features.speed is None:
            raise ValueError("speed must not be None")  # noqa: EM101, TRY003
        if individual_features.acceleration is None:
            raise ValueError("acceleration must not be None")  # noqa: EM101, TRY003
        if individual_features.deceleration is None:
            raise ValueError("deceleration must not be None")  # noqa: EM101, TRY003
        if individual_features.jerk is None:
            raise ValueError("jerk must not be None")  # noqa: EM101, TRY003
        if individual_features.waiting_period is None:
            raise ValueError("waiting_period must not be None")  # noqa: EM101, TRY003

        # Get the agent weights
        weights = self.get_weights(scenario, scenario_features)
        scores = np.zeros(shape=(scenario.agent_data.num_agents,), dtype=np.float32)

        valid_idxs = individual_features.valid_idxs
        for n in range(valid_idxs.shape[0]):
            # TODO: fix this indexing issue.
            valid_idx = valid_idxs[n]
            scores[valid_idx] = weights[valid_idx] * self.score_function(
                speed=individual_features.speed[n],
                speed_weight=self.weights.speed,
                speed_detection=self.detections.speed,
                acceleration=individual_features.acceleration[n],
                acceleration_weight=self.weights.acceleration,
                acceleration_detection=self.detections.acceleration,
                deceleration=individual_features.deceleration[n],
                deceleration_weight=self.weights.deceleration,
                deceleration_detection=self.detections.deceleration,
                jerk=individual_features.jerk[n],
                jerk_weight=self.weights.jerk,
                jerk_detection=self.detections.jerk,
                waiting_period=individual_features.waiting_period[n],
                waiting_period_weight=self.weights.waiting_period,
                waiting_period_detection=self.detections.waiting_period,
            )
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
