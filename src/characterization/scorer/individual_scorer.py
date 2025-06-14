import numpy as np
from omegaconf import DictConfig

from characterization.scorer.base_scorer import BaseScorer
from characterization.utils.common import get_logger
from characterization.utils.schemas import Scenario, ScenarioFeatures, ScenarioScores

logger = get_logger(__name__)


class IndividualScorer(BaseScorer):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the IndividualScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super(IndividualScorer, self).__init__(config)

    def aggregate_simple_score(self, **kwargs) -> np.ndarray:
        """Aggregates a simple score for an agent using weighted feature values.

        Args:
            **kwargs: Feature values for the agent, including speed, acceleration, deceleration,
                jerk, and waiting_period.

        Returns:
            np.ndarray: The aggregated score for the agent.
        """
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

        # Get the agent weights
        weights = self.get_weights(scenario, scenario_features)
        scores = np.zeros(shape=(scenario.num_agents,), dtype=np.float32)

        valid_idxs = scenario_features.valid_idxs
        N = valid_idxs.shape[0]
        for n in range(N):
            # TODO: fix this indexing issue.
            valid_idx = valid_idxs[n]
            scores[valid_idx] = weights[valid_idx] * self.aggregate_simple_score(
                speed=scenario_features.speed[n],
                acceleration=scenario_features.acceleration[n],
                deceleration=scenario_features.deceleration[n],
                jerk=scenario_features.jerk[n],
                waiting_period=scenario_features.waiting_period[n],
            )

        # Normalize the scores
        denom = max(np.where(scores > 0.0)[0].shape[0], 1)
        scene_score = np.clip(scores.sum() / denom, a_min=self.score_clip.min, a_max=self.score_clip.max)
        return ScenarioScores(
            scenario_id=scenario.scenario_id,
            num_agents=scenario.num_agents,
            individual_agent_scores=scores,
            individual_scene_score=scene_score,
        )
