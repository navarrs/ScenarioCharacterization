"""Aviation-domain individual (per-agent) kinematic scorer."""

import numpy as np

from characterization.domains.aviation.schemas.scenario import Scenario
from characterization.domains.aviation.schemas.scenario_features import ScenarioFeatures
from characterization.domains.aviation.scorer.base_scorer import AviationBaseScorer, AviationScorerConfig
from characterization.domains.aviation.scorer.score_functions import INDIVIDUAL_SCORE_REGISTRY
from characterization.schemas.scenario_scores import AgentScore, ScenarioScores


class IndividualScorer(AviationBaseScorer):
    """Computes per-agent individual scores and a scene-level aggregate from kinematic features.

    Each agent's score is a weighted sum of its kinematic features, capped by per-feature detection thresholds.
    The scene score is the sum of all positive agent scores divided by the number of agents with a positive score,
    then clipped to ``config.score_clip``.
    """

    def __init__(self, config: AviationScorerConfig | None = None) -> None:
        """Initialize with aviation scorer configuration."""
        super().__init__(config)
        score_function = self.config.individual_score_function
        if score_function not in INDIVIDUAL_SCORE_REGISTRY:
            error_message = (
                f"Individual score function '{score_function}' not supported. "
                f"Supported: {list(INDIVIDUAL_SCORE_REGISTRY.keys())}"
            )
            raise ValueError(error_message)
        self._score_fn = INDIVIDUAL_SCORE_REGISTRY[score_function]
        if self.config.categorize_scores:
            self.categories = self._load_categorization_file(self.config.individual_categorization_file)

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Compute per-agent individual scores and a scene-level score.

        Args:
            scenario: The scenario to score.
            scenario_features: Pre-computed features containing ``individual_features``.

        Returns:
            ScenarioScores with ``individual_scores`` and ``individual_scene_score`` populated.
        """
        agent_id_to_idx: dict[int, int] = {int(aid): i for i, aid in enumerate(scenario.agent_data.agent_ids)}
        weights = self._compute_agent_weights(scenario)
        raw_scores = np.zeros(scenario.agent_data.num_agents, dtype=np.float32)

        for feat in scenario_features.individual_features:
            idx = agent_id_to_idx.get(feat.agent_id)
            if idx is None:
                continue

            score_value = weights[idx] * self._score_fn(  # pyright: ignore[reportCallIssue]
                speed=feat.speed if feat.speed is not None else 0.0,
                speed_weight=self.config.weights.speed,
                speed_detection=self.config.detections.speed,
                acceleration=feat.acceleration if feat.acceleration is not None else 0.0,
                acceleration_weight=self.config.weights.acceleration,
                acceleration_detection=self.config.detections.acceleration,
                deceleration=feat.deceleration if feat.deceleration is not None else 0.0,
                deceleration_weight=self.config.weights.deceleration,
                deceleration_detection=self.config.detections.deceleration,
                waiting_period=feat.waiting_period if feat.waiting_period is not None else 0.0,
                waiting_period_weight=self.config.weights.waiting_period,
                waiting_period_detection=self.config.detections.waiting_period,
                trajectory_type=feat.trajectory_type or "",
                trajectory_type_weight=self.config.weights.trajectory_type,
                kalman_difficulty=feat.kalman_difficulty if feat.kalman_difficulty is not None else 0.0,
                kalman_difficulty_weight=self.config.weights.kalman_difficulty,
                kalman_difficulty_detection=self.config.detections.kalman_difficulty,
            )
            raw_scores[idx] = float(np.nan_to_num(score_value, nan=0.0))

        if self.config.categorize_scores:
            for i in range(len(raw_scores)):
                raw_scores[i] = self.categorize(float(raw_scores[i]))

        agent_scores = [
            AgentScore(agent_id=int(aid), score=float(raw_scores[i]))
            for i, aid in enumerate(scenario.agent_data.agent_ids)
        ]

        denom = max(int(np.sum(raw_scores > 0.0)), 1)
        scene_score = self.config.score_clip.clip(float(raw_scores.sum() / denom))

        return ScenarioScores(
            scenario_id=scenario_features.scenario_id,
            individual_scores=agent_scores,
            individual_scene_score=scene_score,
        )
