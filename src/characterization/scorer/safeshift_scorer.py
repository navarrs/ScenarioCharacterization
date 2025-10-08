import numpy as np
from omegaconf import DictConfig

from characterization.schemas import Scenario, ScenarioFeatures, ScenarioScores, Score
from characterization.scorer import BaseScorer, IndividualScorer, InteractionScorer
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


class SafeShiftScorer(BaseScorer):
    """Scorer that computes interaction scores for agent pairs and a scene-level score from scenario features."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the SafeShiftScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super().__init__(config)

        self.interaction_scorer = InteractionScorer(config)
        self.individual_scorer = IndividualScorer(config)

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
        scores = np.zeros(shape=(scenario.agent_data.num_agents,), dtype=np.float32)

        # Compute individual scores
        individual_scores: Score = self.individual_scorer.compute_individual_score(scenario, scenario_features)
        scores_ind = individual_scores.agent_scores
        if scores_ind is None:
            scores_ind = scores.copy()
        scene_score_ind = individual_scores.scene_score
        if scene_score_ind is None:
            scene_score_ind = 0.0

        # Compute interaction scores
        interaction_scores: Score = self.interaction_scorer.compute_interaction_score(scenario, scenario_features)
        scores_int = interaction_scores.agent_scores
        if scores_int is None:
            scores_int = scores.copy()

        scene_score_int = interaction_scores.scene_score
        if scene_score_int is None:
            scene_score_int = 0.0

        # Combine the scores
        agent_scores = scores_ind.copy() + scores_int.copy()
        scene_score = np.clip(
            0.5 * (scene_score_int + scene_score_ind), a_min=self.score_clip.min, a_max=self.score_clip.max
        )
        return ScenarioScores(
            metadata=scenario.metadata,
            individual_scores=individual_scores,
            interaction_scores=interaction_scores,
            safeshift_scores=Score(agent_scores=agent_scores, scene_score=scene_score),
        )
