import numpy as np
from omegaconf import DictConfig

from characterization.scorer import BaseScorer, IndividualScorer, InteractionScorer
from characterization.utils.io_utils import get_logger
from characterization.schemas import ScenarioScores, ScenarioFeatures, Scenario, Score
logger = get_logger(__name__)


class SafeShiftScorer(BaseScorer):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the SafeShiftScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super(SafeShiftScorer, self).__init__(config)

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
        individual_scores: Score = self.individual_scorer.compute_individual_score(scenario, scenario_features)
        scores_ind = individual_scores.agent_scores
        assert scores_ind is not None, "Individual agent scores must not be None"
        scene_score_ind = individual_scores.scene_score
        assert scene_score_ind is not None, "Scene score must not be None"

        interaction_scores: Score = self.interaction_scorer.compute_interaction_score(scenario, scenario_features)
        scores_int = interaction_scores.agent_scores
        assert scores_int is not None, "Interaction agent scores must not be None"
        scene_score_int = interaction_scores.scene_score
        assert scene_score_int is not None, "Scene score must not be None"

        # Combine the scores
        scores = scores_ind.copy() + scores_int.copy()
        scene_score = np.clip(scene_score_int + scene_score_ind, a_min=self.score_clip.min, a_max=self.score_clip.max)
        return ScenarioScores(
            metadata=scenario.metadata,
            individual_scores=individual_scores,
            interaction_scores=interaction_scores,
            safeshift_scores=Score(agent_scores=scores, scene_score=scene_score),
        )
