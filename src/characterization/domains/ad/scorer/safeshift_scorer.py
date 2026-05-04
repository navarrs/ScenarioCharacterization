"""AD-domain composite scorer (SafeShift): combines individual and interaction scores."""

import numpy as np
from omegaconf import DictConfig

from characterization.domains.ad.scorer.base_scorer import ADBaseScorer, ADScorerConfig
from characterization.domains.ad.scorer.individual_scorer import IndividualScorer
from characterization.domains.ad.scorer.interaction_scorer import InteractionScorer
from characterization.schemas import Scenario, ScenarioFeatures
from characterization.schemas.scenario_scores import AgentScore, ScenarioScores
from characterization.utils.logging_utils import get_pylogger

logger = get_pylogger(__name__)


class SafeShiftScorer(ADBaseScorer):
    """Composite scorer that combines individual and interaction scores into a single scene score."""

    def __init__(self, config: DictConfig | ADScorerConfig | None = None) -> None:
        """Initialize with AD scorer configuration.

        Args:
            config: Configuration as an :class:`ADScorerConfig`, an OmegaConf ``DictConfig``, or ``None``.
        """
        super().__init__(config)
        self._individual = IndividualScorer(self.config)
        self._interaction = InteractionScorer(self.config)

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Compute combined individual, interaction, and scene-level scores.

        Args:
            scenario: Scenario with agent metadata.
            scenario_features: Pre-computed features.

        Returns:
            :class:`ScenarioScores` with all score fields populated.
        """
        zero_scores = np.zeros(scenario.agent_data.num_agents, dtype=np.float32)

        ind_internal = self._individual._compute_individual_score(scenario, scenario_features)  # noqa: SLF001
        int_internal = self._interaction._compute_interaction_score(scenario, scenario_features)  # noqa: SLF001

        scores_ind = ind_internal.agent_scores if ind_internal.agent_scores is not None else zero_scores.copy()
        scores_int = int_internal.agent_scores if int_internal.agent_scores is not None else zero_scores.copy()
        scene_ind = ind_internal.scene_score or 0.0
        scene_int = int_internal.scene_score or 0.0

        combined_agent_scores = scores_ind + scores_int
        scene_score = float(
            np.clip(
                self.config.aggregated_score_weight * (scene_ind + scene_int),
                a_min=self.config.score_clip.min,
                a_max=self.config.score_clip.max,
            )
        )

        individual_scores = [AgentScore(agent_id=int(i), score=float(s)) for i, s in enumerate(scores_ind)]
        interaction_scores = [AgentScore(agent_id=int(i), score=float(s)) for i, s in enumerate(scores_int)]

        return ScenarioScores(
            scenario_id=scenario.metadata.scenario_id,
            individual_scores=individual_scores,
            interaction_scores=interaction_scores,
            individual_scene_score=ind_internal.scene_score,
            interaction_scene_score=int_internal.scene_score,
            scene_score=scene_score,
        )
