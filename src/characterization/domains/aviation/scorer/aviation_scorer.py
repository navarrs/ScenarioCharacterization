"""Aviation-domain composite scorer: combines individual and interaction scores."""

from characterization.domains.aviation.schemas.scenario import Scenario
from characterization.domains.aviation.schemas.scenario_features import ScenarioFeatures
from characterization.domains.aviation.scorer.base_scorer import AviationBaseScorer, AviationScorerConfig
from characterization.domains.aviation.scorer.individual_scorer import IndividualScorer
from characterization.domains.aviation.scorer.interaction_scorer import InteractionScorer
from characterization.schemas.scenario_scores import ScenarioScores


class AviationScorer(AviationBaseScorer):
    """Computes individual and interaction scores and combines them into a single scene score.

    The combined scene score is:

    .. code-block:: text

        scene_score = w * individual_scene_score + (1 - w) * interaction_scene_score

    where ``w = config.aggregated_score_weight`` (default 0.5).

    This is the recommended entry point for aviation scenario scoring. It instantiates
    :class:`IndividualScorer` and :class:`InteractionScorer` with the same configuration and merges their outputs.
    """

    def __init__(self, config: AviationScorerConfig | None = None) -> None:
        """Initialize sub-scorers with the given configuration."""
        super().__init__(config)
        self._individual = IndividualScorer(self.config)
        self._interaction = InteractionScorer(self.config)

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Compute individual, interaction, and combined scene scores.

        Args:
            scenario: The scenario to score.
            scenario_features: Pre-computed features for the scenario.

        Returns:
            ScenarioScores with all score fields populated.
        """
        individual_result = self._individual.compute(scenario, scenario_features)
        interaction_result = self._interaction.compute(scenario, scenario_features)

        weight = self.config.aggregated_score_weight
        individual_scene = individual_result.individual_scene_score or 0.0
        interaction_scene = interaction_result.interaction_scene_score or 0.0
        scene_score = weight * individual_scene + (1.0 - weight) * interaction_scene

        return ScenarioScores(
            scenario_id=scenario_features.scenario_id,
            individual_scores=individual_result.individual_scores,
            interaction_scores=interaction_result.interaction_scores,
            individual_scene_score=individual_result.individual_scene_score,
            interaction_scene_score=interaction_result.interaction_scene_score,
            scene_score=scene_score,
        )
