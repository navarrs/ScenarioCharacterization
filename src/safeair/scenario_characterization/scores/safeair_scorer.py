"""Top-level scorer that combines individual and interaction scores for SafeAir scenarios."""

from safeair.scenario_characterization.scores.base_scorer import BaseScorer, ScorerConfig
from safeair.scenario_characterization.scores.individual_scorer import IndividualScorer
from safeair.scenario_characterization.scores.interaction_scorer import InteractionScorer
from safeair.schemas.scenario import Scenario
from safeair.schemas.scenario_features import ScenarioFeatures
from safeair.schemas.scenario_scores import ScenarioScores


class SafeAirScorer(BaseScorer):
    """Computes individual and interaction scores and combines them into a single scene score.

    The combined scene score is:

    .. code-block:: text

        scene_score = w * individual_scene_score + (1 - w) * interaction_scene_score

    where ``w = config.aggregated_score_weight`` (default 0.5).

    This is the recommended entry point for scenario scoring. It instantiates ``IndividualScorer`` and
    ``InteractionScorer`` with the same configuration and merges their outputs.

    Args:
        config: Scorer configuration shared by both sub-scorers. Defaults to ``ScorerConfig()``.
        individual_score_function: Scoring function name for individual features (default ``"simple"``).
        interaction_score_function: Scoring function name for interaction features (default ``"simple"``).

    Example::

        scorer = SafeAirScorer()
        scores = scorer.compute(scenario, features)
        print(scores.scene_score)
    """

    def __init__(
        self,
        config: ScorerConfig | None = None,
        *,
        individual_score_function: str = "simple",
        interaction_score_function: str = "simple",
    ) -> None:
        """Instantiate sub-scorers with the given configuration and scoring functions."""
        super().__init__(config)
        self._individual = IndividualScorer(self.config, score_function=individual_score_function)
        self._interaction = InteractionScorer(self.config, score_function=interaction_score_function)

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Compute individual, interaction, and combined scene scores.

        Args:
            scenario: The scenario to score.
            scenario_features: Pre-computed features for the scenario.

        Returns:
            ScenarioScores with all score fields populated.
        """
        individual_score = self._individual.compute(scenario, scenario_features)
        interaction_score = self._interaction.compute(scenario, scenario_features)

        # Combine scene scores using the aggregated score weight
        weight = self.config.aggregated_score_weight
        individual_scene = individual_score.individual_scene_score or 0.0
        interaction_scene = interaction_score.interaction_scene_score or 0.0
        scene_score = weight * individual_scene + (1.0 - weight) * interaction_scene

        return ScenarioScores(
            scenario_id=scenario_features.scenario_id,
            individual_scores=individual_score.individual_scores,
            interaction_scores=interaction_score.interaction_scores,
            individual_scene_score=individual_score.individual_scene_score,
            interaction_scene_score=interaction_score.interaction_scene_score,
            scene_score=scene_score,
        )
