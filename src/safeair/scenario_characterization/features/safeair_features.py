"""Top-level feature extractor that combines individual and interaction features."""

from safeair.scenario_characterization.common import ReturnCriterion
from safeair.scenario_characterization.features.individual_features import IndividualFeatures
from safeair.scenario_characterization.features.interaction_features import InteractionFeatures
from safeair.schemas.scenario import Scenario
from safeair.schemas.scenario_features import CharacterizationParameters, ScenarioFeatures

_DEFAULT_CHARACTERIZATION = CharacterizationParameters()


class SafeAirFeatures:
    """Extracts both individual and interaction features from a scenario.

    This is the recommended entry point for scenario characterization. It instantiates
    ``IndividualFeatures`` and ``InteractionFeatures`` and merges their outputs into a single
    ``ScenarioFeatures`` object. Unit-conversion factors (xyz scale, speed units) are derived
    automatically from ``scenario.metadata`` at compute time, so no unit configuration is required.

    Args:
        return_criterion: Aggregation strategy passed to both sub-extractors.
        characterization: Thresholds controlling feature computation. Defaults to the standard SafeAir values.
        n_jobs: Number of parallel jobs for the interaction feature extractor. ``-1`` uses all CPUs.

    Example::

        extractor = SafeAirFeatures()
        features = extractor.compute(scenario)
        print(features.model_dump())
    """

    def __init__(
        self,
        return_criterion: ReturnCriterion = ReturnCriterion.CRITICAL,
        *,
        characterization: CharacterizationParameters = _DEFAULT_CHARACTERIZATION,
        n_jobs: int = -1,
    ) -> None:
        """Instantiate sub-extractors with the given aggregation settings."""
        self._individual = IndividualFeatures(return_criterion, characterization=characterization)
        self._interaction = InteractionFeatures(return_criterion, characterization=characterization, n_jobs=n_jobs)

    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Extract all features from a scenario.

        Args:
            scenario: The scenario to characterize.

        Returns:
            ScenarioFeatures containing per-agent kinematic features and pairwise interaction
            features for all candidate agent pairs.
        """
        return ScenarioFeatures(
            scenario_id=scenario.metadata.scenario_id,
            individual_features=self._individual.compute(scenario).individual_features,
            interaction_features=self._interaction.compute(scenario).interaction_features,
        )
