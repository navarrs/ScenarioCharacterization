from abc import abstractmethod

from characterization.domains.aviation.schemas.scenario import Scenario
from characterization.domains.aviation.schemas.scenario_features import CharacterizationParameters, ScenarioFeatures
from characterization.features.base_feature import BaseFeature
from characterization.utils.common import ReturnCriterion, SpeedUnits, XYZScale
from characterization.utils.constants import KNOTS_TO_MS, SCALE_FACTOR_TO_M

_DEFAULT_CHARACTERIZATION = CharacterizationParameters()


class AviationBaseFeature(BaseFeature):
    """Abstract base class for aviation feature extractors.

    Subclasses implement ``compute()``, which returns a ``ScenarioFeatures`` object. Depending on the extractor type,
    only ``individual_features`` or ``interaction_features`` may be populated; ``AviationFeatures`` merges both.

    Unit-conversion factors (``_scale_to_m``, ``_speed_to_ms``) are derived at compute time from
    ``scenario.metadata.xyz_scale`` and ``scenario.metadata.speed_units``, so no unit configuration is required at
    construction.

    Args:
        return_criterion: How to aggregate per-timestep values to a single scalar.
        characterization: Thresholds controlling feature computation. Defaults to the standard aviation values.
    """

    characterizer_type: str = "feature"

    def __init__(
        self,
        return_criterion: ReturnCriterion,
        *,
        characterization: CharacterizationParameters = _DEFAULT_CHARACTERIZATION,
    ) -> None:
        """Store the aggregation criterion and characterization parameters."""
        if return_criterion == ReturnCriterion.ALL:
            msg = "ReturnCriterion.ALL is not supported by feature extractors; use CRITICAL or AVERAGE."
            raise ValueError(msg)
        self.return_criterion = return_criterion
        self._characterization = characterization

        # Intitialized to SI defaults; overwritten at compute time via _set_unit_factors().
        self._scale_to_m: float = SCALE_FACTOR_TO_M[XYZScale.M]
        self._speed_to_ms: float = 1.0

    @property
    def characterization(self) -> CharacterizationParameters:
        """Return the characterization parameters used by this extractor."""
        return self._characterization

    def _set_unit_factors(self, scenario: Scenario) -> None:
        """Derive and store unit-conversion factors from the scenario metadata."""
        self._scale_to_m = SCALE_FACTOR_TO_M[scenario.metadata.xyz_scale]
        self._speed_to_ms = KNOTS_TO_MS if scenario.metadata.speed_units == SpeedUnits.KNOTS else 1.0

    @abstractmethod
    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Compute features for the given scenario.

        Args:
            scenario: The scenario to characterize.

        Returns:
            ScenarioFeatures with at least one of ``individual_features`` or
            ``interaction_features`` populated.
        """
        ...


class AviationFeatures(BaseFeature):
    """Extracts both individual and interaction features from a scenario.

    This is the recommended entry point for aviation scenario characterization. It instantiates
    ``IndividualFeatures`` and ``InteractionFeatures`` and merges their outputs into a single
    ``ScenarioFeatures`` object. Unit-conversion factors (xyz scale, speed units) are derived
    automatically from ``scenario.metadata`` at compute time, so no unit configuration is required.

    Args:
        return_criterion: Aggregation strategy passed to both sub-extractors.
        characterization: Thresholds controlling feature computation. Defaults to the standard aviation values.
        n_jobs: Number of parallel jobs for the interaction feature extractor. ``-1`` uses all CPUs.

    Example::

        extractor = AviationFeatures()
        features = extractor.compute(scenario)
        print(features.model_dump())
    """

    characterizer_type: str = "feature"

    def __init__(
        self,
        return_criterion: ReturnCriterion = ReturnCriterion.CRITICAL,
        *,
        characterization: CharacterizationParameters = _DEFAULT_CHARACTERIZATION,
        n_jobs: int = -1,
    ) -> None:
        """Instantiate sub-extractors with the given aggregation settings."""
        # Local imports avoid circular dependency: IndividualFeatures and InteractionFeatures
        # both import AviationBaseFeature from this module.
        from characterization.domains.aviation.features.individual_features import IndividualFeatures  # noqa: PLC0415
        from characterization.domains.aviation.features.interaction_features import InteractionFeatures  # noqa: PLC0415

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
