from abc import ABC, abstractmethod

from characterization.utils.common import SpeedUnits, XYZScale
from characterization.utils.constants import KNOTS_TO_MS, SCALE_FACTOR_TO_M
from safeair.scenario_characterization.common import ReturnCriterion
from safeair.schemas.scenario import Scenario
from safeair.schemas.scenario_features import CharacterizationParameters, ScenarioFeatures

_DEFAULT_CHARACTERIZATION = CharacterizationParameters()


class BaseFeature(ABC):
    """Abstract base class for all scenario feature extractors.

    Subclasses implement ``compute()``, which returns a ``ScenarioFeatures`` object. Depending on the extractor type,
    only ``individual_features`` or ``interaction_features`` may be populated; ``SafeAirFeatures`` merges both.

    Unit-conversion factors (``_scale_to_m``, ``_speed_to_ms``) are derived at compute time from
    ``scenario.metadata.xyz_scale`` and ``scenario.metadata.speed_units``, so no unit configuration is required at
    construction.

    Args:
        return_criterion: How to aggregate per-timestep values to a single scalar.
        characterization: Thresholds controlling feature computation. Defaults to the standard SafeAir values.
    """

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
