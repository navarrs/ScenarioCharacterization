"""Abstract base class for counterfactual scenario probers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from characterization.schemas import Scenario
    from characterization.schemas.critical_probe import CriticalProbe


class BaseProber(ABC):
    """Abstract base class for counterfactual scenario probers.

    Follows the same characterizer interface as :class:`~characterization.features.base_feature.BaseFeature` and
    :class:`~characterization.scorer.base_scorer.BaseScorer` so that
    :class:`~characterization.processors.probe_processor.ProbeProcessor` can be dispatched from the common
    :mod:`characterization.run_processor` entrypoint.
    """

    characterizer_type: str = "probe"

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this prober, used for logging.

        Returns:
            str: The name of the prober.
        """

    @abstractmethod
    def compute(self, scenario: Scenario) -> CriticalProbe | None:
        """Run the counterfactual probe on a single scenario.

        Args:
            scenario (Scenario): The scenario to probe.

        Returns:
            The most impactful :class:`~characterization.schemas.critical_probe.CriticalProbe` found, or ``None`` if no
            probe exceeds the configured threshold.
        """
