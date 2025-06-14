import re
from abc import ABC, abstractmethod

from omegaconf import DictConfig

from characterization.utils.schemas import Scenario, ScenarioFeatures


class BaseFeature(ABC):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseFeature with a configuration.

        Args:
            config (DictConfig): Configuration for the feature.
        """
        self.config = config
        self.features = config.features
        self.characterizer_type = "feature"

    @property
    def name(self) -> str:
        """Gets the class name formatted as a lowercase string with spaces.

        Returns:
            str: The formatted class name (e.g., 'base feature').
        """
        # Get the class name and add a space before each capital letter (except the first)
        return re.sub(r"(?<!^)([A-Z])", r" \1", self.__class__.__name__).lower()

    @abstractmethod
    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Computes features for a given scenario.

        This method should be overridden by subclasses to compute actual features.

        Args:
            scenario (Scenario): Scenario data to compute features for.

        Returns:
            ScenarioFeatures: Computed features for the scenario.

        Raises:
            ValueError: If the scenario does not contain the required information.
        """
        pass
