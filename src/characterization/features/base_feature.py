import re
from abc import ABC, abstractmethod

from omegaconf import DictConfig

from characterization.schemas import Scenario, ScenarioFeatures
from characterization.utils.common import ReturnCriterion


class BaseFeature(ABC):
    """Abstract base class for scenario feature computation.

    Attributes:
        config (DictConfig): Configuration parameters for the feature extractor.
        features (Any): Feature configuration extracted from the config, if available.
        characterizer_type (str): Type identifier for the characterizer, always "feature".
        return_criterion (ReturnCriterion): Criterion determining when to return results.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the BaseFeature with configuration parameters.

        Args:
            config (DictConfig): Configuration dictionary containing feature parameters.
        """
        self.config = config
        self.features = config.get("features", None)
        self.detections = self.config.get("detections", None)
        self.characterizer_type = "feature"
        self.return_criterion = ReturnCriterion[config.get("return_criterion", "critical").upper()]
        self.compute_agent_to_agent_closest_dists = config.get("compute_agent_to_agent_closest_dists", False)

    @property
    def name(self) -> str:
        """Get the class name formatted as lowercase with spaces.

        Returns:
            str: The formatted class name with spaces separating words.
        """
        # Get the class name and add a space before each capital letter (except the first)
        return re.sub(r"(?<!^)([A-Z])", r" \1", self.__class__.__name__).lower()

    @abstractmethod
    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Compute features for a given scenario.

        Args:
            scenario (Scenario): The scenario data containing trajectories, road geometry, and metadata.

        Returns:
            ScenarioFeatures: Computed features for the scenario.

        Raises:
            ValueError: If the scenario lacks required information.
            NotImplementedError: If not implemented by subclass.
        """
