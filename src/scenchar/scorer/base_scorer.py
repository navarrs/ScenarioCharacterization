import re
from abc import ABC, abstractmethod

from omegaconf import DictConfig

from scenchar.utils.schemas import Scenario, ScenarioFeatures


class BaseScorer(ABC):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super(BaseScorer, self).__init__()
        self.config = config
        self.characterizer_type = "score"
        self.features = self.config.features
        self.detections = self.config.detections
        self.weights = self.config.weights
        self.score_clip = self.config.score_clip

    @property
    def name(self) -> str:
        """Gets the class name formatted as a lowercase string with spaces.

        Returns:
            str: The formatted class name.
        """
        # Get the class name and add a space before each capital letter (except the first)
        return re.sub(r"(?<!^)([A-Z])", r"_\1", self.__class__.__name__).lower()

    @abstractmethod
    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> dict:
        """Produces a dummy output for the feature computation.

        This method should be overridden by subclasses to compute actual features.

        Args:
            scenario_features (Dict): A dictionary containing scenario feature data.

        Returns:
            Dict: A dictionary with computed scores.
        """
        pass
