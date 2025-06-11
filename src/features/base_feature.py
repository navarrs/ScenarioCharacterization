import re
from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig

from utils.schemas import Scenario


class BaseFeature(ABC):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseFeature with a configuration.

        Args:
            config (DictConfig): Configuration for the feature. Expected to contain key-value pairs
                relevant to feature computation, such as thresholds or parameters.
        """
        self.config = config
        self.features = config.features
        self.characterizer_type = "feature"

    @property
    def name(self) -> str:
        """Gets the class name formatted as a lowercase string with spaces.

        Returns:
            str: The formatted class name.
        """
        # Get the class name and add a space before each capital letter (except the first)
        return re.sub(r"(?<!^)([A-Z])", r" \1", self.__class__.__name__).lower()

    @abstractmethod
    def compute(self, scenario: Scenario) -> dict:
        """Produces a dummy output for the feature computation.

        This method should be overridden by subclasses to compute actual features.

        Args:
            scenario (Dict): A dictionary containing scenario data.

        Returns:
            Dict: A dictionary with computed features.

        Raises:
            ValueError: If the 'scenario' dictionary does not contain the key 'num_agents'.
        """
        if not scenario.get("num_agents", None):
            raise ValueError("The 'scenario' dictionary must contain the key 'num_agents'.")
        N = scenario["num_agents"]
        return {"random_feature": 10.0 * np.random.rand(N).astype(np.float32)}
