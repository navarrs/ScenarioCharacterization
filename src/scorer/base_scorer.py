import re
from abc import ABC
from itertools import combinations

import numpy as np
from omegaconf import DictConfig


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

    @property
    def name(self) -> str:
        """Gets the class name formatted as a lowercase string with spaces.

        Returns:
            str: The formatted class name.
        """
        # Get the class name and add a space before each capital letter (except the first)
        return re.sub(r"(?<!^)([A-Z])", r"_\1", self.__class__.__name__).lower()

    def compute(self, scenario: dict, scenario_features: dict) -> dict:
        """Produces a dummy output for the feature computation.

        This method should be overridden by subclasses to compute actual features.

        Args:
            scenario_features (Dict): A dictionary containing scenario feature data.

        Returns:
            Dict: A dictionary with computed scores.
        """
        # NOTE: to avoid overhead, it assumes the feature is already on the dictionary.
        feature_data = scenario_features["random_feature"]

        N = feature_data.shape[0]
        pair_indices = list(combinations(range(N), 2))
        scores = np.zeros(N, dtype=np.float32)
        for i, j in pair_indices:
            scores[i] += max(feature_data[i], feature_data[j])
            scores[j] += max(feature_data[i], feature_data[j])

        return {
            self.name: {
                "agent_scores": scores,
                "scene_score": np.mean(scores).astype(np.float32),
            }
        }
