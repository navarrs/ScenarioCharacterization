import numpy as np

from abc import ABC

from omegaconf import DictConfig
from typing import Dict
from itertools import combinations

class BaseScorer(ABC):
    def __init__(self, config: DictConfig) -> None:
        """
        Initialize the BaseScore with a configuration.

        :param config: Configuration for the feature.
        """
        self.config = config
        self.characterizer_type = 'score'
        self.name = 'base_score'

    def compute(self, scenario_features: Dict) -> Dict:
        """ Produces a dummy output for the feature computation.
        This method should be overridden by subclasses to compute actual features.
        
        :param scenario: A dictionary containing scenario data.
        :return: A dictionary with computed features.
        """
        # NOTE: to avoid overhead, it assumes the feature is already on the dictionary. 
        feature_data = scenario_features['random_feature']
        
        N = feature_data.shape[0]
        pair_indices = list(combinations(range(N), 2))
        scores = np.zeros(N, dtype=np.float32)
        for i, j in pair_indices:
            scores[i] += max(feature_data[i], feature_data[j])
            scores[j] += max(feature_data[i], feature_data[j])

        return {
            self.name: {
                'agent_scores': scores,
                'scene_score': np.mean(scores).astype(np.float32)
            }
        }
        