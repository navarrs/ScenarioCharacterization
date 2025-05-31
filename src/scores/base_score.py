import numpy as np

from abc import ABC

from omegaconf import DictConfig
from typing import Dict, AnyStr
from itertools import combinations

class BaseScore(ABC):
    def __init__(self, config: DictConfig) -> None:
        """
        Initialize the BaseScore with a configuration.

        :param config: Configuration for the feature.
        """
        self.config = config
        self.name = 'base_score'

    def compute(self, scenario_features: Dict) -> Dict:
        """ Produces a dummy output for the feature computation.
        This method should be overridden by subclasses to compute actual features.
        
        :param scenario: A dictionary containing scenario data.
        :return: A dictionary with computed features.
        """
        # TODO: generalize this by quering the Dataset class for the trajectories
        # trajectories = scenario['track_infos']['trajs']
        # N, T, D = trajectories.shape
        feature = scenario_features[0]
        scenario_id = list(feature.keys())[0]  
        feature_data = feature[scenario_id].get('random_feature', None)
        if feature_data is None:
            raise ValueError("Feature data not found in the scenario features.")

        N = feature_data.shape[0]
        pair_indices = list(combinations(range(N), 2))
        scores = np.zeros(N, dtype=np.float32)
        for i, j in pair_indices:
            scores[i] += max(feature_data[i], feature_data[j])
            scores[j] += max(feature_data[i], feature_data[j])

        return {
            scenario_id: {
                'agent_scores': scores,
                'scene_score': np.mean(scores).astype(np.float32)
            }
        }
        