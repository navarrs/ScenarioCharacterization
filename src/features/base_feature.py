import numpy as np

from abc import ABC

from omegaconf import DictConfig
from typing import Dict, AnyStr

class BaseFeature(ABC):
    def __init__(self, config: DictConfig) -> None:
        """
        Initialize the BaseFeature with a configuration.

        :param config: Configuration for the feature.
        """
        self.config = config
        self.name = 'base_feature'

    def compute(self, scenario: Dict, scenario_id: AnyStr) -> Dict:
        """ Produces a dummy output for the feature computation.
        This method should be overridden by subclasses to compute actual features.
        
        :param scenario: A dictionary containing scenario data.
        :return: A dictionary with computed features.
        """
        # TODO: generalize this by quering the Dataset class for the trajectories
        trajectories = scenario['track_infos']['trajs']
        N, T, D = trajectories.shape
        return {
            scenario_id: {
                'random_feature': 10.0 * np.random.rand(N).astype(np.float32)
            }
        }