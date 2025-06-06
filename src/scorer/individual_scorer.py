from itertools import combinations
from typing import Dict

import numpy as np
from omegaconf import DictConfig

from src.scorer.base_scorer import BaseScorer
from src.utils.common import EPS, get_logger

logger = get_logger(__name__)

class IndividualScorer(BaseScorer):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super(IndividualScorer, self).__init__(config)

    def aggregate_simple_score(self, **kwargs) -> np.ndarray:
        # Detection values are roughly obtained from: https://arxiv.org/abs/2202.07438
        return min(self.detections.speed, self.weights.speed * kwargs.get('speed', 0.0)) + \
            min(self.detections.acceleration, self.weights.acceleration * kwargs.get('acceleration', 0.0)) + \
            min(self.detections.deceleration, self.weights.deceleration * kwargs.get('deceleration', 0.0)) + \
            min(self.detections.jerk, self.weights.jerk * kwargs.get('jerk', 0.0)) + \
            min(self.detections.waiting_period, self.weights.waiting_period * np.sqrt(kwargs.get('waiting_period', 0.0))) 
        
    def compute(self, scenario: Dict, scenario_features: Dict) -> Dict:
        """Produces a dummy output for the feature computation.

        This method should be overridden by subclasses to compute actual features.

        Args:
            scenario_features (Dict): A dictionary containing scenario feature data.

        Returns:
            Dict: A dictionary with computed scores.
        """
        # NOTE: should we avoid this overhead?
        missing_features = [feature for feature in self.features if feature not in scenario_features]
        if missing_features:
            raise ValueError(f"Missing features in scenario_features: {missing_features}")
        
        agent_to_agent_dists = scenario_features['agent_to_agent_closest_dists']
        relevant_agents = np.where(scenario['agent_relevance'] > 0.0)[0]
        relevant_agents_values = scenario['agent_relevance'][relevant_agents]
        relevant_agents_dists = agent_to_agent_dists[:, relevant_agents]
        
        # TODO: make this configurable/controllable
        # TODO: paralellize this
        N = len(scenario_features['valid_agents'])
        scores = np.zeros(shape=(N,), dtype=np.float32)
        for n, idx in enumerate(scenario_features['valid_agents']):
            
            # scenario includes all valid and non-valid agents, which is why we index using `idx`
            min_dist, argmin_dist = relevant_agents_dists[idx].min(), relevant_agents_dists[idx].argmin()
            # An agent's contribution to the score is inversely proportional to the closest distance
            # between the agent and the relevant agents
            weight = relevant_agents_values[argmin_dist] * min(1.0 / (min_dist + EPS), 1.0)

            scores[n] = weight * self.aggregate_simple_score(
                speed = scenario_features['speed'][n],
                acceleration = scenario_features['acceleration'][n],
                deceleration = scenario_features['deceleration'][n],
                jerk = scenario_features['jerk'][n],
                waiting_period = scenario_features['waiting_period'][n],
                waiting_intervals = scenario_features['waiting_intervals'][n],
                waiting_distances = scenario_features['waiting_distances'][n],
                # agent_type = scenario_features['agent_types'][n]
            )
            
        return {
            self.name: {
                "agent_scores": scores,
                "scene_score": scores.mean(),
            }
        }
