import itertools
from enum import Enum

import numpy as np
from omegaconf import DictConfig

import features.interaction_utils as interaction
from features.base_feature import BaseFeature
from utils.common import EPS, get_logger
from utils.schemas import Scenario

logger = get_logger(__name__)


class InteractionStatus(Enum):
    UNKNOWN = -1
    COMPUTED_OK = 0
    MASK_NOT_VALID = 1
    AGENT_DISTANCE_TOO_FAR = 2
    AGENTS_STATIONARY = 3


class InteractionFeatures(BaseFeature):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseFeature with a configuration.

        Args:
            config (DictConfig): Configuration for the feature. Expected to contain key-value pairs
                relevant to feature computation, such as thresholds or parameters.
        """
        super(InteractionFeatures, self).__init__(config)

        self.return_criteria = config.get("return_criteria", "critical")
        self.agent_i = interaction.InteractionAgent()
        self.agent_j = interaction.InteractionAgent()

    def reset_state(self, agent_combinations: list, agent_types: list) -> dict:
        """Resets the state dictionary for feature computation.

        Returns:
            Dict: A dictionary with empty lists for each feature to be computed.
        """
        self.agent_i.reset()
        self.agent_j.reset()
        interaction_status = np.asarray([InteractionStatus.UNKNOWN for _ in agent_combinations])
        return {
            "interaction_status": interaction_status.copy(),
            "separation": [],
            "intersection": [],
            "mttcp": [],
            "collision": [],
            "agent_pair_indeces": [(i, j) for i, j in agent_combinations],
            "agents_pair_types": [(agent_types[i], agent_types[j]) for i, j in agent_combinations],
        }

    def compute(self, scenario: Scenario) -> dict:
        """Computes features for each agent in the scenario.

        Args:
            scenario (Dict): A dictionary containing scenario data.

        Returns:
            Dict: A dictionary with computed features for each agent.

        Raises:
            ValueError: If the 'scenario' dictionary does not contain the key 'num_agents'.
        """
        agent_combinations = list(itertools.combinations(range(scenario.num_agents), 2))

        agent_types = scenario.agent_types
        agent_masks = scenario.agent_valid
        agent_positions = scenario.agent_positions
        agent_velocities = np.linalg.norm(scenario.agent_velocities, axis=-1) + EPS
        agent_headings = scenario.agent_headings.squeeze(-1)
        conflict_points = scenario.map_conflict_points
        dists_to_conflict_points = scenario.agent_distances_to_conflict_points

        # TODO: Figure out where's best place to get these from
        stationary_speed = scenario.stationary_speed
        agent_to_agent_max_distance = scenario.agent_to_agent_max_distance
        agent_to_conflict_point_max_distance = scenario.agent_to_conflict_point_max_distance
        agent_to_agent_distance_breach = scenario.agent_to_agent_distance_breach

        state = self.reset_state(agent_combinations, agent_types)

        # Compute distance to conflict points
        for n, (i, j) in enumerate(agent_combinations):
            for feature in self.features:
                state[feature].append([])  # Initialize feature lists

            mask_i, mask_j = agent_masks[i], agent_masks[j]
            mask = np.where(mask_i & mask_j)[0]
            if not mask.sum():
                # No valid data for this pair of agents
                state["interaction_status"][n] = InteractionStatus.MASK_NOT_VALID
                continue

            self.agent_i.position, self.agent_j.position = agent_positions[i][mask], agent_positions[j][mask]
            self.agent_i.velocity, self.agent_j.velocity = agent_velocities[i][mask], agent_velocities[j][mask]
            self.agent_i.heading, self.agent_j.heading = agent_headings[i][mask], agent_headings[j][mask]
            self.agent_i.agent_type, self.agent_j.agent_type = agent_types[i], agent_types[j]

            if conflict_points is not None:
                self.agent_j.dists_to_conflict = dists_to_conflict_points[i][mask]
                self.agent_j.dists_to_conflict = dists_to_conflict_points[j][mask]

            # Check if agents are within a valid distance threshold to compute interactions
            separation = interaction.compute_separation(self.agent_i, self.agent_j)
            if not np.any(separation <= agent_to_agent_max_distance):
                state["interaction_status"][n] = InteractionStatus.AGENT_DISTANCE_TOO_FAR
                continue

            # Check if agents are stationary
            self.agent_i.stationary_speed = stationary_speed
            self.agent_j.stationary_speed = stationary_speed
            if self.agent_i.is_stationary and self.agent_i.is_stationary:
                state["interaction_status"][n] = InteractionStatus.AGENTS_STATIONARY
                continue

            # Check if agents are close to conflict points
            # self.agent_i.in_conflict_point, self.agent_j.in_conflict_point = 0, 0

            # Compute interaction features
            separation = interaction.compute_separation(self.agent_i, self.agent_j)
            intersections = interaction.compute_intersections(self.agent_i, self.agent_j)
            collisions = (separation <= agent_to_agent_distance_breach) | intersections
            intersections = intersections.astype(np.float32)
            collisions = collisions.astype(np.float32)

            # Minimum time to conflict point (mTTCP)
            mttcp = interaction.compute_mttcp(self.agent_i, self.agent_j, agent_to_conflict_point_max_distance)

            if self.return_criteria == "critical":
                separation = separation.min()
                intersections = intersections.sum()
                collisions = collisions.sum()
                mttcp = mttcp.min()
            elif self.return_criteria == "average":
                separation = separation.mean()
                intersections = intersections.mean()
                collisions = collisions.mean()
                mttcp = mttcp.mean()

            self.agent_i.reset()
            self.agent_j.reset()

            # Store computed features in the state dictionary
            state["separation"][n] = separation
            state["intersection"][n] = intersections
            state["collision"][n] = collisions
            state["mttcp"][n] = mttcp
            state["interaction_status"][n] = InteractionStatus.COMPUTED_OK

        return state
