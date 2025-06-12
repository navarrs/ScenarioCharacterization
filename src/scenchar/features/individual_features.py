import numpy as np
from omegaconf import DictConfig

import scenchar.features.individual_utils as individual
from scenchar.features.base_feature import BaseFeature
from scenchar.utils.common import get_logger
from scenchar.utils.schemas import Scenario

logger = get_logger(__name__)


class IndividualFeatures(BaseFeature):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseFeature with a configuration.

        Args:
            config (DictConfig): Configuration for the feature. Expected to contain key-value pairs
                relevant to feature computation, such as thresholds or parameters.
        """
        super(IndividualFeatures, self).__init__(config)

        self.return_criteria = config.get("return_criteria", "critical")

    def reset_state(self) -> dict:
        """Resets the state dictionary for feature computation.

        Returns:
            Dict: A dictionary with empty lists for each feature to be computed.
        """
        return {
            "valid_agents": [],
            "speed": [],
            "speed_limit_diff_feature_n": [],
            "acceleration": [],
            "deceleration": [],
            "jerk": [],
            "waiting_period": [],
            "waiting_intervals": [],
            "waiting_distances": [],
            "agent_types": [],
            "in_lane": [],
            "trajectory_anomaly": [],
            "agent_to_agent_closest_dists": [],
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
        state = self.reset_state()

        agent_positions = scenario.agent_positions
        agent_velocities = scenario.agent_velocities
        agent_types = scenario.agent_types
        agent_valid = scenario.agent_valid
        timestamps = scenario.timestamps
        conflict_points = scenario.map_conflict_points
        stationary_speed = scenario.stationary_speed

        # NOTE: this is not really an individual feature and would be useful for interactive features.
        state["agent_to_agent_closest_dists"] = np.linalg.norm(
            agent_positions[:, np.newaxis, :] - agent_positions[np.newaxis, :, :],
            axis=-1,
        ).min(axis=-1)

        # NOTE: Handling sequentially since each agent may have different valid masks which will
        # result in trajectories of different lengths.
        for n in range(scenario.num_agents):
            mask_n = agent_valid[n].squeeze(-1)
            if not mask_n.any() or mask_n.sum() < 2:
                # logger.warning(f"Agent {n} has no valid positions, skipping.")
                continue

            pos_n = agent_positions[n][mask_n, :]
            vel_n = agent_velocities[n][mask_n, :]
            type_n = agent_types[n]
            timestamps_n = timestamps[mask_n]

            # Compute agent features

            # Speed Profile
            speed_feature_n, speed_limit_diff_feature_n = individual.compute_speed(vel_n)
            if speed_feature_n is None or speed_limit_diff_feature_n is None:
                continue

            # Acceleration/Deceleration Profile
            # NOTE: acc and dec are accumulated abs acceleration and deceleration profiles.
            acceleration, acc_feature_n, dec_feature_n = individual.compute_acceleration_profile(
                speed_feature_n, timestamps_n
            )
            if acc_feature_n is None or dec_feature_n is None:
                continue

            # Jerk Profile
            jerk_feature_n = individual.compute_jerk(speed_feature_n, timestamps_n)

            # Waiting period
            waiting_period_feature_n, waiting_intervals_feature_n, waiting_distances_feature_n = (
                individual.compute_waiting_period(
                    pos_n,
                    speed_feature_n,
                    timestamps_n,
                    conflict_points,
                    stationary_speed,
                )
            )

            if self.return_criteria == "critical":
                speed_feature_n = speed_feature_n.max()
                speed_limit_diff_feature_n = speed_limit_diff_feature_n.max()
                acc_feature_n = acc_feature_n.max()
                dec_feature_n = dec_feature_n.max()
                jerk_feature_n = jerk_feature_n.max()
                waiting_period_feature_n = waiting_period_feature_n.max()
                waiting_intervals_feature_n = waiting_intervals_feature_n.max()
                waiting_distances_feature_n = waiting_distances_feature_n.min()

            elif self.return_criteria == "average":
                speed_feature_n = speed_feature_n.mean()
                speed_limit_diff_feature_n = speed_limit_diff_feature_n.mean()
                acc_feature_n = acc_feature_n.mean()
                dec_feature_n = dec_feature_n.mean()
                jerk_feature_n = jerk_feature_n.mean()
                waiting_period_feature_n = waiting_period_feature_n.mean()
                waiting_intervals_feature_n = waiting_intervals_feature_n.mean()
                waiting_distances_feature_n = waiting_distances_feature_n.mean()

            else:
                raise ValueError(f"Unknown return criteria: {self.return_criteria}")

            state["speed"].append(speed_feature_n)
            state["speed_limit_diff_feature_n"].append(speed_limit_diff_feature_n)
            state["acceleration"].append(acc_feature_n)
            state["deceleration"].append(dec_feature_n)
            state["jerk"].append(jerk_feature_n)
            state["waiting_period"].append(waiting_period_feature_n)
            state["waiting_intervals"].append(waiting_intervals_feature_n)
            state["waiting_distances"].append(waiting_distances_feature_n)
            state["agent_types"].append(type_n)
            state["valid_agents"].append(n)

        return state
