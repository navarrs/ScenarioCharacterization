import numpy as np
from omegaconf import DictConfig

import src.features.individual_utils as individual
from src.features.base_feature import BaseFeature
from src.utils.common import get_logger

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
            "speed_limit_diff": [],
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

    def compute(self, scenario: dict) -> dict:
        """Computes features for each agent in the scenario.

        Args:
            scenario (Dict): A dictionary containing scenario data.

        Returns:
            Dict: A dictionary with computed features for each agent.

        Raises:
            ValueError: If the 'scenario' dictionary does not contain the key 'num_agents'.
        """
        if not scenario.get("num_agents", None):
            raise ValueError(
                "The 'scenario' dictionary must contain the key 'num_agents'."
            )

        state = self.reset_state()

        agent_positions = scenario["agent_positions"]
        # NOTE: this is not really an individual feature and would be useful for interactive features.
        state["agent_to_agent_closest_dists"] = np.linalg.norm(
            agent_positions[:, np.newaxis, :] - agent_positions[np.newaxis, :, :],
            axis=-1,
        ).min(axis=-1)

        N = scenario["num_agents"]

        # NOTE: Handling sequentially since each agent may have different valid masks which will
        # result in trajectories of different lengths.
        conflict_points = scenario["map_conflict_points"]
        stationary_speed = scenario["stationary_speed"]
        for n in range(N):
            agent_mask = scenario["agent_valid"][n].squeeze(-1)
            if not agent_mask.any() or agent_mask.sum() < 2:
                # logger.warning(f"Agent {n} has no valid positions, skipping.")
                continue

            agent_positions = scenario["agent_positions"][n][agent_mask, :]
            agent_velocities = scenario["agent_velocities"][n][agent_mask, :]
            agent_headings = scenario["agent_headings"][n][agent_mask, :]
            agent_type = scenario["agent_types"][n]
            timestamps = scenario["timestamps"][agent_mask]

            # Speed Profile
            speed, speed_limit_diff = individual.compute_speed(agent_velocities)
            if speed is None:
                continue

            # Acceleration/Deceleration Profile
            # NOTE: acc and dec are accumulated abs acceleration and deceleration profiles.
            acceleration, acc, dec = individual.compute_acceleration_profile(
                speed, timestamps
            )

            # Jerk Profile
            jerk = individual.compute_jerk(speed, timestamps)

            waiting_period, waiting_intervals, waiting_distances = (
                individual.compute_waiting_period(
                    agent_positions,
                    speed,
                    timestamps,
                    conflict_points,
                    stationary_speed,
                )
            )

            if self.return_criteria == "critical":
                speed, speed_limit_diff = speed.max(), speed_limit_diff.max()
                acc, dec, jerk = acc.max(), dec.max(), jerk.max()
                waiting_period, waiting_intervals = (
                    waiting_period.max(),
                    waiting_intervals.max(),
                )
                waiting_distances = waiting_distances.min()

            elif self.return_criteria == "average":
                speed, speed_limit_diff = speed.mean(), speed_limit_diff.mean()
                acc, dec, jerk = acc.mean(), dec.mean(), jerk.mean()
                waiting_period, waiting_intervals = (
                    waiting_period.mean(),
                    waiting_intervals.mean(),
                )
                waiting_distances = waiting_distances.mean()

            state["speed"].append(speed)
            state["speed_limit_diff"].append(speed_limit_diff)
            state["acceleration"].append(acc)
            state["deceleration"].append(dec)
            state["jerk"].append(jerk)
            state["waiting_period"].append(waiting_period)
            state["waiting_intervals"].append(waiting_intervals)
            state["waiting_distances"].append(waiting_distances)
            state["agent_types"].append(agent_type)
            state["valid_agents"].append(n)

        return state
