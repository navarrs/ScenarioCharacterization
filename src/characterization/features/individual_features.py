import numpy as np
from omegaconf import DictConfig

import characterization.features.individual_utils as individual
from characterization.features.base_feature import BaseFeature
from characterization.utils.common import get_logger
from characterization.utils.schemas import Scenario, ScenarioFeatures

logger = get_logger(__name__)


class IndividualFeatures(BaseFeature):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the IndividualFeatures class.

        Args:
            config (DictConfig): Configuration for the feature. Expected to contain key-value pairs
                relevant to feature computation, such as thresholds or parameters. Must include
                'return_criteria' (str), which determines whether to return 'critical' or 'average'
                statistics for each feature.
        """
        super(IndividualFeatures, self).__init__(config)

        self.return_criteria = config.get("return_criteria", "critical")

    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Computes features for each agent in the scenario.

        Args:
            scenario (Scenario): Scenario object containing agent positions, velocities, validity masks,
                timestamps, map conflict points, stationary speed, and other scenario-level information.

        Returns:
            ScenarioFeatures: An object containing computed features for each valid agent, including
                speed, speed limit difference, acceleration, deceleration, jerk, waiting period,
                waiting interval, waiting distance, and agent-to-agent closest distances.

        Raises:
            ValueError: If an unknown return criteria is provided in the configuration.
        """
        agent_positions = scenario.agent_positions
        agent_velocities = scenario.agent_velocities
        agent_valid = scenario.agent_valid
        scenario_timestamps = scenario.timestamps
        conflict_points = scenario.map_conflict_points
        stationary_speed = scenario.stationary_speed

        # Meta information to be included within ScenarioFeatures. For an agent to be valid it needs to have at least
        # two valid timestamps. The indeces of such agents will be added to `valid_idxs` list.
        scenario_valid_idxs = []

        # Features to be included in ScenarioFeatures
        scenario_speeds = []
        scenario_speed_limit_diffs = []
        scenario_accelerations = []
        scenario_decelerations = []
        scenario_jerks = []
        scenario_waiting_periods = []
        scenario_waiting_intervals = []
        scenario_waiting_distances = []

        # NOTE: Handling sequentially since each agent may have different valid masks which will
        # result in trajectories of different lengths.
        for n in range(scenario.num_agents):
            mask = agent_valid[n].squeeze(-1)
            if not mask.any() or mask.sum() < 2:
                continue

            velocities = agent_velocities[n][mask, :]
            positions = agent_positions[n][mask, :]
            timestamps = scenario_timestamps[mask]

            # Compute agent features

            # Speed Profile
            speeds, speed_limit_diffs = individual.compute_speed(velocities)
            if speeds is None or speed_limit_diffs is None:
                continue

            # Acceleration/Deceleration Profile
            # NOTE: acc and dec are accumulated abs acceleration and deceleration profiles.
            _, accelerations, decelerations = individual.compute_acceleration_profile(speeds, timestamps)
            if accelerations is None or decelerations is None:
                continue

            # Jerk Profile
            jerks = individual.compute_jerk(speeds, timestamps)

            # Waiting period
            waiting_periods, waiting_intervals, waiting_distances = individual.compute_waiting_period(
                positions,
                speeds,
                timestamps,
                conflict_points,
                stationary_speed,
            )

            if self.return_criteria == "critical":
                speed = speeds.max()
                speed_limit_diff = speed_limit_diffs.max()
                acceleration = accelerations.max()
                deceleration = decelerations.max()
                jerk = jerks.max()
                waiting_period = waiting_periods.max()
                waiting_interval = waiting_intervals.max()
                waiting_distance = waiting_distances.min()

            elif self.return_criteria == "average":
                speed = speeds.mean()
                speed_limit_diff = speed_limit_diffs.mean()
                acceleration = accelerations.mean()
                deceleration = decelerations.mean()
                jerk = jerks.mean()
                waiting_period = waiting_periods.mean()
                waiting_interval = waiting_intervals.mean()
                waiting_distance = waiting_distances.mean()

            else:
                raise ValueError(f"Unknown return criteria: {self.return_criteria}")

            scenario_valid_idxs.append(n)
            scenario_speeds.append(speed)
            scenario_speed_limit_diffs.append(speed_limit_diff)
            scenario_accelerations.append(acceleration)
            scenario_decelerations.append(deceleration)
            scenario_jerks.append(jerk)
            scenario_waiting_periods.append(waiting_period)
            scenario_waiting_intervals.append(waiting_interval)
            scenario_waiting_distances.append(waiting_distance)

        # NOTE: this is not really an individual feature and would be useful for interactive features.
        agent_to_agent_closest_dists = (
            np.linalg.norm(agent_positions[:, np.newaxis, :] - agent_positions[np.newaxis, :, :], axis=-1)
            .min(axis=-1)
            .astype(np.float32)
        )

        return ScenarioFeatures(
            num_agents=scenario.num_agents,
            scenario_id=scenario.scenario_id,
            agent_types=scenario.agent_types,
            valid_idxs=np.array(scenario_valid_idxs, dtype=np.int32) if scenario_valid_idxs else None,
            speed=np.array(scenario_speeds, dtype=np.float32) if scenario_speeds else None,
            speed_limit_diff=(
                np.array(scenario_speed_limit_diffs, dtype=np.float32) if scenario_speed_limit_diffs else None
            ),
            acceleration=np.array(scenario_accelerations, dtype=np.float32) if scenario_accelerations else None,
            deceleration=np.array(scenario_decelerations, dtype=np.float32) if scenario_decelerations else None,
            jerk=np.array(scenario_jerks, dtype=np.float32) if scenario_jerks else None,
            waiting_period=np.array(scenario_waiting_periods, dtype=np.float32) if scenario_waiting_periods else None,
            waiting_interval=(
                np.array(scenario_waiting_intervals, dtype=np.float32) if scenario_waiting_intervals else None
            ),
            waiting_distance=(
                np.array(scenario_waiting_distances, dtype=np.float32) if scenario_waiting_distances else None
            ),
            agent_to_agent_closest_dists=agent_to_agent_closest_dists,
        )
