import numpy as np
from omegaconf import DictConfig

import characterization.features.individual_utils as individual
from characterization.features.base_feature import BaseFeature
from characterization.schemas import Individual, Scenario, ScenarioFeatures
from characterization.utils.common import MIN_VALID_POINTS, AgentTrajectoryMasker, ReturnCriterion
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


class IndividualFeatures(BaseFeature):
    """Computes individual agent features from scenario data.

    Attributes:
        config (DictConfig): Configuration parameters for feature computation.
        features (Any): Feature-specific configuration extracted from config.
        characterizer_type (str): Type identifier, always "feature".
        return_criterion (ReturnCriterion): Criterion for returning results (CRITICAL or AVERAGE).
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the IndividualFeatures extractor.

        Args:
            config (DictConfig): Configuration dictionary containing feature parameters.
                Expected keys:
                - return_criterion (str, optional): Determines whether to return 'critical'
                  (max/min values) or 'average' statistics for each feature. Defaults to 'critical'.
                - features (optional): Feature-specific configuration parameters.

        Note:
            The return_criterion is automatically converted to uppercase and mapped to
            the ReturnCriterion enum during initialization.
        """
        super().__init__(config)

    @staticmethod
    def compute_individual_features(scenario: Scenario, return_criterion: ReturnCriterion) -> Individual:
        """Compute individual motion features for all valid agents in a scenario.

        Args:
            scenario (Scenario): Complete scenario data containing:
                - agent_data: Agent positions, velocities, validity masks, and types
                - metadata: Timestamps, stationary speed threshold, and other parameters
                - static_map_data: Map conflict points for waiting behavior analysis
            return_criterion (ReturnCriterion): Determines feature aggregation method:
                - CRITICAL: Returns maximum values for most features, minimum for waiting_distance
                - AVERAGE: Returns mean values for all features

        Returns:
            Individual: Structured object containing computed features for valid agents:
                - valid_idxs: Indices of agents with sufficient valid data
                - agent_types: Agent type classifications (if available)
                - speed: Maximum/average speed values per agent
                - speed_limit_diff: Speed limit difference values per agent
                - acceleration: Maximum/average acceleration values per agent
                - deceleration: Maximum/average deceleration values per agent
                - jerk: Maximum/average jerk values per agent
                - waiting_period: Maximum/average waiting periods near conflict points
                - waiting_interval: Maximum/average waiting intervals between movements
                - waiting_distance: Minimum/average distances during waiting periods

        Raises:
            ValueError: If an unknown return_criterion is provided.
        """
        # Unpack senario fields
        agent_data = scenario.agent_data
        agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)

        agent_positions = agent_trajectories.agent_xyz_pos
        agent_velocities = agent_trajectories.agent_xy_vel
        agent_valid = agent_trajectories.agent_valid.squeeze(-1).astype(bool)

        metadata = scenario.metadata
        scenario_timestamps = metadata.timestamps_seconds
        stationary_speed = metadata.stationary_speed

        map_data = scenario.static_map_data
        conflict_points = map_data.map_conflict_points if map_data is not None else None

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
        for n in range(agent_data.num_agents):
            mask = agent_valid[n]
            if not mask.any() or mask.sum() < MIN_VALID_POINTS:
                continue

            velocities = agent_velocities[n][mask, :]
            positions = agent_positions[n][mask, :]
            timestamps = np.asarray(scenario_timestamps)[mask]

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

            match return_criterion:
                case ReturnCriterion.CRITICAL:
                    speed = speeds.max()
                    speed_limit_diff = speed_limit_diffs.max()
                    acceleration = accelerations.max()
                    deceleration = decelerations.max()
                    jerk = jerks.max() if jerks is not None else None
                    waiting_period = waiting_periods.max()
                    waiting_interval = waiting_intervals.max()
                    waiting_distance = waiting_distances.min()
                case ReturnCriterion.AVERAGE:
                    speed = speeds.mean()
                    speed_limit_diff = speed_limit_diffs.mean()
                    acceleration = accelerations.mean()
                    deceleration = decelerations.mean()
                    jerk = jerks.mean() if jerks is not None else None
                    waiting_period = waiting_periods.mean()
                    waiting_interval = waiting_intervals.mean()
                    waiting_distance = waiting_distances.mean()
                case _:
                    error_message = f"Unknown return criteria: {return_criterion}"
                    raise ValueError(error_message)

            scenario_valid_idxs.append(n)
            scenario_speeds.append(speed)
            scenario_speed_limit_diffs.append(speed_limit_diff)
            scenario_accelerations.append(acceleration)
            scenario_decelerations.append(deceleration)
            scenario_jerks.append(jerk)
            scenario_waiting_periods.append(waiting_period)
            scenario_waiting_intervals.append(waiting_interval)
            scenario_waiting_distances.append(waiting_distance)

        return Individual(
            valid_idxs=np.array(scenario_valid_idxs, dtype=np.int32) if scenario_valid_idxs else None,
            agent_types=agent_data.agent_types if agent_data.agent_types else None,
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
        )

    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Compute comprehensive scenario features including individual agent features.

        Args:
            scenario (Scenario): Complete scenario data containing:
                - agent_data: Agent positions, velocities, validity masks, and type information
                - metadata: Scenario timestamps, stationary speed thresholds, and other parameters
                - static_map_data: Map conflict points and road geometry information

        Returns:
            ScenarioFeatures: Comprehensive feature object containing:
                - metadata: Original scenario metadata for reference
                - individual_features: Individual agent motion features (speed, acceleration,
                  deceleration, jerk, waiting behaviors) computed using the configured
                  return criterion (critical or average values)
                - agent_to_agent_closest_dists: Minimum pairwise distances between all
                  agent pairs at their closest approach points

        Raises:
            ValueError: If an unknown return criterion is specified in the configuration.
        """
        # Unpack senario fields
        agent_data = scenario.agent_data
        agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
        agent_positions = agent_trajectories.agent_xyz_pos

        metadata = scenario.metadata

        # NOTE: this is not really an individual feature and would be useful for interactive features.
        agent_to_agent_closest_dists = (
            np.linalg.norm(agent_positions[:, np.newaxis, :] - agent_positions[np.newaxis, :, :], axis=-1)
            .min(axis=-1)
            .astype(np.float32)
        )

        return ScenarioFeatures(
            metadata=metadata,
            individual_features=IndividualFeatures.compute_individual_features(scenario, self.return_criterion),
            agent_to_agent_closest_dists=agent_to_agent_closest_dists,
        )
