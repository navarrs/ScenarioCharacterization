import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

import characterization.features.individual_utils as individual
from characterization.domains.ad.features.base_feature import ADBaseFeature
from characterization.domains.ad.features.individual_utils import compute_trajectory_type
from characterization.domains.ad.scenario_types import AgentTrajectoryMasker, AgentType, LaneMasker
from characterization.schemas import IndividualAgentFeatures, Scenario, ScenarioFeatures
from characterization.utils.common import FeatureType, ReturnCriterion, categorize_from_thresholds
from characterization.utils.constants import MIN_VALID_POINTS
from characterization.utils.geometric_utils import compute_agent_to_agent_closest_dists
from characterization.utils.logging_utils import get_pylogger

logger = get_pylogger(__name__)


class IndividualFeatures(ADBaseFeature):
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

        self.categorize_features = FeatureType(self.config.get("feature_type", "continuous")) == FeatureType.CATEGORICAL
        if self.categorize_features:
            vehicle_file = Path(self.config.get("vehicle_categorization_file", ""))
            if not vehicle_file.is_file():
                msg = f"Vehicle categorization file {vehicle_file} does not exist."
                raise FileNotFoundError(msg)
            with vehicle_file.open("r") as f:
                self.vehicle_categories = json.load(f)

            cyclist_file = Path(self.config.get("cyclist_categorization_file", ""))
            if not cyclist_file.is_file():
                msg = f"Cyclist categorization file {cyclist_file} does not exist."
                raise FileNotFoundError(msg)
            with cyclist_file.open("r") as f:
                self.cyclist_categories = json.load(f)

            pedestrian_file = Path(self.config.get("pedestrian_categorization_file", ""))
            if not pedestrian_file.is_file():
                msg = f"Pedestrian categorization file {pedestrian_file} does not exist."
                raise FileNotFoundError(msg)
            with pedestrian_file.open("r") as f:
                self.pedestrian_categories = json.load(f)

    def categorize(self, value: float, agent_type: AgentType, feature_name: str) -> float:
        """Categorize a feature value based on agent type and predefined percentiles.

        Args:
            value (float): The feature value to categorize.
            agent_type (AgentType): The type of the agent (VEHICLE, CYCLIST, PEDESTRIAN).
            feature_name (str): The name of the feature being categorized.

        Returns:
            float: The categorized feature value based on percentiles.
        """
        match agent_type:
            case AgentType.TYPE_VEHICLE | AgentType.TYPE_EGO_AGENT:
                categories = self.vehicle_categories.get(feature_name, None)
            case AgentType.TYPE_CYCLIST:
                categories = self.cyclist_categories.get(feature_name, None)
            case AgentType.TYPE_PEDESTRIAN:
                categories = self.pedestrian_categories.get(feature_name, None)
            case _:
                logger.warning("Unknown agent type: %s", agent_type)
                return -1.0

        if categories is None:
            logger.error("No categories found for feature %s and agent type %s.", feature_name, agent_type)
            error_message = f"No categories found for feature {feature_name} and agent type {agent_type}."
            raise ValueError(error_message)

        threshold_values = list(categories.values())
        return float(categorize_from_thresholds(value, threshold_values))

    def compute_individual_features(self, scenario: Scenario) -> list[IndividualAgentFeatures]:
        """Compute individual motion features for all valid agents in a scenario.

        Args:
            scenario (Scenario): Complete scenario data containing:
                - agent_data: Agent positions, velocities, validity masks, and types
                - metadata: Timestamps, stationary speed threshold, and other parameters
                - static_map_data: Map conflict points for waiting behavior analysis

        Returns:
            List of :class:`IndividualAgentFeatures`, one entry per valid agent (agents with
            insufficient valid timesteps are omitted). Each entry carries the agent's
            scenario index in ``agent_id``.

        Raises:
            ValueError: If an unknown return_criterion is provided.
        """
        # Unpack scenario fields
        agent_data = scenario.agent_data
        agent_types = agent_data.agent_types
        agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)

        agent_positions = agent_trajectories.agent_xyz_pos
        agent_velocities = agent_trajectories.agent_xy_vel
        agent_headings = np.rad2deg(agent_trajectories.agent_headings)
        agent_valid = agent_trajectories.agent_valid.squeeze(-1).astype(bool)

        metadata = scenario.metadata
        scenario_timestamps = metadata.timestamps_seconds
        stationary_speed = metadata.max_stationary_speed
        current_time_index = metadata.current_time_index

        map_data = scenario.static_map_data
        conflict_points, closest_lanes, lane_speed_limits = None, None, None
        if map_data is not None:
            conflict_points = map_data.map_conflict_points
            closest_lanes = map_data.agent_closest_lanes
            lane_speed_limits = map_data.lane_speed_limits_mph

        result: list[IndividualAgentFeatures] = []

        # NOTE: Handling sequentially since each agent may have different valid masks which will
        # result in trajectories of different lengths.
        for n in range(agent_data.num_agents):
            mask = agent_valid[n]
            if not mask.any() or mask.sum() < MIN_VALID_POINTS:
                continue

            agent_type = agent_types[n]
            velocities = agent_velocities[n][mask, :]
            positions = agent_positions[n][mask, :]
            headings = agent_headings[n][mask]
            timestamps = np.asarray(scenario_timestamps)[mask]

            # Speed Profile
            closest_lane_n = LaneMasker(closest_lanes[n, mask]) if closest_lanes is not None else None
            speeds, speed_limit_diffs = individual.compute_speed_meta(velocities, closest_lane_n, lane_speed_limits)
            if speeds is None or speed_limit_diffs is None:
                continue

            speed = individual.compute_speed_profile(speeds, self.return_criterion)
            assert isinstance(speed, float)

            match self.return_criterion:
                case ReturnCriterion.CRITICAL:
                    speed_limit_diff = float(speed_limit_diffs.max())
                case ReturnCriterion.AVERAGE:
                    speed_limit_diff = float(speed_limit_diffs.mean())
                case _:
                    error_message = f"Unknown return criteria: {self.return_criterion}"
                    raise ValueError(error_message)

            # Acceleration/Deceleration Profile
            acceleration, deceleration = individual.compute_acceleration_profile(
                speeds,
                timestamps,
                self.return_criterion,
            )
            if not isinstance(acceleration, float) or not isinstance(deceleration, float):
                continue

            # Jerk Profile
            # NOTE: Jerk values are not smoothed, nor they consider differences in the x, y components of velocity,
            # nor vehicle dynamics. This can lead to high jerk values for noisy data.
            jerks = individual.compute_jerk(speeds, timestamps)
            jerk: float | None = None
            if jerks is not None:
                jerk = float(jerks.max() if self.return_criterion == ReturnCriterion.CRITICAL else jerks.mean())

            # Waiting period
            empty_conflict_points = np.empty((0, 3), dtype=np.float32)
            waiting_period = individual.compute_waiting_period(
                positions,
                timestamps,
                speeds,
                conflict_points if conflict_points is not None else empty_conflict_points,
                stationary_speed,
                metadata.agent_to_conflict_point_max_distance,
                return_criterion=self.return_criterion,
            )
            if not isinstance(waiting_period, float) or np.isnan(waiting_period):
                waiting_period = 0.0

            # Trajectory Type
            trajectory_type = compute_trajectory_type(positions, speeds, headings, metadata)

            # Kalman Difficulty
            kalman_difficulty = individual.compute_kalman_difficulty(agent_positions[n], mask, current_time_index + 1)

            if self.categorize_features:
                speed = self.categorize(speed, agent_type, "speed")
                speed_limit_diff = self.categorize(speed_limit_diff, agent_type, "speed_limit_diff")
                acceleration = self.categorize(acceleration, agent_type, "acceleration")
                deceleration = self.categorize(deceleration, agent_type, "deceleration")
                jerk = self.categorize(jerk, agent_type, "jerk") if jerk is not None else 0.0
                waiting_period = self.categorize(waiting_period, agent_type, "waiting_period")
                kalman_difficulty = (
                    self.categorize(kalman_difficulty, agent_type, "kalman_difficulty")
                    if kalman_difficulty >= 0
                    else 0.0
                )

            result.append(
                IndividualAgentFeatures(
                    agent_id=n,
                    agent_type=agent_type.name,
                    speed=speed,
                    speed_limit_diff=speed_limit_diff,
                    acceleration=acceleration,
                    deceleration=deceleration,
                    jerk=jerk,
                    waiting_period=waiting_period,
                    trajectory_type=trajectory_type.name,
                    kalman_difficulty=kalman_difficulty if kalman_difficulty >= 0 else None,
                )
            )

        return result

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
        agent_to_agent_closest_dists = None
        if self.compute_agent_to_agent_closest_dists:
            agent_data = scenario.agent_data
            agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
            agent_positions = agent_trajectories.agent_xyz_pos
            agent_to_agent_closest_dists = compute_agent_to_agent_closest_dists(agent_positions)

        return ScenarioFeatures(
            metadata=scenario.metadata,
            individual_features=self.compute_individual_features(scenario),
            agent_to_agent_closest_dists=agent_to_agent_closest_dists,
        )
