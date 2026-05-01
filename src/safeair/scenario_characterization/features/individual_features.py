"""Individual (per-agent) feature extractor for SafeAir scenarios."""

from characterization.domains.aviation.scenario_types import AgentTrajectory
from safeair.scenario_characterization.common import (
    ReturnCriterion,
    ValueClipper,
    get_conflict_points_from_scenario,
    raw_to_agent_type,
)
from safeair.scenario_characterization.features import individual_utils as indv
from safeair.scenario_characterization.features.base_feature import BaseFeature
from safeair.schemas.scenario import Scenario
from safeair.schemas.scenario_features import CharacterizationParameters, IndividualAgentFeatures, ScenarioFeatures

_MIN_VALID_FRAMES = 2
_DEFAULT_CHARACTERIZATION = CharacterizationParameters()


class IndividualFeatures(BaseFeature):
    """Computes kinematic features for every agent in a scenario.

    Features computed per agent:
    - ``speed``: peak or mean smoothed speed in m/s, depending on ``return_criterion``.
    - ``acceleration``: peak or mean positive acceleration in m/s².
    - ``deceleration``: peak or mean deceleration magnitude in m/s².
    - ``waiting_period``: total seconds near a conflict point while stationary.
    - ``trajectory_type``: STATIONARY / STRAIGHT / TURNING / U_TURN.
    - ``kalman_difficulty``: normalised Kalman prediction error.

    Unit-conversion factors are derived automatically from ``scenario.metadata`` at compute time.

    Args:
        return_criterion: How to reduce per-timestep values to a scalar. ``CRITICAL`` returns the
            peak (most extreme) value; ``AVERAGE`` returns the mean.
        characterization: Thresholds controlling feature computation. Defaults to the standard SafeAir values.
    """

    def __init__(
        self,
        return_criterion: ReturnCriterion,
        *,
        characterization: CharacterizationParameters = _DEFAULT_CHARACTERIZATION,
    ) -> None:
        """Intitialize with the aggregation criterion and characterization parameters."""
        super().__init__(return_criterion, characterization=characterization)
        self.kalman_value_clipper = ValueClipper(min=0.0, max=50.0)

    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Compute individual features for all agents in the scenario.

        Args:
            scenario: The scenario to characterize.

        Returns:
            ScenarioFeatures with ``individual_features`` populated and ``interaction_features`` empty.
        """
        self._set_unit_factors(scenario)
        features = [self._compute_agent_features(scenario, idx) for idx in range(scenario.agent_data.num_agents)]
        return ScenarioFeatures(
            scenario_id=scenario.metadata.scenario_id,
            individual_features=features,
            interaction_features=[],
        )

    def _compute_agent_features(self, scenario: Scenario, agent_index: int) -> IndividualAgentFeatures:
        """Compute all individual features for a single agent.

        Args:
            scenario: The scenario containing the agent.
            agent_index: Index of the agent in ``scenario.agent_data``.

        Returns:
            IndividualAgentFeatures for the agent. Feature values are ``None`` when not computable.
        """
        agent_id = int(scenario.agent_data.agent_ids[agent_index])
        agent_type = raw_to_agent_type(scenario.agent_data.agent_types[agent_index]).name
        agent_trajectory = AgentTrajectory(scenario.agent_data.agent_trajectories[agent_index])

        # Check if the trajectory has enough valid frames to compute feature; if not, return an empty feature.
        valid_mask = agent_trajectory.get_valid_mask()
        if valid_mask.sum() < _MIN_VALID_FRAMES:
            return IndividualAgentFeatures(agent_id=agent_id, agent_type=agent_type)

        speeds_ms = agent_trajectory.get_valid_speeds(speed_to_ms=self._speed_to_ms)
        positions = agent_trajectory.get_valid_positions(scale=self._scale_to_m)
        timestamps = indv.get_valid_timestamps(scenario.metadata.timestamps_seconds, valid_mask)

        # Basic kinematic features
        speed = indv.compute_speed_profile(speeds_ms, self.return_criterion)
        assert isinstance(speed, float)
        acceleration, deceleration = indv.compute_acceleration_profile(speeds_ms, timestamps, self.return_criterion)
        assert isinstance(acceleration, float)
        assert isinstance(deceleration, float)

        # Conflict points from map data (hold-short lines), if available, for waiting period calculation.
        conflict_points = get_conflict_points_from_scenario(scenario, scale=self._scale_to_m)
        waiting_period = None
        if conflict_points is not None:
            waiting_period = indv.compute_waiting_period(
                positions,
                timestamps,
                speeds_ms,
                conflict_points,
                self._characterization.max_stationary_speed,
                self._characterization.agent_to_conflict_point_max_distance,
                return_criterion=self.return_criterion,
            )
            assert isinstance(waiting_period, float)

        trajectory_type = indv.classify_trajectory_type(
            positions,
            speeds_ms,
            config=self._characterization,
        )

        current_time_index = scenario.metadata.current_time_index
        kalman_difficulty: float | None = None
        if current_time_index is not None:
            kalman_difficulty = indv.compute_kalman_difficulty(
                agent_trajectory,
                current_time_index,
                scale_to_m=self._scale_to_m,
                value_clipper=self.kalman_value_clipper,
            )

        return IndividualAgentFeatures(
            agent_id=agent_id,
            agent_type=agent_type,
            speed=speed,
            acceleration=acceleration,
            deceleration=deceleration,
            waiting_period=waiting_period,
            trajectory_type=trajectory_type,
            kalman_difficulty=kalman_difficulty,
        )
