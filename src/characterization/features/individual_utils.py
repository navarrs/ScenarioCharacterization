import numpy as np
from numpy.typing import NDArray

from characterization.schemas import ScenarioMetadata
from characterization.utils.common import MIN_VALID_POINTS, SMALL_EPS, LaneMasker, TrajectoryType, mph_to_ms
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


def compute_speed_meta(
    velocities: NDArray[np.float32], closest_lanes: LaneMasker | None, lane_speed_limits: NDArray[np.float32] | None
) -> tuple[NDArray[np.float32] | None, ...]:
    """Computes the speed profile of an agent.

    Args:
        velocities (np.ndarray): The velocity vectors of the agent over time (shape: [T, D]).
        closest_lanes (np.ndarray or None): closest lanes information (shape: [T, K, 6]) or None.
        lane_speed_limits (np.ndarray or None): Speed limits for each lane (shape: [K,]) or None.

    Returns:
        tuple:
            speeds (np.ndarray or None): The speed time series (shape: [T,]), or None if NaN values are present.
            speeds_limit_diff (np.ndarray or None): The difference between speed and speed limit (currently zeros),
                or None if NaN values are present.
    """
    speeds = np.linalg.norm(velocities, axis=-1)
    if np.isnan(speeds).any():
        logger.warning("Nan value in agent speed: %s", speeds)
        return None, None, None

    # If lane information is provided, compute the difference between the agent's speed and the speed limit
    speeds_limit_diff = np.zeros_like(speeds, dtype=np.float32)
    if closest_lanes is not None and lane_speed_limits is not None:
        # closest_lane_dist_and_idx shape: (T, K)
        k_closest_lane_idx = closest_lanes.lane_idx.squeeze(-1)  # shape: (T, K)
        k_speed_limits = mph_to_ms(lane_speed_limits[k_closest_lane_idx])  # shape: (T, K)
        speeds_limit_diff = np.abs(speeds[:, None] - k_speed_limits).mean(axis=-1)  # shape: (T,)

    return speeds, speeds_limit_diff


def compute_acceleration_profile(speed: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray | None, ...]:
    """Computes the acceleration profile from the speed (m/s) and time delta.

    Args:
        speed (np.ndarray): The speed time series (m/s) (shape: [T,]).
        timestamps (np.ndarray): The timestamps corresponding to each speed measurement (shape: [T,]).

    Returns:
        tuple:
            acceleration_raw (np.ndarray or None): The raw acceleration time series (shape: [T,]), or None if NaN values
                are present.
            acceleration (np.ndarray or None): The sum of positive acceleration intervals, or None if NaN values are
                present.
            deceleration (np.ndarray or None): The sum of negative acceleration intervals (absolute value), or None if
                NaN values are present.

    Raises:
        ValueError: If speed and timestamps do not have the same shape.
    """

    def get_acc_sums(acc: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
        diff = idx[1:] - idx[:-1]
        diff = np.array([-1] + np.where(diff > 1)[0].tolist() + [diff.shape[0]])  # noqa: RUF005
        se_idxs = [(idx[s + 1], idx[e] + 1) for s, e in zip(diff[:-1], diff[1:], strict=False)]  # noqa: RUF007
        sums = np.array([acc[s:e].sum() for s, e in se_idxs])
        return sums, se_idxs  # pyright: ignore[reportReturnType]

    if speed.shape != timestamps.shape:
        error_message = "Speed and timestamps must have the same shape."
        raise ValueError(error_message)

    acceleration_raw = np.gradient(speed, timestamps)  # m/s^2

    if np.isnan(acceleration_raw).any():
        logger.warning("Nan value in agent acceleration: %s", acceleration_raw)
        return None, None, None

    dr_idx = np.where(acceleration_raw < 0.0)[0]

    # If the agent is accelerating or maintaining acceleration
    if dr_idx.shape[0] == 0:
        deceleration = np.zeros(shape=(1,))
        acceleration = acceleration_raw.copy()
    # If the agent is decelerating
    elif dr_idx.shape[0] == acceleration_raw.shape[0]:
        deceleration = acceleration_raw.copy()
        acceleration = np.zeros(shape=(1,))
    # If both
    else:
        deceleration, _ = get_acc_sums(acceleration_raw, dr_idx)

        ar_idx = np.where(acceleration_raw >= 0.0)[0]
        acceleration, _ = get_acc_sums(acceleration_raw, ar_idx)

    return acceleration_raw, acceleration, np.abs(deceleration)


def compute_jerk(speed: np.ndarray, timestamps: np.ndarray) -> np.ndarray | None:
    """Computes the jerk from the acceleration profile and time delta.

    Args:
        speed (np.ndarray): The speed time series (m/s) (shape: [T,]).
        timestamps (np.ndarray): The timestamps corresponding to each speed measurement (shape: [T,]).

    Returns:
        np.ndarray or None: The jerk time series (m/s^3), or None if NaN values are present.

    Raises:
        ValueError: If speed and timestamps do not have the same shape.
    """
    if speed.shape != timestamps.shape:
        error_message = "Speed and timestamps must have the same shape."
        raise ValueError(error_message)

    acceleration = np.gradient(speed, timestamps)
    jerk = np.gradient(acceleration, timestamps)

    if np.isnan(jerk).any():
        logger.warning("Nan value in agent jerk: %s", jerk)
        return None

    return jerk


def compute_waiting_period(
    position: np.ndarray,
    speed: np.ndarray,
    timestamps: np.ndarray,
    conflict_points: np.ndarray | None,
    stationary_speed: float = 0.0,
) -> tuple[np.ndarray, ...]:
    """Computes the waiting period for an agent based on its position and speed.

    Args:
        position (np.ndarray): The positions of the agent over time (shape: [T, 2]).
        speed (np.ndarray): The speeds of the agent over time (shape: [T,]).
        timestamps (np.ndarray): The timestamps corresponding to each position/speed (shape: [T,]).
        conflict_points (np.ndarray or None): The conflict points to check against (shape: [C, 2] or None).
        stationary_speed (float, optional): The speed threshold below which the agent is considered stationary. Defaults
            to 0.0.

    Returns:
        tuple:
            waiting_period (np.ndarray): The waiting interval over the distance to the closest conflict point at that
                distance (shape: [N,]).
            waiting_intervals (np.ndarray): The duration of each waiting interval (shape: [N,]).
            waiting_distances (np.ndarray): The minimum distance to conflict points during each waiting interval
                (shape: [N,]).
    """
    waiting_intervals = np.zeros(shape=(position.shape[0]))
    waiting_distances = np.inf * np.ones(shape=(position.shape[0]))
    waiting_period = np.zeros(shape=(position.shape[0]))
    if conflict_points is None or conflict_points.shape[0] == 0:
        return waiting_period, waiting_intervals, waiting_distances

    dt = timestamps[1:] - timestamps[:-1]
    # On a per-timestep basis, this considers an agent to be waiting if its speed is less than or
    # equal to the predefined stationary speed.
    is_waiting = speed <= stationary_speed
    if sum(is_waiting) > 0:
        # Find all the transitions between moving and being stationary
        is_waiting = np.hstack([[False], is_waiting, [False]])
        is_waiting = np.diff(is_waiting.astype(int))

        # Get all intervals where the agent is waiting
        starts = np.where(is_waiting == 1)[0]
        ends = np.where(is_waiting == -1)[0]

        waiting_intervals = np.array([dt[start:end].sum() for start, end in zip(starts, ends, strict=False)])
        # intervals = np.array([end - start for start, end in zip(starts, ends)])

        # For every timestep, get the minimum distance to the set of conflict points
        waiting_distances = np.linalg.norm(conflict_points[:, None] - position[starts], axis=-1).min(axis=0)

        # TODO:
        # # Get the index of the longest interval. Then, get the longest interval and the distance to
        # # the closest conflict point at that interval
        # idx = intervals.argmax()
        # # breakpoint()
        # waiting_period_interval_longest = intervals[idx]
        # waiting_period_distance_longest = dists_cps[idx] + SMALL_EPS

        # # Get the index of the closest conflict point for each interval. Then get the interval for
        # # that index and the distance to that conflict point
        # idx = dists_cps.argmin()
        # waiting_period_interval_closest_conflict = intervals[idx]
        # waiting_period_distance_closest_conflict = dists_cps[idx] + SMALL_EPS

    # waiting_intervals = np.asarray(
    #     [waiting_period_interval_longest, waiting_period_interval_closest_conflict])
    # waiting_distances_to_conflict = np.asarray(
    #     [waiting_period_distance_longest, waiting_period_distance_closest_conflict])

    waiting_period = waiting_intervals / (waiting_distances + SMALL_EPS)
    return waiting_period, waiting_intervals, waiting_distances


def _rotate_to_local_coordinates(
    displacement_vector: NDArray[np.float32], start_heading: np.float32
) -> NDArray[np.float32]:
    """Rotate displacement vector to agent's local coordinate system."""
    rotation_matrix = np.array(
        [[np.cos(-start_heading), -np.sin(-start_heading)], [np.sin(-start_heading), np.cos(-start_heading)]]
    )
    return np.dot(rotation_matrix.squeeze(-1), displacement_vector)


def _is_stationary(
    max_speed: np.float32, displacement: np.float32, max_stationary_speed: float, max_stationary_displacement: float
) -> bool:
    """Check if trajectory represents a stationary agent."""
    return bool(max_speed < max_stationary_speed and displacement < max_stationary_displacement)


def _is_straight_trajectory(heading_change: np.float32, max_straight_absolute_heading_diff: float) -> bool:
    """Check if trajectory is generally straight (small heading change)."""
    return bool(np.abs(heading_change) < max_straight_absolute_heading_diff)


def _classify_straight_trajectory(
    lateral_displacement: np.float32, max_straight_lateral_displacement: float
) -> TrajectoryType:
    """Classify straight trajectory based on lateral displacement."""
    if np.abs(lateral_displacement) < max_straight_lateral_displacement:
        return TrajectoryType.TYPE_STRAIGHT
    return TrajectoryType.TYPE_STRAIGHT_RIGHT if lateral_displacement < 0 else TrajectoryType.TYPE_STRAIGHT_LEFT


def _is_right_turn(
    heading_change: np.float32, lateral_displacement: np.float32, max_straight_absolute_heading_diff: float
) -> bool:
    """Check if trajectory represents a right turn."""
    return bool(heading_change < -max_straight_absolute_heading_diff and lateral_displacement < 0)


def _classify_right_trajectory(
    longitudinal_displacement: np.float32, min_uturn_longitudinal_displacement: float
) -> TrajectoryType:
    """Classify right turn trajectory (turn vs U-turn)."""
    if longitudinal_displacement < min_uturn_longitudinal_displacement:
        return TrajectoryType.TYPE_RIGHT_U_TURN
    return TrajectoryType.TYPE_RIGHT_TURN


def _classify_left_trajectory(
    longitudinal_displacement: np.float32, min_uturn_longitudinal_displacement: float
) -> TrajectoryType:
    """Classify left turn trajectory (turn vs U-turn)."""
    if longitudinal_displacement < min_uturn_longitudinal_displacement:
        return TrajectoryType.TYPE_LEFT_U_TURN
    return TrajectoryType.TYPE_LEFT_TURN


def compute_trajectory_type(
    positions: NDArray[np.float32],
    speeds: NDArray[np.float32],
    headings: NDArray[np.float32],
    metadata: ScenarioMetadata,
) -> TrajectoryType:
    """Classify trajectory type based on movement patterns and geometry.

    The classification strategy is adapted from waymo_open_dataset/metrics/motion_metrics_utils.cc#L28
    and UniTraj: https://github.com/vita-epfl/UniTraj/blob/main/unitraj/datasets/common_utils.py#L395

    Args:
        positions: Agent positions over time with shape [T, 3].
        speeds: Agent speeds over time with shape [T].
        headings: Agent headings over time with shape [T].
        metadata: Scenario metadata containing classification thresholds.

    Returns:
        The classified trajectory type.
    """
    # Calculate basic trajectory metrics
    start_point, end_point = positions[0], positions[-1]
    displacement_vector = end_point[:2] - start_point[:2]  # Only use x, y components
    final_displacement = np.linalg.norm(displacement_vector)

    # Calculate heading change
    start_heading, end_heading = headings[0], headings[-1]
    heading_change = end_heading - start_heading

    # Transform displacement to agent's local coordinate system (relative to start heading)
    local_displacement = _rotate_to_local_coordinates(displacement_vector, np.deg2rad(start_heading))
    longitudinal_displacement, lateral_displacement = local_displacement  # dx, dy

    # Get speed metrics
    start_speed, end_speed = speeds[0], speeds[-1]
    max_endpoint_speed = max(start_speed, end_speed)

    # Classification logic
    max_stationary_speed = metadata.max_stationary_speed
    max_stationary_displacement = metadata.max_stationary_displacement
    if _is_stationary(max_endpoint_speed, final_displacement, max_stationary_speed, max_stationary_displacement):
        return TrajectoryType.TYPE_STATIONARY

    max_straight_absolute_heading_diff = metadata.max_straight_absolute_heading_diff
    max_straight_lateral_displacement = metadata.max_straight_lateral_displacement
    if _is_straight_trajectory(heading_change, max_straight_absolute_heading_diff):
        return _classify_straight_trajectory(lateral_displacement, max_straight_lateral_displacement)

    min_uturn_longitudinal_displacement = metadata.min_uturn_longitudinal_displacement
    if _is_right_turn(heading_change, lateral_displacement, max_straight_absolute_heading_diff):
        return _classify_right_trajectory(longitudinal_displacement, min_uturn_longitudinal_displacement)

    # Default to left turn classification
    return _classify_left_trajectory(longitudinal_displacement, min_uturn_longitudinal_displacement)


def _compute_average_velocity(positions: NDArray[np.float32]) -> float:
    """Compute average velocity from position differences."""
    velocity_sum = 0.0
    for i in range(len(positions) - 1):
        velocity_sum += positions[i + 1] - positions[i]
    return velocity_sum / (len(positions) - 1)


class KalmanState:
    """Container for Kalman filter state variables."""

    def __init__(self, size: int) -> None:
        """Initialize KalmanState with position arrays.

        Args:
            size (int): The number of time steps to allocate for the state.
        """
        self.x = np.zeros(size + 1, dtype=np.float32)  # Position estimates (x)
        self.y = np.zeros(size + 1, dtype=np.float32)  # Position estimates (y)


class KalmanCovariance:
    """Container for Kalman filter covariance matrices."""

    def __init__(self, size: int) -> None:
        """Initialize KalmanCovariance with covariance arrays.

        Args:
            size (int): The number of time steps to allocate for the covariance.
        """
        self.pos_x = np.zeros(size + 1, dtype=np.float32)  # Position uncertainty (x)
        self.pos_y = np.zeros(size + 1, dtype=np.float32)  # Position uncertainty (y)
        self.vel_x = np.zeros(size + 1, dtype=np.float32)  # Velocity uncertainty (x)
        self.vel_y = np.zeros(size + 1, dtype=np.float32)  # Velocity uncertainty (y)


class KalmanGains:
    """Container for Kalman gain matrices."""

    def __init__(self, size: int) -> None:
        """Initialize KalmanGains with gain arrays.

        Args:
            size (int): The number of time steps to allocate for the gains.
        """
        self.pos_x = np.zeros(size + 1, dtype=np.float32)  # Kalman gain for position (x)
        self.pos_y = np.zeros(size + 1, dtype=np.float32)  # Kalman gain for position (y)
        self.vel_x = np.zeros(size + 1, dtype=np.float32)  # Kalman gain for velocity (x)
        self.vel_y = np.zeros(size + 1, dtype=np.float32)  # Kalman gain for velocity (y)


def _initialize_kalman_state(size: int, initial_x: float, initial_y: float) -> KalmanState:
    """Initialize Kalman filter state with initial position."""
    state = KalmanState(size)
    state.x[0] = initial_x
    state.y[0] = initial_y
    return state


def _initialize_kalman_covariance(size: int) -> KalmanCovariance:
    """Initialize Kalman filter covariance matrices with unit uncertainty."""
    covariance = KalmanCovariance(size)
    # Initialize with unit Gaussian uncertainty
    covariance.pos_x[0] = 1.0
    covariance.pos_y[0] = 1.0
    covariance.vel_x[0] = 1.0
    covariance.vel_y[0] = 1.0
    return covariance


def _initialize_kalman_gains(size: int) -> KalmanGains:
    """Initialize Kalman gain matrices."""
    return KalmanGains(size)


def _kalman_prediction_step(
    state: KalmanState,
    covariance: KalmanCovariance,
    timestep: int,
    velocity_x: float,
    velocity_y: float,
    process_noise: float,
) -> None:
    """Perform Kalman filter prediction step (time update)."""
    k = timestep

    # Predict next state using constant velocity model
    state.x[k + 1] = state.x[k] + velocity_x
    state.y[k + 1] = state.y[k] + velocity_y

    # Update covariance matrices (add process noise)
    covariance.pos_x[k + 1] = covariance.pos_x[k] + covariance.vel_x[k] + process_noise
    covariance.pos_y[k + 1] = covariance.pos_y[k] + covariance.vel_y[k] + process_noise
    covariance.vel_x[k + 1] = covariance.vel_x[k] + process_noise
    covariance.vel_y[k + 1] = covariance.vel_y[k] + process_noise


def _kalman_correction_step(
    state: KalmanState,
    covariance: KalmanCovariance,
    gains: KalmanGains,
    timestep: int,
    observed_x: NDArray[np.float32],
    observed_y: NDArray[np.float32],
    measurement_noise: float,
) -> None:
    """Perform Kalman filter correction step (measurement update)."""
    k = timestep + 1

    # Calculate Kalman gains
    gains.pos_x[k] = covariance.pos_x[k] / (covariance.pos_x[k] + measurement_noise)
    gains.pos_y[k] = covariance.pos_y[k] / (covariance.pos_y[k] + measurement_noise)

    # Update state estimates with observations
    position_error_x = observed_x[k] - state.x[k]
    position_error_y = observed_y[k] - state.y[k]

    state.x[k] = state.x[k] + gains.pos_x[k] * position_error_x
    state.y[k] = state.y[k] + gains.pos_y[k] * position_error_y

    # Update covariance matrices
    covariance.pos_x[k] = covariance.pos_x[k] - gains.pos_x[k] * covariance.pos_x[k]
    covariance.pos_y[k] = covariance.pos_y[k] - gains.pos_y[k] * covariance.pos_y[k]

    # Update velocity gains and covariance (simplified model)
    gains.vel_x[k] = covariance.vel_x[k] / (covariance.vel_x[k] + measurement_noise)
    gains.vel_y[k] = covariance.vel_y[k] / (covariance.vel_y[k] + measurement_noise)

    covariance.vel_x[k] = covariance.vel_x[k] - gains.vel_x[k] * covariance.vel_x[k]
    covariance.vel_y[k] = covariance.vel_y[k] - gains.vel_y[k] * covariance.vel_y[k]


def estimate_kalman_filter(
    history: NDArray[np.float32],
    prediction_horizon: int,
    process_noise: float = 0.00001,
    measurement_noise: float = 0.0001,
) -> NDArray[np.float32]:
    """Predict future position using a simplified Kalman filter for constant velocity motion.

    This implements a basic Kalman filter assuming constant velocity motion model. The state vector contains
        [position_x, position_y, velocity_x, velocity_y].

    Code adapted from: https://github.com/vita-epfl/UniTraj/blob/main/unitraj/datasets/common_utils.py#L258

    Args:
        history: Historical positions with shape [length_of_history, 2].
        prediction_horizon: Number of time steps into the future to predict.
        process_noise: Process noise covariance (Q) representing uncertainty in the motion model.
        measurement_noise: Measurement noise covariance (R) representing uncertainty in the observations.

    Returns:
        Predicted future position as [x, y] coordinates.
    """
    if history.shape[0] < MIN_VALID_POINTS:
        # Not enough history for prediction, return last known position
        return history[-1]

    # Extract position observations
    num_timesteps = history.shape[0]
    observed_x = history[:, 0]
    observed_y = history[:, 1]

    # Calculate average velocity from history
    avg_velocity_x = _compute_average_velocity(observed_x)
    avg_velocity_y = _compute_average_velocity(observed_y)

    # Initialize Kalman filter state and covariance matrices
    state = _initialize_kalman_state(num_timesteps, observed_x[0], observed_y[0])
    covariance = _initialize_kalman_covariance(num_timesteps)
    kalman_gains = _initialize_kalman_gains(num_timesteps)

    # Run Kalman filter through historical observations
    for timestep in range(num_timesteps - 1):
        _kalman_prediction_step(state, covariance, timestep, avg_velocity_x, avg_velocity_y, process_noise)
        _kalman_correction_step(state, covariance, kalman_gains, timestep, observed_x, observed_y, measurement_noise)

    # Make final prediction for future position
    final_timestep = num_timesteps - 1
    predicted_x = state.x[final_timestep] + avg_velocity_x * prediction_horizon
    predicted_y = state.y[final_timestep] + avg_velocity_y * prediction_horizon
    return np.array([predicted_x, predicted_y], dtype=np.float32)


def compute_kalman_difficulty(
    positions: NDArray[np.float32],
    mask: NDArray[np.bool_],
    last_observed_time_index: int,
    scale_factor: float = 100.0,
    ndim: int = 2,
) -> float:
    """Compute trajectory prediction difficulty using Kalman filter error.

    This function measures how difficult it is to predict an agent's future trajectory by comparing Kalman filter
    predictions against ground truth future positions. Higher scores indicate more unpredictable/difficult trajectories.

    Args:
        positions: Agent positions over time with shape [T, 3].
        mask: Boolean mask indicating valid timesteps with shape [T].
        last_observed_time_index: Index separating observed from future positions.
        scale_factor: Scaling factor to normalize difficulty scores. Defaults to 100.0.
        ndim: Number of spatial dimensions to consider (2 for x,y or 3 for x,y,z). Defaults to 2.

    Returns:
        Normalized difficulty score. Higher values indicate more unpredictable trajectories.
        Returns 0.0 if insufficient data for prediction.
    """
    # Split data into past (observed) and future (ground truth) segments
    past_mask = mask[:last_observed_time_index]
    future_mask = mask[last_observed_time_index:]

    # Extract valid positions from each segment
    past_positions = positions[:last_observed_time_index, :ndim][past_mask]
    # Check if we have sufficient data for prediction
    if future_mask.sum() == 0 or past_positions.shape[0] < MIN_VALID_POINTS:
        return 0.0

    # Get the prediction target (last valid future position)
    last_valid_future_index = np.where(future_mask)[0][-1]

    # Generate Kalman filter prediction
    predicted_position = estimate_kalman_filter(past_positions, last_valid_future_index + 1)

    # Compute prediction error
    prediction_target = positions[last_observed_time_index:, :ndim][future_mask][-1]
    prediction_error = np.linalg.norm(predicted_position - prediction_target).item()

    # Scale error by trajectory length to account for prediction difficulty
    trajectory_length_factor = last_valid_future_index + 1
    # Normalize by scale factor
    return (prediction_error * trajectory_length_factor) / scale_factor
