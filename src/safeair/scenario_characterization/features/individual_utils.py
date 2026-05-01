"""Utility functions and Kalman filter classes for individual-agent feature computation."""

import numpy as np
from numpy.typing import NDArray

from characterization.domains.aviation.scenario_types import AgentTrajectory
from characterization.utils.geometric_utils import transform_to_reference_frame
from safeair.scenario_characterization.common import (
    ReturnCriterion,
    TrajectoryType,
    ValueClipper,
    moving_average,
    return_by_criterion,
)
from safeair.schemas.scenario_features import CharacterizationParameters

# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity motion model)
# Adapted from: https://github.com/vita-epfl/UniTraj/blob/main/unitraj/datasets/common_utils.py#L258
# ---------------------------------------------------------------------------


_MIN_ACCEL_SAMPLES = 2
_MIN_POSITION_SAMPLES = 2
_POSITION_N_DIMS = 3  # x, y, altitude


class KalmanState:
    """Position estimates for the Kalman filter."""

    def __init__(self, size: int) -> None:
        """Intitialize position arrays of length ``size + 1``."""
        self.x = np.zeros(size + 1, dtype=np.float32)
        self.y = np.zeros(size + 1, dtype=np.float32)
        self.z = np.zeros(size + 1, dtype=np.float32)


class KalmanCovariance:
    """Covariance matrices for the Kalman filter."""

    def __init__(self, size: int) -> None:
        """Intitialize covariance arrays of length ``size + 1``."""
        self.pos_x = np.zeros(size + 1, dtype=np.float32)
        self.pos_y = np.zeros(size + 1, dtype=np.float32)
        self.pos_z = np.zeros(size + 1, dtype=np.float32)
        self.vel_x = np.zeros(size + 1, dtype=np.float32)
        self.vel_y = np.zeros(size + 1, dtype=np.float32)
        self.vel_z = np.zeros(size + 1, dtype=np.float32)


class KalmanGains:
    """Kalman gain matrices."""

    def __init__(self, size: int) -> None:
        """Intitialize gain arrays of length ``size + 1``."""
        self.pos_x = np.zeros(size + 1, dtype=np.float32)
        self.pos_y = np.zeros(size + 1, dtype=np.float32)
        self.pos_z = np.zeros(size + 1, dtype=np.float32)
        self.vel_x = np.zeros(size + 1, dtype=np.float32)
        self.vel_y = np.zeros(size + 1, dtype=np.float32)
        self.vel_z = np.zeros(size + 1, dtype=np.float32)


def _avg_velocity(positions: NDArray[np.float32]) -> float:
    return float(np.mean(np.diff(positions)))


def estimate_kalman_filter(
    history: NDArray[np.float32],
    prediction_horizon: int,
    process_noise: float = 1e-5,
    measurement_noise: float = 1e-4,
    min_valid_points: int = 2,
    frame_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float32]:
    """Predict a future position using a constant-velocity Kalman filter.

    Args:
        history: Shape ``(T, 3)`` array of observed [x, y, z] positions in meters.
        prediction_horizon: Number of time steps to extrapolate.
        process_noise: Process noise covariance Q.
        measurement_noise: Measurement noise covariance R.
        min_valid_points: Minimum valid history points required for prediction.
        frame_indices: Shape ``(T,)`` array of the original frame indices corresponding to each row in ``history``.
            When provided, velocity is estimated as position-difference divided by the actual timestep gap between
            valid frames, correcting for sparse/invalid frames that were removed before calling this function.
            When ``None``, consecutive rows are assumed to be adjacent frames.

    Returns:
        Predicted ``[x, y, z]`` position as a float32 array.
    """
    if history.shape[0] < min_valid_points:
        return history[-1].astype(np.float32)

    n = history.shape[0]
    obs_x, obs_y, obs_z = history[:, 0], history[:, 1], history[:, 2]

    if frame_indices is not None and len(frame_indices) > 1:
        dt = np.diff(frame_indices).astype(np.float32)  # actual timestep gaps between valid frames
        avg_vx = float(np.mean(np.diff(obs_x) / dt))
        avg_vy = float(np.mean(np.diff(obs_y) / dt))
        avg_vz = float(np.mean(np.diff(obs_z) / dt))
    else:
        avg_vx = _avg_velocity(obs_x)
        avg_vy = _avg_velocity(obs_y)
        avg_vz = _avg_velocity(obs_z)

    state = KalmanState(n)
    cov = KalmanCovariance(n)
    gains = KalmanGains(n)

    state.x[0], state.y[0], state.z[0] = obs_x[0], obs_y[0], obs_z[0]
    cov.pos_x[0] = cov.pos_y[0] = cov.pos_z[0] = 1.0
    cov.vel_x[0] = cov.vel_y[0] = cov.vel_z[0] = 1.0

    for k in range(n - 1):
        # Predict
        state.x[k + 1] = state.x[k] + avg_vx
        state.y[k + 1] = state.y[k] + avg_vy
        state.z[k + 1] = state.z[k] + avg_vz
        cov.pos_x[k + 1] = cov.pos_x[k] + cov.vel_x[k] + process_noise
        cov.pos_y[k + 1] = cov.pos_y[k] + cov.vel_y[k] + process_noise
        cov.pos_z[k + 1] = cov.pos_z[k] + cov.vel_z[k] + process_noise
        cov.vel_x[k + 1] = cov.vel_x[k] + process_noise
        cov.vel_y[k + 1] = cov.vel_y[k] + process_noise
        cov.vel_z[k + 1] = cov.vel_z[k] + process_noise

        # Correct
        kp = k + 1
        gains.pos_x[kp] = cov.pos_x[kp] / (cov.pos_x[kp] + measurement_noise)
        gains.pos_y[kp] = cov.pos_y[kp] / (cov.pos_y[kp] + measurement_noise)
        gains.pos_z[kp] = cov.pos_z[kp] / (cov.pos_z[kp] + measurement_noise)
        gains.vel_x[kp] = cov.vel_x[kp] / (cov.vel_x[kp] + measurement_noise)
        gains.vel_y[kp] = cov.vel_y[kp] / (cov.vel_y[kp] + measurement_noise)
        gains.vel_z[kp] = cov.vel_z[kp] / (cov.vel_z[kp] + measurement_noise)

        state.x[kp] += gains.pos_x[kp] * (obs_x[kp] - state.x[kp])
        state.y[kp] += gains.pos_y[kp] * (obs_y[kp] - state.y[kp])
        state.z[kp] += gains.pos_z[kp] * (obs_z[kp] - state.z[kp])

        cov.pos_x[kp] = (1 - gains.pos_x[kp]) * cov.pos_x[kp]
        cov.pos_y[kp] = (1 - gains.pos_y[kp]) * cov.pos_y[kp]
        cov.pos_z[kp] = (1 - gains.pos_z[kp]) * cov.pos_z[kp]
        cov.vel_x[kp] = (1 - gains.vel_x[kp]) * cov.vel_x[kp]
        cov.vel_y[kp] = (1 - gains.vel_y[kp]) * cov.vel_y[kp]
        cov.vel_z[kp] = (1 - gains.vel_z[kp]) * cov.vel_z[kp]

    ft = n - 1
    return np.array(
        [
            state.x[ft] + avg_vx * prediction_horizon,
            state.y[ft] + avg_vy * prediction_horizon,
            state.z[ft] + avg_vz * prediction_horizon,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Trajectory extraction helpers
# ---------------------------------------------------------------------------


def get_valid_timestamps(timestamps: NDArray[np.float32], valid_mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Return the timestamps that correspond to valid trajectory frames.

    Args:
        timestamps: Shape ``(T,)`` array of timestamps in seconds.
        valid_mask: Shape ``(T,)`` boolean array produced by ``get_valid_mask()``.

    Returns:
        Shape ``(T_valid,)`` array of timestamps in seconds.
    """
    return timestamps[valid_mask].astype(np.float32)


# ---------------------------------------------------------------------------
# Individual kinematic feature computations
# ---------------------------------------------------------------------------


def compute_speed_profile(speeds: NDArray[np.float32], criterion: ReturnCriterion) -> float | NDArray[np.float32]:
    """Compute the aggregated speed after applying a 5-point moving average.

    Args:
        speeds: Valid speed samples in m/s.
        criterion: How to reduce the per-timestep smoothed speeds to a scalar.
            ``CRITICAL`` returns the peak speed; ``AVERAGE`` returns the mean.

    Returns:
        Aggregated smoothed speed, or ``nan`` if the input is empty.
    """
    if len(speeds) == 0:
        return float("nan")
    smoothed = moving_average(speeds, window=5)
    return return_by_criterion(smoothed, criterion)


def compute_acceleration_profile(
    speeds: NDArray[np.float32], timestamps: NDArray[np.float32], criterion: ReturnCriterion
) -> tuple[float | NDArray[np.float32], float | NDArray[np.float32]]:
    """Compute the aggregated positive and negative acceleration.

    Args:
        speeds: Valid speed samples in m/s.
        timestamps: Corresponding timestamps in seconds (same length as speeds).
        criterion: How to reduce the per-timestep positive acceleration values to a scalar.
            ``CRITICAL`` returns the peak acceleration; ``AVERAGE`` returns the mean.

    Returns:
        Aggregated positive and negative acceleration in m/s², or ``0.0`` if no positive/negative values exist.
    """
    if len(speeds) < _MIN_ACCEL_SAMPLES:
        return float("nan"), float("nan")

    accel = np.gradient(speeds, timestamps)
    positive_accel = accel[accel > 0]
    negative_accel = accel[accel < 0]

    return (
        return_by_criterion(positive_accel.astype(np.float32), criterion) if len(positive_accel) > 0 else 0.0,
        return_by_criterion(np.abs(negative_accel).astype(np.float32), criterion) if len(negative_accel) > 0 else 0.0,
    )


def compute_waiting_period(
    positions: NDArray[np.float32],
    timestamps: NDArray[np.float32],
    speeds: NDArray[np.float32],
    conflict_points: NDArray[np.float32],
    max_stationary_speed: float,
    agent_to_conflict_point_max_distance: float,
    return_criterion: ReturnCriterion,
) -> float | NDArray[np.float32]:
    """Compute total time the agent was stationary near a conflict point.

    Stationarity is defined as speed < ``max_stationary_speed``. Proximity is defined as distance to the nearest
    conflict point <= ``agent_to_conflict_point_max_distance``.

    Args:
        positions: Shape ``(T_valid, 3)`` positions in meters.
        timestamps: Shape ``(T_valid,)`` timestamps in seconds.
        speeds: Shape ``(T_valid,)`` speeds in m/s.
        conflict_points: Shape ``(K, 3)`` conflict-point positions in meters (hold-short lines).
        max_stationary_speed: Speed threshold in m/s below which the agent is considered stationary.
        agent_to_conflict_point_max_distance: Distance threshold in meters for proximity.
        return_criterion: How to reduce multiple waiting intervals to a single scalar. ``CRITICAL`` returns the longest
            waiting period; ``AVERAGE`` returns the mean.

    Returns:
        Total waiting time in seconds. Returns ``nan`` if no conflict points are available or fewer than 2 valid
        timesteps.
    """
    if len(conflict_points) == 0 or len(positions) < _MIN_POSITION_SAMPLES:
        return float("nan")

    # (T_valid, K) pairwise distances to conflict points
    diffs = positions[:, None, :] - conflict_points[None, :, :]  # (T, K, 3)
    min_dists = np.linalg.norm(diffs, axis=-1).min(axis=1)  # (T,)

    # Identify timesteps where the agent is near a conflict point and stationary
    is_near_conflict = min_dists <= agent_to_conflict_point_max_distance
    is_stationary = speeds < max_stationary_speed
    waiting_mask = is_near_conflict & is_stationary
    if not np.any(waiting_mask):
        return 0.0

    dt = np.diff(timestamps)

    # Compute contiguous waiting intervals (in seconds), then reduce by criterion
    is_waiting = np.hstack([[False], waiting_mask, [False]])
    is_waiting = np.diff(is_waiting.astype(int))
    starts = np.where(is_waiting == 1)[0]
    ends = np.where(is_waiting == -1)[0]
    waiting_intervals = np.array([dt[start:end].sum() for start, end in zip(starts, ends, strict=False)])

    return return_by_criterion(np.asarray(waiting_intervals, dtype=np.float32), return_criterion)


def classify_trajectory_type(  # noqa: PLR0911
    positions: NDArray[np.float32],
    speeds: NDArray[np.float32],
    *,
    config: CharacterizationParameters,
) -> TrajectoryType:
    """Classify the trajectory as STATIONARY, TAKEOFF, LANDING, LANDED, U_TURN, STRAIGHT, or TURNING.

    Classification is applied in that order; the first matching criterion wins.

    Args:
        positions: Shape ``(T_valid, 3)`` positions ``[x, y, altitude]`` in meters.
        speeds: Shape ``(T_valid,)`` speeds in m/s.
        config: Thresholds controlling each classification boundary.

    Returns:
        The trajectory type.
    """
    if positions.ndim != 2 or positions.shape[1] != _POSITION_N_DIMS:  # noqa: PLR2004
        msg = f"positions must have shape (T, 3), got {positions.shape}"
        raise ValueError(msg)

    if len(positions) < _MIN_POSITION_SAMPLES:
        return TrajectoryType.STATIONARY

    # 1. Stationary check
    total_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    if float(np.max(speeds)) < config.max_stationary_speed or total_displacement < config.max_stationary_displacement:
        return TrajectoryType.STATIONARY

    # 2. Takeoff check: climbing, or accelerating on the ground before liftoff
    altitude_change = float(positions[-1, 2] - positions[0, 2])
    speed_gain = float(speeds[-1] - speeds[0])
    if altitude_change >= config.min_takeoff_altitude_gain or speed_gain >= config.min_takeoff_speed_gain:
        return TrajectoryType.TAKEOFF

    # 3. Landing / landed check: significant altitude loss, split by whether the ground was reached
    if altitude_change <= -config.min_landing_altitude_loss:
        if float(positions[-1, 2]) <= config.max_landed_altitude:
            return TrajectoryType.LANDED
        return TrajectoryType.LANDING

    # Compute trajectory in a reference frame aligned with the initial heading
    ref_heading = np.arctan2(positions[1, 1] - positions[0, 1], positions[1, 0] - positions[0, 0]).astype(np.float32)
    ref_origin = positions[0].astype(np.float32)

    local_positions = transform_to_reference_frame(positions, ref_origin, np.asarray([ref_heading]))
    longitudinal_disp = float(local_positions[-1, 0])  # forward/backward
    lateral_disp = float(np.max(np.abs(local_positions[:, 1])))  # max lateral deviation

    # Heading change over the full trajectory
    heading_diff = np.arctan2(positions[-1, 1] - positions[-2, 1], positions[-1, 0] - positions[-2, 0]) - ref_heading
    heading_diff_deg = float(np.abs(np.degrees(np.arctan2(np.sin(heading_diff), np.cos(heading_diff)))))

    # 4. U-turn check
    if longitudinal_disp < config.min_uturn_longitudinal_displacement:
        return TrajectoryType.U_TURN

    # 5. Straight check
    if (
        lateral_disp < config.max_straight_lateral_displacement
        and heading_diff_deg < config.max_straight_absolute_heading_diff
    ):
        return TrajectoryType.STRAIGHT

    # 6. Otherwise, it's a turn
    return TrajectoryType.TURNING


def compute_kalman_difficulty(
    trajectory: AgentTrajectory,
    current_time_index: int,
    *,
    scale_to_m: float,
    value_clipper: ValueClipper,
    min_past_valid_points: int = 2,
    downscale_factor: float = 100.0,
) -> float:
    """Compute Kalman-filter prediction difficulty for an agent.

    Splits the trajectory at ``current_time_index``, fits a constant-velocity Kalman filter on the past, predicts to the
    last valid future frame, and returns the normalised prediction error.

    Args:
        trajectory: Agent trajectory accessor.
        current_time_index: Frame index separating past and future.
        scale_to_m: Factor to convert stored position units to meters.
        min_past_valid_points: Minimum valid past frames required.
        downscale_factor: Factor by which to downscale the prediction error.
        value_clipper: Bounds for clipping the final difficulty score.

    Returns:
        Normalised difficulty score (higher = harder to predict). Returns ``nan`` if insufficient data.
    """
    if downscale_factor <= 0:
        error_message = f"downscale_factor must be positive, got {downscale_factor}"
        raise ValueError(error_message)

    valid_mask = trajectory.get_valid_mask()
    past_mask = valid_mask[:current_time_index]
    future_mask = valid_mask[current_time_index:]
    if past_mask.sum() < min_past_valid_points or future_mask.sum() == 0:
        return float("nan")

    # Extract xyz in meters
    xyz = trajectory.xyz_position * scale_to_m  # (T, 3)
    past_indices = np.where(past_mask)[0]  # original frame indices of valid past positions
    past_positions = xyz[:current_time_index][past_mask].astype(np.float32)

    last_valid_future_idx = int(np.where(future_mask)[0][-1]) + 1
    total_future_steps = int(future_mask.shape[0])
    predicted = estimate_kalman_filter(
        past_positions, last_valid_future_idx, min_valid_points=min_past_valid_points, frame_indices=past_indices
    )

    target = xyz[current_time_index:][future_mask][-1].astype(np.float32)
    error = float(np.linalg.norm(predicted - target))
    kalman_difficulty = (error * last_valid_future_idx / total_future_steps) / downscale_factor
    return value_clipper.clip(kalman_difficulty)
