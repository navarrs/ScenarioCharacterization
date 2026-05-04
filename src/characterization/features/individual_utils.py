"""Shared individual-agent feature utilities."""

import numpy as np
from numpy.typing import NDArray

from characterization.domains.ad.scenario_types import LaneMasker
from characterization.utils.common import ReturnCriterion, SpeedUnits, return_by_criterion
from characterization.utils.constants import MIN_VALID_POINTS, SPEED_TO_MS
from characterization.utils.geometric_utils import compute_moving_average
from characterization.utils.logging_utils import get_pylogger

logger = get_pylogger(__name__)

# ---------------------------------------------------------------------------
# Kalman filter
# Adapted from: https://github.com/vita-epfl/UniTraj/blob/main/unitraj/datasets/common_utils.py#L258
# ---------------------------------------------------------------------------


class KalmanState:
    """Position estimates for the Kalman filter."""

    def __init__(self, size: int) -> None:
        """Initialise position arrays of length ``size + 1``."""
        self.x = np.zeros(size + 1, dtype=np.float32)
        self.y = np.zeros(size + 1, dtype=np.float32)
        self.z = np.zeros(size + 1, dtype=np.float32)


class KalmanCovariance:
    """Covariance matrices for the Kalman filter."""

    def __init__(self, size: int) -> None:
        """Initialise covariance arrays of length ``size + 1``."""
        self.pos_x = np.zeros(size + 1, dtype=np.float32)
        self.pos_y = np.zeros(size + 1, dtype=np.float32)
        self.pos_z = np.zeros(size + 1, dtype=np.float32)
        self.vel_x = np.zeros(size + 1, dtype=np.float32)
        self.vel_y = np.zeros(size + 1, dtype=np.float32)
        self.vel_z = np.zeros(size + 1, dtype=np.float32)


class KalmanGains:
    """Kalman gain matrices."""

    def __init__(self, size: int) -> None:
        """Initialise gain arrays of length ``size + 1``."""
        self.pos_x = np.zeros(size + 1, dtype=np.float32)
        self.pos_y = np.zeros(size + 1, dtype=np.float32)
        self.pos_z = np.zeros(size + 1, dtype=np.float32)
        self.vel_x = np.zeros(size + 1, dtype=np.float32)
        self.vel_y = np.zeros(size + 1, dtype=np.float32)
        self.vel_z = np.zeros(size + 1, dtype=np.float32)


def estimate_kalman_filter(
    history: NDArray[np.float32],
    prediction_horizon: int,
    process_noise: float = 1e-5,
    measurement_noise: float = 1e-4,
    min_valid_points: int = MIN_VALID_POINTS,
    frame_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float32]:
    """Predict a future position using a constant-velocity Kalman filter.

    Args:
        history: Shape ``(T, 3)`` array of observed ``[x, y, z]`` positions.
        prediction_horizon: Number of time steps to extrapolate.
        process_noise: Process noise covariance Q.
        measurement_noise: Measurement noise covariance R.
        min_valid_points: Minimum valid history points required for prediction.
        frame_indices: Shape ``(T,)`` array of the original frame indices for each row in ``history``.
            When provided, velocity is estimated as position-difference divided by the actual timestep
            gap between valid frames, correcting for sparse/invalid frames removed before calling this
            function.  When ``None``, consecutive rows are assumed to be adjacent frames.

    Returns:
        Predicted ``[x, y, z]`` position as a float32 array.
    """
    if history.shape[0] < min_valid_points:
        return history[-1].astype(np.float32)

    n = history.shape[0]
    obs_x, obs_y, obs_z = history[:, 0], history[:, 1], history[:, 2]

    if frame_indices is not None and len(frame_indices) > 1:
        dt = np.diff(frame_indices).astype(np.float32)
        avg_vx = float(np.mean(np.diff(obs_x) / dt))
        avg_vy = float(np.mean(np.diff(obs_y) / dt))
        avg_vz = float(np.mean(np.diff(obs_z) / dt))
    else:
        avg_vx = float(np.mean(np.diff(obs_x)))
        avg_vy = float(np.mean(np.diff(obs_y)))
        avg_vz = float(np.mean(np.diff(obs_z)))

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


def _kalman_difficulty(
    past_positions: NDArray[np.float32],
    past_frame_indices: NDArray[np.intp] | None,
    future_positions: NDArray[np.float32],
    last_valid_future_step: int,
    downscale_factor: float,
    min_past_valid_points: int = MIN_VALID_POINTS,
    ndim: int = 3,
) -> float:
    """Compute Kalman-filter prediction difficulty from pre-extracted numpy arrays.

    Args:
        past_positions: Shape ``(T_past, 3)`` valid past positions.
        past_frame_indices: Original frame indices for each past position (for velocity estimation).
        future_positions: Shape ``(T_future, 3)`` valid future positions.
        last_valid_future_step: Number of steps ahead to predict (1-based index into future).
        downscale_factor: Normalisation divisor applied to the raw prediction error.
        min_past_valid_points: Minimum valid past frames required.
        ndim: Number of spatial dimensions to use when computing prediction error.

    Returns:
        Normalised difficulty score; higher values indicate harder-to-predict trajectories.
        Returns ``-1.0`` if there is insufficient data.
    """
    if past_positions.shape[0] < min_past_valid_points or future_positions.shape[0] == 0:
        return -1.0

    predicted = estimate_kalman_filter(
        past_positions,
        last_valid_future_step,
        min_valid_points=min_past_valid_points,
        frame_indices=past_frame_indices,
    )
    target = future_positions[-1]
    error = float(np.linalg.norm(predicted[:ndim] - target[:ndim]))
    return (error * last_valid_future_step) / downscale_factor


def compute_kalman_difficulty(
    positions: NDArray[np.float32],
    mask: NDArray[np.bool_],
    last_observed_time_index: int,
    scale_factor: float = 100.0,
    ndim: int = 2,
) -> float:
    """Compute trajectory prediction difficulty using Kalman filter error (AD interface).

    Splits the trajectory at ``last_observed_time_index``, fits a constant-velocity Kalman filter
    on the past, predicts to the last valid future frame, and returns the normalised prediction error.

    Args:
        positions: Agent positions over time with shape ``[T, 3]``.
        mask: Boolean mask indicating valid timesteps with shape ``[T]``.
        last_observed_time_index: Index separating observed from future positions.
        scale_factor: Divisor for normalising the difficulty score.
        ndim: Number of spatial dimensions to consider (2 for x,y or 3 for x,y,z).

    Returns:
        Normalised difficulty score; higher values indicate more unpredictable trajectories.
        Returns ``-1.0`` if insufficient data for prediction.
    """
    past_mask = mask[:last_observed_time_index]
    future_mask = mask[last_observed_time_index:]

    past_positions = positions[:last_observed_time_index, :3][past_mask].astype(np.float32)
    future_positions = positions[last_observed_time_index:, :3][future_mask].astype(np.float32)

    if future_positions.shape[0] == 0 or past_positions.shape[0] < MIN_VALID_POINTS:
        return -1.0

    last_valid_future_step = int(np.where(future_mask)[0][-1]) + 1
    past_frame_indices = np.where(past_mask)[0]

    return _kalman_difficulty(
        past_positions,
        past_frame_indices,
        future_positions,
        last_valid_future_step,
        scale_factor,
        ndim=ndim,
    )


def get_valid_timestamps(timestamps: NDArray[np.float32], valid_mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Return the timestamps that correspond to valid trajectory frames.

    Args:
        timestamps: Shape ``(T,)`` array of timestamps in seconds.
        valid_mask: Shape ``(T,)`` boolean array of valid frames.

    Returns:
        Shape ``(T_valid,)`` array of timestamps in seconds.
    """
    return timestamps[valid_mask].astype(np.float32)


def compute_speed_profile(
    speeds: NDArray[np.float32],
    criterion: ReturnCriterion,
) -> float | NDArray[np.float32]:
    """Aggregate the speed after applying a 5-point moving average.

    Args:
        speeds: Valid speed samples in m/s.
        criterion: How to reduce the per-timestep smoothed speeds to a scalar.
            ``CRITICAL`` returns the peak speed; ``AVERAGE`` returns the mean.

    Returns:
        Aggregated smoothed speed, or ``nan`` if the input is empty.
    """
    if len(speeds) == 0:
        return float("nan")
    smoothed = compute_moving_average(speeds, window_size=5)
    return return_by_criterion(smoothed, criterion)


def compute_acceleration_profile(
    speeds: NDArray[np.float32],
    timestamps: NDArray[np.float32],
    criterion: ReturnCriterion,
) -> tuple[float | NDArray[np.float32], float | NDArray[np.float32]]:
    """Compute the aggregated positive and negative acceleration.

    Args:
        speeds: Valid speed samples in m/s.
        timestamps: Corresponding timestamps in seconds (same length as speeds).
        criterion: How to reduce per-timestep acceleration values to a scalar.
            ``CRITICAL`` returns the peak; ``AVERAGE`` returns the mean.

    Returns:
        Tuple of (positive acceleration, deceleration magnitude) in m/s².
        Each element is ``0.0`` when no values of that sign exist, or ``nan`` when
        fewer than two speed samples are provided.
    """
    if len(speeds) < MIN_VALID_POINTS:
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

    Stationarity is defined as speed < ``max_stationary_speed``. Proximity is defined as distance
    to the nearest conflict point <= ``agent_to_conflict_point_max_distance``.

    Positions and conflict_points must have the same number of spatial dimensions.

    Args:
        positions: Shape ``(T_valid, D)`` positions in meters.
        timestamps: Shape ``(T_valid,)`` timestamps in seconds.
        speeds: Shape ``(T_valid,)`` speeds in m/s.
        conflict_points: Shape ``(K, D)`` conflict-point positions in meters.
        max_stationary_speed: Speed threshold in m/s below which the agent is considered stationary.
        agent_to_conflict_point_max_distance: Distance threshold in meters for proximity.
        return_criterion: How to reduce multiple waiting intervals to a single scalar.
            ``CRITICAL`` returns the longest; ``AVERAGE`` returns the mean.

    Returns:
        Total waiting time in seconds. Returns ``nan`` if no conflict points are available or fewer
        than 2 valid timesteps.
    """
    if len(conflict_points) == 0 or len(positions) < MIN_VALID_POINTS:
        return float("nan")

    # (T_valid, K) pairwise distances to conflict points
    diffs = positions[:, None, :] - conflict_points[None, :, :]
    min_dists = np.linalg.norm(diffs, axis=-1).min(axis=1)

    is_near_conflict = min_dists <= agent_to_conflict_point_max_distance
    is_stationary = speeds < max_stationary_speed
    waiting_mask = is_near_conflict & is_stationary
    if not np.any(waiting_mask):
        return 0.0

    dt = np.diff(timestamps)
    is_waiting = np.hstack([[False], waiting_mask, [False]])
    is_waiting = np.diff(is_waiting.astype(int))
    starts = np.where(is_waiting == 1)[0]
    ends = np.where(is_waiting == -1)[0]
    waiting_intervals = np.array([dt[start:end].sum() for start, end in zip(starts, ends, strict=False)])

    return return_by_criterion(np.asarray(waiting_intervals, dtype=np.float32), return_criterion)


def compute_speed_meta(
    velocities: NDArray[np.float32],
    closest_lanes: LaneMasker | None,
    lane_speed_limits: NDArray[np.float32] | None,
    *,
    apply_smoothing: bool = True,
) -> tuple[NDArray[np.float32] | None, NDArray[np.float32] | None]:
    """Compute the speed profile and speed-limit deviation from velocity vectors.

    Args:
        velocities: The velocity vectors of the agent over time (shape: ``[T, D]``).
        closest_lanes: Closest lanes information (shape: ``[T, K, 6]``) or ``None``.
        lane_speed_limits: Speed limits for each lane (shape: ``[K,]``) or ``None``.
        apply_smoothing: Whether to apply a 5-point moving average to the speed profile.

    Returns:
        Tuple of:
            speeds (NDArray or None): Speed time series (shape ``[T,]``), or ``None`` if NaN values present.
            speeds_limit_diff (NDArray or None): Mean absolute deviation from lane speed limits
                (shape ``[T,]``), or ``None`` if NaN values present.
    """
    speeds = np.linalg.norm(velocities, axis=-1)
    if apply_smoothing:
        speeds = compute_moving_average(speeds, window_size=5)

    if np.isnan(speeds).any():
        logger.warning("Nan value in agent speed: %s", speeds)
        return None, None

    speeds_limit_diff = np.zeros_like(speeds, dtype=np.float32)
    if closest_lanes is not None and lane_speed_limits is not None:
        k_closest_lane_idx = closest_lanes.lane_idx.squeeze(-1)
        k_speed_limits = (
            lane_speed_limits[k_closest_lane_idx] * SPEED_TO_MS[SpeedUnits.MPH]  # pyright: ignore[reportArgumentType]
        )
        speeds_limit_diff = np.abs(speeds[:, None] - k_speed_limits).mean(axis=-1)

    return speeds, speeds_limit_diff


def compute_jerk(speed: NDArray[np.float32], timestamps: NDArray[np.float32]) -> NDArray[np.float32] | None:
    """Compute the jerk (rate of change of acceleration) from speed and timestamps.

    Args:
        speed: Speed time series in m/s (shape: ``[T,]``).
        timestamps: Timestamps corresponding to each speed measurement (shape: ``[T,]``).

    Returns:
        Absolute jerk time series in m/s³ (shape: ``[T,]``), or ``None`` if NaN values are
        present or timestamps are not strictly increasing.

    Raises:
        ValueError: If speed and timestamps do not have the same shape.
    """
    if speed.shape != timestamps.shape:
        error_message = "Speed and timestamps must have the same shape."
        raise ValueError(error_message)

    if not np.all(np.diff(timestamps) > 0):
        logger.warning("Timestamps must be strictly increasing for jerk computation.")
        return None

    acceleration = np.gradient(speed, timestamps)
    if np.isnan(acceleration).any():
        logger.warning("Nan value in agent acceleration during jerk computation: %s", acceleration)
        return None

    jerk = np.gradient(acceleration, timestamps)
    if np.isnan(jerk).any():
        logger.warning("Nan value in agent jerk: %s", jerk)
        return None

    return np.abs(jerk)
