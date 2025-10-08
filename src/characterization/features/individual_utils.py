import numpy as np
from numpy.typing import NDArray

from characterization.schemas import ScenarioMetadata
from characterization.utils.common import SMALL_EPS, LaneMasker, TrajectoryType, mph_to_ms
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
