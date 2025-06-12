import numpy as np

from scenchar.utils.common import EPS, get_logger

logger = get_logger(__name__)


def compute_speed(velocities: np.ndarray) -> tuple[np.ndarray | None, ...]:
    """Computes the speed profile of an agent.

    Args:
        velocities (np.ndarray): The velocity vectors of the agent over time.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - speeds: The speed time series.
            - speeds_limit_diff: The difference between speed and speed limit (currently zeros).
    """
    speeds = np.linalg.norm(velocities, axis=-1)
    if np.isnan(speeds).any():
        logger.warning(f"Nan value in agent speed: {speeds}")
        return None, None

    # -----------------------------------------------------------------------------------------
    # TODO: Add speed limit difference feature. Depends on the context and lane information.
    speeds_limit_diff = np.zeros_like(speeds, dtype=np.float32)
    # speed_limits = np.zeros(velocities.shape[0])
    # in_lane = np.zeros(velocities.shape[0]).astype(bool)

    #     for i in range(lane_idx.shape[0]):
    #         speed_limits[i] = speed_limits[-1]
    #         if lane_idx[i][0] > 0 and lane_idx[i][0] < len(lane_info):
    #             speed_limits[i] = mph_to_ms(lane_info[int(lane_idx[i][0])]['speed_limit_mph'])
    #             in_lane[i] = True
    #         else:
    #             in_lane[i] = False
    # speed_limit_diff = speed - speed_limits
    # -----------------------------------------------------------------------------------------

    return speeds, speeds_limit_diff


def compute_acceleration_profile(velocity: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray | None, ...]:
    """Computes the acceleration profile from the speed (m/s) and time delta.

    Args:
        velocity (np.ndarray): The velocity (speed) time series (m/s).
        timestamps (np.ndarray): The timestamps corresponding to each velocity measurement.

    Returns:
        tuple[tuple, np.array]:
            - acceleration_raw: The raw acceleration time series.
            - acceleration: The sum of positive acceleration intervals.
            - deceleration: The sum of negative acceleration intervals (absolute value).
    """

    def get_acc_sums(acc: np.array, idx: np.array) -> tuple[np.array, np.array]:
        diff = idx[1:] - idx[:-1]
        diff = np.array([-1] + np.where(diff > 1)[0].tolist() + [diff.shape[0]])
        se_idxs = [(idx[s + 1], idx[e] + 1) for s, e in zip(diff[:-1], diff[1:])]
        sums = np.array([acc[s:e].sum() for s, e in se_idxs])
        return sums, se_idxs

    if not velocity.shape == timestamps.shape:
        raise ValueError("Velocity and timestamps must have the same shape.")

    acceleration_raw = np.gradient(velocity, timestamps)  # m/s^2

    if np.isnan(acceleration_raw).any():
        logger.warning(f"Nan value in agent acceleration: {acceleration_raw}")
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
        deceleration, idx_dec = get_acc_sums(acceleration_raw, dr_idx)

        ar_idx = np.where(acceleration_raw >= 0.0)[0]
        acceleration, idx_acc = get_acc_sums(acceleration_raw, ar_idx)

    return acceleration_raw, acceleration, np.abs(deceleration)


def compute_jerk(velocity: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Computes the jerk from the acceleration profile and time delta.

    Args:
        velocity (np.ndarray): The velocity (speed) time series (m/s).
        timestamps (np.ndarray): The timestamps corresponding to each velocity measurement.

    Returns:
        np.ndarray: The jerk time series (m/s^3), or None if NaN values are present.
    """
    if not velocity.shape == timestamps.shape:
        raise ValueError("Velocity and timestamps must have the same shape.")

    acceleration = np.gradient(velocity, timestamps)
    jerk = np.gradient(acceleration, timestamps)

    if np.isnan(jerk).any():
        logger.warning(f"Nan value in agent jerk: {jerk}")
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
        conflict_points (np.ndarray | None): The conflict points to check against (shape: [C, 2] or None).
        stationary_speed (float, optional): The speed threshold below which the agent is considered stationary.
            Defaults to 0.0.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - waiting_period: The waiting interval over the distance to the closest conflict point at that distance.
            - waiting_intervals: The duration of each waiting interval.
            - waiting_distances: The minimum distance to conflict points during each waiting interval.
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

        waiting_intervals = np.array([dt[start:end].sum() for start, end in zip(starts, ends)])
        # intervals = np.array([end - start for start, end in zip(starts, ends)])

        # For every timestep, get the minimum distance to the set of conflict points
        waiting_distances = np.linalg.norm(conflict_points[:, None] - position[starts], axis=-1).min(axis=0)

        # TODO:
        # # Get the index of the longest interval. Then, get the longest interval and the distance to
        # # the closest conflict point at that interval
        # idx = intervals.argmax()
        # # breakpoint()
        # waiting_period_interval_longest = intervals[idx]
        # waiting_period_distance_longest = dists_cps[idx] + EPS

        # # Get the index of the closest conflict point for each interval. Then get the interval for
        # # that index and the distance to that conflict point
        # idx = dists_cps.argmin()
        # waiting_period_interval_closest_conflict = intervals[idx]
        # waiting_period_distance_closest_conflict = dists_cps[idx] + EPS

    # waiting_intervals = np.asarray(
    #     [waiting_period_interval_longest, waiting_period_interval_closest_conflict])
    # waiting_distances_to_conflict = np.asarray(
    #     [waiting_period_distance_longest, waiting_period_distance_closest_conflict])

    waiting_period = waiting_intervals / (waiting_distances + EPS)
    return waiting_period, waiting_intervals, waiting_distances
