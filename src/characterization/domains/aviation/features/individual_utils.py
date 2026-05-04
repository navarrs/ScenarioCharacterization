"""Aviation-domain individual-agent utilities: trajectory type classification and Kalman difficulty."""

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from characterization.domains.aviation.scenario_types import AgentTrajectory
from characterization.domains.aviation.schemas.scenario_features import CharacterizationParameters
from characterization.features.individual_utils import estimate_kalman_filter
from characterization.utils.common import ValueClipper
from characterization.utils.geometric_utils import transform_to_reference_frame

_MIN_POSITION_SAMPLES = 2
_POSITION_N_DIMS = 3  # x, y, altitude
_MIN_PAST_VALID_POINTS = 2


class TrajectoryType(StrEnum):
    """Classification of an aviation agent's trajectory shape.

    Attributes:
        TYPE_STATIONARY: The agent did not move meaningfully.
        TYPE_TAKEOFF: The agent is taking off — either climbing or accelerating on the ground before liftoff.
        TYPE_LANDING: The agent lost significant altitude but had not reached the ground by the end of the trajectory.
        TYPE_LANDED: The agent lost significant altitude and reached the ground by the end of the trajectory.
        TYPE_STRAIGHT: The agent moved in a roughly straight line.
        TYPE_TURNING: The agent turned.
        TYPE_U_TURN: The agent reversed direction.
    """

    TYPE_STATIONARY = "TYPE_STATIONARY"
    TYPE_TAKEOFF = "TYPE_TAKEOFF"
    TYPE_LANDING = "TYPE_LANDING"
    TYPE_LANDED = "TYPE_LANDED"
    TYPE_STRAIGHT = "TYPE_STRAIGHT"
    TYPE_TURNING = "TYPE_TURNING"
    TYPE_U_TURN = "TYPE_U_TURN"


def classify_trajectory_type(  # noqa: PLR0911
    positions: NDArray[np.float32],
    speeds: NDArray[np.float32],
    *,
    config: CharacterizationParameters,
) -> TrajectoryType:
    """Classify the trajectory type.

    Types checked in order: TYPE_STATIONARY, TYPE_TAKEOFF, TYPE_LANDING, TYPE_LANDED, TYPE_U_TURN,
    TYPE_STRAIGHT, TYPE_TURNING. The first matching criterion wins.

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
        return TrajectoryType.TYPE_STATIONARY

    # 1. Stationary check
    total_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    if float(np.max(speeds)) < config.max_stationary_speed or total_displacement < config.max_stationary_displacement:
        return TrajectoryType.TYPE_STATIONARY

    # 2. Takeoff check: climbing, or accelerating on the ground before liftoff
    altitude_change = float(positions[-1, 2] - positions[0, 2])
    speed_gain = float(speeds[-1] - speeds[0])
    if altitude_change >= config.min_takeoff_altitude_gain or speed_gain >= config.min_takeoff_speed_gain:
        return TrajectoryType.TYPE_TAKEOFF

    # 3. Landing / landed check: significant altitude loss, split by whether the ground was reached
    if altitude_change <= -config.min_landing_altitude_loss:
        if float(positions[-1, 2]) <= config.max_landed_altitude:
            return TrajectoryType.TYPE_LANDED
        return TrajectoryType.TYPE_LANDING

    # Compute trajectory in a reference frame aligned with the initial heading
    ref_heading = np.arctan2(positions[1, 1] - positions[0, 1], positions[1, 0] - positions[0, 0]).astype(np.float32)
    ref_origin = positions[0].astype(np.float32)

    local_positions = transform_to_reference_frame(positions, ref_origin, np.asarray([ref_heading]))
    longitudinal_disp = float(local_positions[-1, 0])
    lateral_disp = float(np.max(np.abs(local_positions[:, 1])))

    heading_diff = np.arctan2(positions[-1, 1] - positions[-2, 1], positions[-1, 0] - positions[-2, 0]) - ref_heading
    heading_diff_deg = float(np.abs(np.degrees(np.arctan2(np.sin(heading_diff), np.cos(heading_diff)))))

    # 4. U-turn check
    if longitudinal_disp < config.min_uturn_longitudinal_displacement:
        return TrajectoryType.TYPE_U_TURN

    # 5. Straight check
    if (
        lateral_disp < config.max_straight_lateral_displacement
        and heading_diff_deg < config.max_straight_absolute_heading_diff
    ):
        return TrajectoryType.TYPE_STRAIGHT

    # 6. Otherwise, it's a turn
    return TrajectoryType.TYPE_TURNING


def compute_kalman_difficulty(
    trajectory: AgentTrajectory,
    current_time_index: int,
    *,
    scale_to_m: float,
    value_clipper: ValueClipper,
    min_past_valid_points: int = _MIN_PAST_VALID_POINTS,
    downscale_factor: float = 100.0,
) -> float:
    """Compute Kalman-filter prediction difficulty for an aviation agent.

    Splits the trajectory at ``current_time_index``, fits a constant-velocity Kalman filter on the
    past, predicts to the last valid future frame, and returns the normalised prediction error.

    Args:
        trajectory: Agent trajectory accessor.
        current_time_index: Frame index separating past and future.
        scale_to_m: Factor to convert stored position units to metres.
        value_clipper: Bounds for clipping the final difficulty score.
        min_past_valid_points: Minimum valid past frames required.
        downscale_factor: Factor by which to downscale the prediction error.

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

    xyz = trajectory.xyz_position * scale_to_m
    past_frame_indices: NDArray[np.intp] = np.where(past_mask)[0]
    past_positions = xyz[:current_time_index][past_mask].astype(np.float32)
    future_positions = xyz[current_time_index:][future_mask].astype(np.float32)

    last_valid_future_step = int(np.where(future_mask)[0][-1]) + 1
    predicted = estimate_kalman_filter(
        past_positions,
        last_valid_future_step,
        min_valid_points=min_past_valid_points,
        frame_indices=past_frame_indices,
    )
    error = float(np.linalg.norm(predicted - future_positions[-1]))
    kalman_difficulty = (error * last_valid_future_step) / downscale_factor
    return value_clipper.clip(kalman_difficulty)
