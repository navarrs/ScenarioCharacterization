"""AD-domain individual-agent utilities: trajectory type classification (WOMD-based)."""

import numpy as np
from numpy.typing import NDArray

from characterization.domains.ad.schemas.scenario import ScenarioMetadata
from characterization.utils.common import TrajectoryType


def _rotate_to_local_coordinates(
    displacement_vector: NDArray[np.float32],
    start_heading: np.float32,
) -> NDArray[np.float32]:
    """Rotate displacement vector to agent's local coordinate system."""
    rotation_matrix = np.array(
        [[np.cos(-start_heading), -np.sin(-start_heading)], [np.sin(-start_heading), np.cos(-start_heading)]],
    )
    return np.dot(rotation_matrix.squeeze(-1), displacement_vector)


def _is_stationary(
    max_speed: np.float32,
    displacement: np.float32,
    max_stationary_speed: float,
    max_stationary_displacement: float,
) -> bool:
    """Check if trajectory represents a stationary agent."""
    return bool(max_speed < max_stationary_speed and displacement < max_stationary_displacement)


def _is_straight_trajectory(heading_change: np.float32, max_straight_absolute_heading_diff: float) -> bool:
    """Check if trajectory is generally straight (small heading change)."""
    return bool(np.abs(heading_change) < max_straight_absolute_heading_diff)


def _classify_straight_trajectory(
    lateral_displacement: np.float32,
    max_straight_lateral_displacement: float,
) -> TrajectoryType:
    """Classify straight trajectory based on lateral displacement."""
    if np.abs(lateral_displacement) < max_straight_lateral_displacement:
        return TrajectoryType.TYPE_STRAIGHT
    return TrajectoryType.TYPE_STRAIGHT_RIGHT if lateral_displacement < 0 else TrajectoryType.TYPE_STRAIGHT_LEFT


def _is_right_turn(
    heading_change: np.float32,
    lateral_displacement: np.float32,
    max_straight_absolute_heading_diff: float,
) -> bool:
    """Check if trajectory represents a right turn."""
    return bool(heading_change < -max_straight_absolute_heading_diff and lateral_displacement < 0)


def _classify_right_trajectory(
    longitudinal_displacement: np.float32,
    min_uturn_longitudinal_displacement: float,
) -> TrajectoryType:
    """Classify right turn trajectory (turn vs U-turn)."""
    if longitudinal_displacement < min_uturn_longitudinal_displacement:
        return TrajectoryType.TYPE_RIGHT_U_TURN
    return TrajectoryType.TYPE_RIGHT_TURN


def _classify_left_trajectory(
    longitudinal_displacement: np.float32,
    min_uturn_longitudinal_displacement: float,
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
    """Classify trajectory type based on movement patterns and geometry (WOMD taxonomy).

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
    start_point, end_point = positions[0], positions[-1]
    displacement_vector = end_point[:2] - start_point[:2]
    final_displacement = np.linalg.norm(displacement_vector)

    start_heading, end_heading = headings[0], headings[-1]
    heading_change = end_heading - start_heading

    local_displacement = _rotate_to_local_coordinates(displacement_vector, np.deg2rad(start_heading))
    longitudinal_displacement, lateral_displacement = local_displacement

    start_speed, end_speed = speeds[0], speeds[-1]
    max_endpoint_speed = max(start_speed, end_speed)

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

    return _classify_left_trajectory(longitudinal_displacement, min_uturn_longitudinal_displacement)
