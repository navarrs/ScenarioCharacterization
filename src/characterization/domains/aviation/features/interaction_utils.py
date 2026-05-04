"""Aviation-specific interaction utility functions for pairwise agent interaction feature computation."""

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString

from characterization.features.interaction_utils import compute_separation
from characterization.utils.common import ReturnCriterion, return_by_criterion
from characterization.utils.constants import MIN_VALID_POINTS


def compute_intersection(
    pos_a: NDArray[np.float32],
    pos_b: NDArray[np.float32],
    joint_valid: NDArray[np.bool_],
    vertical_separation_threshold: float,
) -> float:
    """Check whether any temporally-aligned trajectory segment pair intersects in 2D with adequate vertical proximity.

    Splits each agent's joint-valid trajectory into consecutive segments and checks whether corresponding segments at
    the same timestep ``[t, t+1]`` intersect geometrically. Only corresponding segments are compared: segment ``t`` of
    agent A is never compared to segment ``t'≠t`` of agent B. This prevents spurious detections when agents pass
    through the same geographic point at different times.

    If a segment pair intersects in 2D, the vertical separation at timestep ``t`` is checked. The crossing is counted
    only if altitude difference is below ``vertical_separation_threshold``.

    Args:
        pos_a: Shape ``(T, 3)`` positions of agent A in meters.
        pos_b: Shape ``(T, 3)`` positions of agent B in meters.
        joint_valid: Shape ``(T,)`` joint-validity mask.
        vertical_separation_threshold: Altitude difference in meters below which a 2-D crossing counts as a real 3-D
            intersection.

    Returns:
        1.0 if any temporally-aligned segment pair intersects with adequate vertical proximity, 0.0 if no such crossing
            exists, or ``nan`` if fewer than 2 joint-valid positions are available to form segments.
    """
    pts_a = pos_a[joint_valid, :2]
    pts_b = pos_b[joint_valid, :2]
    if len(pts_a) < MIN_VALID_POINTS or len(pts_b) < MIN_VALID_POINTS:
        return float("nan")

    valid_pos_a = pos_a[joint_valid]
    valid_pos_b = pos_b[joint_valid]

    alt_diffs = np.abs(valid_pos_a[:-1, 2] - valid_pos_b[:-1, 2])
    vert_ok = alt_diffs < vertical_separation_threshold
    if not np.any(vert_ok):
        return 0.0

    segments_a = np.stack([pts_a[:-1], pts_a[1:]], axis=1)
    segments_b = np.stack([pts_b[:-1], pts_b[1:]], axis=1)

    for t in np.where(vert_ok)[0]:
        if LineString(segments_a[t]).intersects(LineString(segments_b[t])):
            return 1.0
    return 0.0


def compute_loss_of_separation(
    pos_a: NDArray[np.float32],
    pos_b: NDArray[np.float32],
    joint_valid: NDArray[np.bool_],
    lateral_separation_breach: float,
    vertical_separation_breach: float,
    criterion: ReturnCriterion,
) -> float | NDArray[np.float32]:
    """Compute a loss of separation (LoS) indicator between two agents.

    Returns ``1.0`` if the agents' 2-D trajectory paths intersect and the vertical separation at the crossing is below
    ``vertical_separation_breach``. Otherwise returns the fraction (or maximum, depending on ``criterion``) of
    joint-valid timesteps where both lateral and vertical separation are simultaneously below their respective
    thresholds.

    Args:
        pos_a: Shape ``(T, 3)`` positions of agent A in meters.
        pos_b: Shape ``(T, 3)`` positions of agent B in meters.
        joint_valid: Shape ``(T,)`` joint-validity mask.
        lateral_separation_breach: XY distance threshold in meters for per-timestep breach detection.
        vertical_separation_breach: Altitude difference threshold in meters, used for both the intersection vertical
            check and per-timestep breach detection.
        criterion: How to reduce per-timestep breach indicators when no path intersection is found. ``CRITICAL``
            returns the maximum (1.0 if any breach occurred); ``AVERAGE`` returns the fraction of breaching timesteps.

    Returns:
        LoS value in ``[0, 1]``, or ``nan`` if no joint-valid timesteps.
    """
    if not np.any(joint_valid):
        return float("nan")

    if compute_intersection(pos_a, pos_b, joint_valid, vertical_separation_breach) > 0:
        return 1.0

    lateral_dists, vertical_dists = compute_separation(pos_a, pos_b, joint_valid)
    breach = ((lateral_dists < lateral_separation_breach) & (vertical_dists < vertical_separation_breach)).astype(
        np.float32,
    )
    return return_by_criterion(breach, criterion)
