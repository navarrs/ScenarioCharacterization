"""AD-specific interaction utility functions for pairwise agent interaction feature computation."""

import numpy as np
from numpy.typing import NDArray
from shapely import LineString

from characterization.utils.constants import MIN_VALID_POINTS


def compute_intersections_per_timestep(
    pos_a: NDArray[np.float32],
    pos_b: NDArray[np.float32],
    joint_valid: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Return a shape-(T,) boolean array indicating whether trajectory segments intersect in 2D.

    Only joint-valid consecutive position pairs form segments. A segment at index k covers
    ``[joint_valid_pos[k], joint_valid_pos[k+1]]``. The result entry for that segment-start timestep is True if
    the corresponding segment of agent A intersects that of agent B in the XY plane. Timesteps that are not
    joint-valid are always False.

    Args:
        pos_a: Shape ``(T, 3)`` positions of agent A. Only xy (first two coords) are used.
        pos_b: Shape ``(T, 3)`` positions of agent B. Only xy (first two coords) are used.
        joint_valid: Shape ``(T,)`` joint-validity mask.

    Returns:
        Shape ``(T,)`` boolean array.
    """
    result = np.zeros(len(joint_valid), dtype=bool)
    pts_a = pos_a[joint_valid, :2]
    pts_b = pos_b[joint_valid, :2]
    if len(pts_a) < MIN_VALID_POINTS:
        return result

    segments_a = np.stack([pts_a[:-1], pts_a[1:]], axis=1)
    segments_b = np.stack([pts_b[:-1], pts_b[1:]], axis=1)
    valid_indices = np.where(joint_valid)[0]
    for k, (sa, sb) in enumerate(zip(segments_a, segments_b, strict=False)):
        if LineString(sa).intersects(LineString(sb)):
            result[valid_indices[k]] = True
    return result
