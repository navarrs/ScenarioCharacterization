import numpy as np
from numpy.typing import NDArray

XY_DIMENSIONS = 2
XYZ_DIMENSIONS = 3


def compute_moving_average(values: NDArray[np.float32], window_size: int = 5) -> NDArray[np.float32]:
    """Applies a simple moving average filter to smooth a time series.

    Args:
        values (NDArray[np.float32]): The raw time series (shape: [T,]).
        window_size (int, optional): The size of the moving average window. Defaults to 5.

    Returns:
        NDArray[np.float32]: The smoothed time series (shape: [T,]).
    """
    if window_size < 1:
        return values

    pad_size = window_size // 2
    padded_values = np.pad(values, (pad_size, pad_size), mode="edge")
    return np.convolve(padded_values, np.ones(window_size) / window_size, mode="valid")


def compute_median_filter(values: NDArray[np.float32], window_size: int = 5) -> NDArray[np.float32]:
    """Applies a median filter to smooth a time series.

    Args:
        values (NDArray[np.float32]): The raw time series (shape: [T,]).
        window_size (int, optional): The size of the median filter window. Defaults to 5.

    Returns:
        NDArray[np.float32]: The smoothed time series (shape: [T,]).
    """
    if window_size < 1:
        return values

    pad_size = window_size // 2
    padded_values = np.pad(values, (pad_size, pad_size), mode="edge")
    return np.array([np.median(padded_values[i : i + window_size]) for i in range(len(values))])


def compute_agent_to_agent_closest_dists(positions: NDArray[np.float32]) -> NDArray[np.float32]:
    """Computes the closest distance between each agent and any other agent over their trajectories.

    Args:
        positions: Array of agent positions over time with shape [num_agents, num_time_steps, D].

    Returns:
        Minimum distance from each agent to any other agent over time with shape [num_agents, num_time_steps]. NaN
            values are replaced with infinity.
    """
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
    return np.nan_to_num(np.nanmin(dists, axis=-1), nan=np.inf).astype(np.float32)


def transform_to_reference_frame(
    points: NDArray[np.float32],
    reference_origin: NDArray[np.float32],
    reference_heading: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Transforms points to a reference frame defined by reference_origin and reference_heading.

    Supports both 2D ``(N, 2)`` and 3D ``(N, 3)`` inputs. For 3D points the z coordinate is
    shifted but not rotated (rotation is around the Z-axis only).

    Args:
        points: Array of shape ``(N, 2)`` or ``(N, 3)`` representing points to be transformed.
        reference_origin: Array of shape ``(2,)`` or ``(3,)`` representing the origin of the new frame.
        reference_heading: Scalar representing the heading of the new frame in radians.

    Returns:
        Transformed points of the same shape as ``points``.
    """
    shifted = (points - reference_origin[None]).astype(np.float32)
    return rotate_points_along_z(shifted[None], -reference_heading.astype(np.float32)).squeeze(0)


def wrap_angle(angle: NDArray[np.float32]) -> NDArray[np.float32]:
    """Wrap angle to [-pi, pi].

    Args:
        angle: Angle in radians.

    Returns:
        Wrapped angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotate_points_along_z(points: NDArray[np.float32], angle: NDArray[np.float32]) -> NDArray[np.float32]:
    """Rotate points around the Z-axis using the given angle.

    Args:
        points: Array of shape (B, N, 3 + C) — B batches, N points, 3 coordinates (x, y, z) + C extra channels.
        angle: Array of shape (B,) — rotation angles per batch in radians.

    Returns:
        Rotated points of shape (B, N, 3 + C).
    """
    axis = 0 if len(angle) == 1 else 1

    is_2d = points.shape[-1] == XY_DIMENSIONS

    cosa = np.cos(angle)
    sina = np.sin(angle)

    if is_2d:
        rot_matrix = np.stack((cosa, sina, -sina, cosa), axis=1).reshape(-1, XY_DIMENSIONS, XY_DIMENSIONS)
        return np.matmul(points, rot_matrix)

    rot_matrix = np.stack(
        (
            cosa,
            sina,
            np.zeros_like(angle),
            -sina,
            cosa,
            np.zeros_like(angle),
            np.zeros_like(angle),
            np.zeros_like(angle),
            np.ones_like(angle),
        ),
        axis=axis,
    ).reshape(-1, XYZ_DIMENSIONS, XYZ_DIMENSIONS)

    points_rot = np.matmul(points[:, :, :XYZ_DIMENSIONS], rot_matrix)

    if points.shape[-1] > XYZ_DIMENSIONS:
        points_rot = np.concatenate((points_rot, points[:, :, XYZ_DIMENSIONS:]), axis=-1)

    return points_rot
