import numpy as np
from numpy.typing import NDArray


def compute_dists_to_conflict_points(
    conflict_points: NDArray[np.float32] | None, trajectories: NDArray[np.float32]
) -> NDArray[np.float32] | None:
    """Computes distances from agent trajectories to conflict points.

    Args:
        conflict_points (np.ndarray | None): Array of conflict points (shape: [num_conflict_points, 3]) or None.
        trajectories (np.ndarray): Array of agent trajectories (shape: [num_agents, num_time_steps, 3]).

    Returns:
        np.ndarray | None: Distances from each agent at each timestep to each conflict point
            (shape: [num_agents, num_time_steps, num_conflict_points]) or None if conflict_points is None.
    """
    if conflict_points is None:
        return None
    diff = conflict_points[None, None, :] - trajectories[:, :, None, :]
    return np.linalg.norm(diff, axis=-1)  # shape (num_agents, num_time_steps, num_conflict_points)


def compute_agent_to_agent_closest_dists(positions: NDArray[np.float32]) -> NDArray[np.float32]:
    """Computes the closest distance between each agent and any other agent over their trajectories.

    Args:
        positions (np.ndarray): Array of agent positions over time (shape: [num_agents, num_time_steps, 3]).
    """
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
    return np.nanmin(dists, axis=-1).astype(np.float32)
