import numpy as np


def compute_dists_to_conflict_points(conflict_points: np.ndarray, trajectories: np.ndarray) -> np.ndarray:
    """Computes distances from agent trajectories to conflict points.

    Args:
        conflict_points (np.ndarray): Array of conflict points (shape: [num_conflict_points, 3]).
        trajectories (np.ndarray): Array of agent trajectories (shape: [num_agents, num_time_steps, 3]).

    Returns:
        np.ndarray: Distances from each agent at each timestep to each conflict point
            (shape: [num_agents, num_time_steps, num_conflict_points]).
    """
    diff = conflict_points[None, None, :] - trajectories[:, :, None, :]
    return np.linalg.norm(diff, axis=-1)  # shape (num_agents, num_time_steps, num_conflict_points)
