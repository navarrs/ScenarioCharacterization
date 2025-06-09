import numpy as np
from enum import Enum


from shapely import LineString

from utils.common import EPS, get_logger

logger = get_logger(__name__)


class InteractionAgent:
    def __init__(self):
        self.reset()

    @property
    def position(self) -> np.ndarray:
        """Returns the position of the agent."""
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        """Sets the position of the agent."""
        if value is not None:
            self._position = np.asarray(value, dtype=np.float32)
        else:
            self._position = None

    @property
    def velocity(self) -> np.ndarray:
        """Returns the velocity of the agent."""
        return self._velocity

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        """Sets the velocity of the agent."""
        if value is not None:
            self._velocity = np.asarray(value, dtype=np.float32)
        else:
            self._velocity = None

    @property
    def heading(self) -> np.ndarray:
        """Returns the heading of the agent."""
        return self._heading

    @heading.setter
    def heading(self, value: np.ndarray) -> None:
        """Sets the heading of the agent."""
        if value is not None:
            self._heading = np.asarray(value, dtype=np.float32)
        else:
            self._heading = None

    @property
    def agent_type(self) -> str:
        """Returns the type of the agent."""
        return self._agent_type

    @agent_type.setter
    def agent_type(self, value: str) -> None:
        """Sets the type of the agent."""
        if value is not None:
            self._agent_type = str(value)
        else:
            self._agent_type = None

    @property
    def is_stationary(self) -> bool:
        """Returns whether the agent is stationary."""
        return self._is_stationary

    def is_stationary(self, stationary_speed: float) -> None:
        """Sets whether the agent is stationary."""
        if self._velocity is None:
            self._is_stationary = False
        else:
            self._is_stationary = self.velocity.mean() < stationary_speed
        return self._is_stationary

    @property
    def in_conflict_point(self) -> bool:
        """Returns whether the agent is in a conflict point."""
        return self._in_conflict_point

    def in_conflict_point(self, agent_to_conflict_point_max_distance: bool) -> None:
        """Sets whether the agent is in a conflict point."""
        if self._dists_to_conflict is None:
            self._in_conflict_point = False
        else:
            self._in_conflict_point = np.any(self._dists_to_conflict <= agent_to_conflict_point_max_distance)
        return self._in_conflict_point

    @property
    def dists_to_conflict(self) -> np.ndarray:
        """Returns the distances to conflict points."""
        return self._dists_to_conflict

    @dists_to_conflict.setter
    def dists_to_conflict(self, value: np.ndarray) -> None:
        """Sets the distances to conflict points."""
        if value is not None:
            self._dists_to_conflict = np.asarray(value, dtype=np.float32)
        else:
            self._dists_to_conflict = None

    def reset(self) -> None:
        self._position = None
        self._velocity = None
        self._heading = None
        self._agent_type = None
        self._is_stationary = None
        self._in_conflict_point = None
        self._dists_to_conflict = None


def compute_separation(agent_i: InteractionAgent, agent_j: InteractionAgent) -> np.ndarray:
    """
    Computes the separation distance between two agents.

    Args:
        agent_i (InteractionAgent): The first agent.
        agent_j (InteractionAgent): The second agent.

    Returns:
        np.ndarray: A float array containing the separation distances between each segment of agent i
                    and the corresponding segment of agent j.
    """
    pos_i, pos_j = agent_i.position, agent_j.position
    return np.linalg.norm(pos_i - pos_j, axis=-1)


def compute_intersections(agent_i: InteractionAgent, agent_j: InteractionAgent) -> np.ndarray:
    """
    Computes whether two agents' trajectory segments are intersecting.

    Args:
        agent_i (InteractionAgent): The first agent.
        agent_j (InteractionAgent): The second agent.

    Returns:
        np.ndarray: A boolean array indicating whether each segment of agent i intersects with
                    the corresponding segment of agent j.
    """
    pos_i, pos_j = agent_i.position, agent_j.position
    if pos_i.shape[0] < 2 or pos_j.shape[0] < 2:
        return np.zeros((pos_i.shape[0],), dtype=np.bool)

    segments_i = np.stack([pos_i[:-1], pos_i[1:]], axis=1)
    segments_j = np.stack([pos_j[:-1], pos_j[1:]], axis=1)
    segments_i = [LineString(x) for x in segments_i]
    segments_j = [LineString(x) for x in segments_j]

    intersections = [x.intersects(y) for x, y in zip(segments_i, segments_j)]
    # Make it consistent with the number of timesteps
    return np.array([intersections[0]] + intersections, dtype=np.bool)


def compute_mttcp(
    agent_i: InteractionAgent,
    agent_j: InteractionAgent,
    agent_to_agent_max_distance: float = 0.5,
) -> tuple:
    """Computes the minimum time to conflict point (mTTCP):

                                   | 𝚫xi(t)     𝚫xj(t)  |
        𝚫TTCP  =       min         |------  ‒‒  ------  |
                  t in {0, tcp}    | 𝚫vi(t)     𝚫vj(t)  |

    between any two timesteps between two trajectories that are within a distance threshold from each
    other. Here t=0 is the time the two agents appear in the scene, and t=tcp is the first time one
    of the agents crosses the conflict point.
    Args:
        agent_i (InteractionAgent): The first agent.
        agent_j (InteractionAgent): The second agent.
        agent_to_agent_max_distance (float): The maximum distance between agents to consider for mTTCP.
    Returns:
        np.ndarray: An array of mTTCP values for each timestep.
    """
    pos_i, pos_j = agent_i.position, agent_j.position
    vel_i, vel_j = agent_i.velocity, agent_j.velocity

    # T, 2 -> T, T
    dists = np.linalg.norm(pos_i[:, None, :] - pos_j, axis=-1)
    i_idx, j_idx = np.where(dists <= agent_to_agent_max_distance)

    vals, i_unique = np.unique(i_idx, return_index=True)
    ti = i_idx[i_unique]
    if len(ti) == 0:
        mttcp = np.array([np.inf], dtype=np.float32)
        return mttcp

    conflict_points = pos_i[ti]
    mttcp = np.inf * np.ones(conflict_points.shape[0])

    cp_to_pos_i = np.linalg.norm(pos_i - conflict_points[:, None], axis=-1)
    cp_to_pos_j = np.linalg.norm(pos_j - conflict_points[:, None], axis=-1)
    tj = cp_to_pos_j.argmin(axis=-1)

    t_min = np.minimum(ti, tj) + 1
    for n, t in enumerate(t_min):
        # Compute the time to conflict point for each agent
        ttcp_i = cp_to_pos_i[n, :t] / vel_i[:t]  # Shape: (num. conflict points, 0 to t)
        ttcp_j = cp_to_pos_j[n, :t] / vel_j[:t]

        # Calculate the absolute difference in time to conflict point
        ttcp = np.abs(ttcp_i - ttcp_j)

        # Update the minimum mTTCP
        mttcp[n] = ttcp.min()

    return mttcp
