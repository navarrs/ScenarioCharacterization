"""Autonomous Driving (AD) agent type definitions for WOMD."""

from enum import Enum
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


class AgentType(Enum):
    """Agent Types for WOMD."""

    TYPE_UNSET = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_CYCLIST = 3
    TYPE_OTHER = 4
    TYPE_EGO_AGENT = 5
    TYPE_RELEVANT = 6


class AgentPairType(Enum):
    """Agent Pair Types for WOMD."""

    TYPE_UNSET = 0
    TYPE_VEHICLE_VEHICLE = 1
    TYPE_VEHICLE_PEDESTRIAN = 2
    TYPE_VEHICLE_CYCLIST = 3
    TYPE_PEDESTRIAN_PEDESTRIAN = 4
    TYPE_PEDESTRIAN_CYCLIST = 5
    TYPE_CYCLIST_CYCLIST = 6
    TYPE_OTHER = 10


def get_agent_pair_type(agent_type1: AgentType, agent_type2: AgentType) -> AgentPairType:
    """Determines the AgentPairType based on two AgentTypes.

    Args:
        agent_type1 (AgentType): The first agent type.
        agent_type2 (AgentType): The second agent type.

    Returns:
        AgentPairType: The determined agent pair type.
    """
    # Possible vehicle-to-vehicle combinations
    agent_pair_type = AgentPairType.TYPE_OTHER  # Default return value
    if (
        (agent_type1 == AgentType.TYPE_EGO_AGENT and agent_type2 == AgentType.TYPE_VEHICLE)
        or (agent_type1 == AgentType.TYPE_VEHICLE and agent_type2 == AgentType.TYPE_EGO_AGENT)
        or (agent_type1 == AgentType.TYPE_VEHICLE and agent_type2 == AgentType.TYPE_VEHICLE)
    ):
        agent_pair_type = AgentPairType.TYPE_VEHICLE_VEHICLE

    # Posible vehicle-to-pedestrian combinations
    if (
        (agent_type1 == AgentType.TYPE_EGO_AGENT and agent_type2 == AgentType.TYPE_PEDESTRIAN)
        or (agent_type1 == AgentType.TYPE_PEDESTRIAN and agent_type2 == AgentType.TYPE_EGO_AGENT)
        or (agent_type1 == AgentType.TYPE_VEHICLE and agent_type2 == AgentType.TYPE_PEDESTRIAN)
        or (agent_type1 == AgentType.TYPE_PEDESTRIAN and agent_type2 == AgentType.TYPE_VEHICLE)
    ):
        agent_pair_type = AgentPairType.TYPE_VEHICLE_PEDESTRIAN

    # Possible vehicle-to-cyclist combinations
    if (
        (agent_type1 == AgentType.TYPE_EGO_AGENT and agent_type2 == AgentType.TYPE_CYCLIST)
        or (agent_type1 == AgentType.TYPE_CYCLIST and agent_type2 == AgentType.TYPE_EGO_AGENT)
        or (agent_type1 == AgentType.TYPE_VEHICLE and agent_type2 == AgentType.TYPE_CYCLIST)
        or (agent_type1 == AgentType.TYPE_CYCLIST and agent_type2 == AgentType.TYPE_VEHICLE)
    ):
        agent_pair_type = AgentPairType.TYPE_VEHICLE_CYCLIST

    # Possible pedestrian-to-pedestrian combinations
    if agent_type1 == AgentType.TYPE_PEDESTRIAN and agent_type2 == AgentType.TYPE_PEDESTRIAN:
        agent_pair_type = AgentPairType.TYPE_PEDESTRIAN_PEDESTRIAN

    # Possible pedestrian-to-cyclist combinations
    if (agent_type1 == AgentType.TYPE_PEDESTRIAN and agent_type2 == AgentType.TYPE_CYCLIST) or (
        agent_type1 == AgentType.TYPE_CYCLIST and agent_type2 == AgentType.TYPE_PEDESTRIAN
    ):
        agent_pair_type = AgentPairType.TYPE_PEDESTRIAN_CYCLIST

    # Possible cyclist-to-cyclist combinations
    if agent_type1 == AgentType.TYPE_CYCLIST and agent_type2 == AgentType.TYPE_CYCLIST:
        agent_pair_type = AgentPairType.TYPE_CYCLIST_CYCLIST

    return agent_pair_type


class AgentTrajectoryMasker:
    """Masks for indexing trajectory data from the reformatted by the dataloader classes.

    The class expects an input of type (N, T, D=10) or (T, D=10) where N is the number of agents, T is the number of
    timesteps and D is the number of features per trajectory point, organized as follows:
        idx 0 to 2: the agent's (x, y, z) center coordinates.
        idx 3 to 5: the agent's length, width and height in meters.
        idx 6: the agent's angle (heading) of the forward direction in radians
        idx 7 to 8: the agent's (x, y) velocity in meters/second
        idx 9: a flag indicating if the information is valid
    """

    # Agent position masks
    _TRAJECTORY_XYZ_POS: ClassVar[list[bool]] = [True, True, True, False, False, False, False, False, False, False]
    _TRAJECTORY_XY_POS: ClassVar[list[bool]] = [True, True, False, False, False, False, False, False, False, False]

    # Agent dimensions masks
    _TRAJECTORY_DIMS: ClassVar[list[bool]] = [False, False, False, True, True, True, False, False, False, False]
    _TRAJECTORY_LENGTHS: ClassVar[list[bool]] = [False, False, False, True, False, False, False, False, False, False]
    _TRAJECTORY_WIDTHS: ClassVar[list[bool]] = [False, False, False, False, True, False, False, False, False, False]
    _TRAJECTORY_HEIGHTS: ClassVar[list[bool]] = [False, False, False, False, False, True, False, False, False, False]

    # Agent heading mask
    _TRAJECTORY_HEADING: ClassVar[list[bool]] = [False, False, False, False, False, False, True, False, False, False]

    # Agent velocity masks
    _TRAJECTORY_XY_VEL: ClassVar[list[bool]] = [False, False, False, False, False, False, False, True, True, False]
    _TRAJECTORY_X_VEL: ClassVar[list[bool]] = [False, False, False, False, False, False, False, True, False, False]
    _TRAJECTORY_Y_VEL: ClassVar[list[bool]] = [False, False, False, False, False, False, False, False, True, False]

    # Agent state, all features except valid mask
    _TRAJECTORY_STATE: ClassVar[list[bool]] = [True, True, True, True, True, True, True, True, True, False]

    # Agent valid mask
    _TRAJECTORY_VALID: ClassVar[list[bool]] = [False, False, False, False, False, False, False, False, False, True]

    _agent_trajectory: NDArray[np.float32]

    def __init__(self, trajectory: NDArray[np.float32]) -> None:
        """Initializes the AgentTrajectoryMasker with trajectory data.

        Args:
            trajectory (NDArray[np.float32]): The trajectory data of shape (N, T, D=10) or (T, D=10).
        """
        self._agent_trajectory = trajectory

    # Mask accessors
    @property
    def xyz_pos_mask(self) -> list[bool]:
        """Mask for the (x, y, z) position feature."""
        return self._TRAJECTORY_XYZ_POS

    @property
    def xy_pos_mask(self) -> list[bool]:
        """Mask for the (x, y) position feature."""
        return self._TRAJECTORY_XY_POS

    @property
    def xy_vel_mask(self) -> list[bool]:
        """Mask for the (x, y) velocity feature."""
        return self._TRAJECTORY_XY_VEL

    @property
    def heading_mask(self) -> list[bool]:
        """Mask for the heading feature."""
        return self._TRAJECTORY_HEADING

    # Trajectory accessors
    @property
    def agent_trajectories(self) -> NDArray[np.float32]:
        """Returns the full agent trajectory data."""
        return self._agent_trajectory

    @property
    def agent_dims(self) -> NDArray[np.float32]:
        """Returns the agents dimensions: length, width, height."""
        return self._agent_trajectory[..., self._TRAJECTORY_DIMS]

    @property
    def agent_lengths(self) -> NDArray[np.float32]:
        """Returns the length."""
        return self._agent_trajectory[..., self._TRAJECTORY_LENGTHS]

    @property
    def agent_widths(self) -> NDArray[np.float32]:
        """Returns the width."""
        return self._agent_trajectory[..., self._TRAJECTORY_WIDTHS]

    @property
    def agent_heights(self) -> NDArray[np.float32]:
        """Returns the height."""
        return self._agent_trajectory[..., self._TRAJECTORY_HEIGHTS]

    @property
    def agent_headings(self) -> NDArray[np.float32]:
        """Returns the heading."""
        return self._agent_trajectory[..., self._TRAJECTORY_HEADING]

    @property
    def agent_xyz_pos(self) -> NDArray[np.float32]:
        """Returns the (x, y, z) position."""
        return self._agent_trajectory[..., self._TRAJECTORY_XYZ_POS]

    @property
    def agent_xy_pos(self) -> NDArray[np.float32]:
        """Returns the (x, y) position."""
        return self._agent_trajectory[..., self._TRAJECTORY_XY_POS]

    @property
    def agent_xy_vel(self) -> NDArray[np.float32]:
        """Returns the (x, y) velocity."""
        return self._agent_trajectory[..., self._TRAJECTORY_XY_VEL]

    @property
    def agent_valid(self) -> NDArray[np.float32]:
        """Returns the valid mask."""
        valid = self._agent_trajectory[..., self._TRAJECTORY_VALID]
        return np.nan_to_num(valid, nan=0.0)

    @property
    def agent_state(self) -> NDArray[np.float32]:
        """Returns all features except the valid mask."""
        return self._agent_trajectory[..., self._TRAJECTORY_STATE]


class InteractionAgent:
    """Class representing an agent for interaction feature computation."""

    def __init__(self) -> None:
        """Initializes an InteractionAgent and resets all attributes."""
        self.reset()

    @property
    def position(self) -> NDArray[np.float32]:
        """NDArray[np.float32]: The positions of the agent over time (shape: [T, 2])."""
        return self._position

    @position.setter
    def position(self, value: NDArray[np.float32]) -> None:
        """Sets the positions of the agent.

        Args:
            value (NDArray[np.float32]): The positions of the agent over time (shape: [T, 2]).
        """
        self._position = np.asarray(value, dtype=np.float32)

    @property
    def speed(self) -> NDArray[np.float32]:
        """NDArray[np.float32]: The velocities of the agent over time (shape: [T,])."""
        return self._speed

    @speed.setter
    def speed(self, value: NDArray[np.float32]) -> None:
        """Sets the velocities of the agent.

        Args:
            value (NDArray[np.float32]): The velocities of the agent over time (shape: [T,]).
        """
        self._speed = np.asarray(value, dtype=np.float32)

    @property
    def heading(self) -> NDArray[np.float32]:
        """NDArray[np.float32]: The headings of the agent over time (shape: [T,])."""
        return self._heading

    @heading.setter
    def heading(self, value: NDArray[np.float32]) -> None:
        """Sets the headings of the agent.

        Args:
            value (NDArray[np.float32]): The headings of the agent over time (shape: [T,]).
        """
        self._heading = np.asarray(value, dtype=np.float32)

    @property
    def length(self) -> NDArray[np.float32]:
        """NDArray[np.float32]: The lengths of the agent over time (shape: [T,])."""
        return self._length

    @length.setter
    def length(self, value: NDArray[np.float32]) -> None:
        """Sets the lengths of the agent.

        Args:
            value (NDArray[np.float32]): The lengths of the agent over time (shape: [T,]).
        """
        self._length = np.asarray(value, dtype=np.float32)

    @property
    def width(self) -> NDArray[np.float32]:
        """NDArray[np.float32]: The widths of the agent over time (shape: [T,])."""
        return self._width

    @width.setter
    def width(self, value: NDArray[np.float32]) -> None:
        """Sets the widths of the agent.

        Args:
            value (NDArray[np.float32]): The widths of the agent over time (shape: [T,]).
        """
        self._width = np.asarray(value, dtype=np.float32)

    @property
    def height(self) -> NDArray[np.float32]:
        """NDArray[np.float32]: The heights of the agent over time (shape: [T,])."""
        return self._height

    @height.setter
    def height(self, value: NDArray[np.float32]) -> None:
        """Sets the heights of the agent.

        Args:
            value (NDArray[np.float32]): The heights of the agent over time (shape: [T,]).
        """
        self._height = np.asarray(value, dtype=np.float32)

    @property
    def agent_type(self) -> AgentType:
        """str: The type of the agent."""
        return self._agent_type

    @agent_type.setter
    def agent_type(self, value: AgentType) -> None:
        """Sets the type of the agent.

        Args:
            value (str): The type of the agent.
        """
        self._agent_type = value

    @property
    def is_stationary(self) -> bool:
        """Bool: Whether the agent is stationary (True/False)."""
        self._is_stationary = self.speed.mean() < self._stationary_speed
        return self._is_stationary

    @property
    def stationary_speed(self) -> float:
        """float: The speed threshold below which the agent is considered stationary."""
        return self._stationary_speed

    @stationary_speed.setter
    def stationary_speed(self, value: float) -> None:
        """Sets the stationary speed threshold.

        Args:
            value (float): The speed threshold below which the agent is considered stationary.
        """
        self._stationary_speed = value

    @property
    def in_conflict_point(self) -> bool:
        """bool: Whether the agent is in a conflict point."""
        self._in_conflict_point = np.any(
            self._dists_to_conflict <= self._agent_to_conflict_point_max_distance,
        ).__bool__()
        return self._in_conflict_point

    @property
    def agent_to_conflict_point_max_distance(self) -> float:
        """float: The maximum distance to a conflict point."""
        return self._agent_to_conflict_point_max_distance

    @agent_to_conflict_point_max_distance.setter
    def agent_to_conflict_point_max_distance(self, value: float) -> None:
        """Sets the maximum distance to a conflict point.

        Args:
            value (float): The maximum distance to a conflict point.
        """
        self._agent_to_conflict_point_max_distance = value

    @property
    def dists_to_conflict(self) -> NDArray[np.float32]:
        """NDArray[np.float32]: The distances to conflict points (shape: [T,])."""
        return self._dists_to_conflict

    @dists_to_conflict.setter
    def dists_to_conflict(self, value: NDArray[np.float32] | None) -> None:
        """Sets the distances to conflict points.

        Args:
            value (NDArray[np.float32] | None): The distances to conflict points (shape: [T,]).
        """
        self._dists_to_conflict = np.asarray(value, dtype=np.float32)

    @property
    def lane(self) -> NDArray[np.float32] | None:
        """NDArray[np.float32] or None: The lane of the agent, if available."""
        return self._lane

    @lane.setter
    def lane(self, value: NDArray[np.float32] | None) -> None:
        """Sets the lane of the agent.

        Args:
            value (NDArray[np.float32] | None): The lane of the agent, if available.
        """
        if value is not None:
            self._lane = np.asarray(value, dtype=np.float32)
        else:
            self._lane = None

    def reset(self) -> None:
        """Resets all agent attributes to their default values."""
        self._position = np.empty((0, 2), dtype=np.float32)
        self._speed = np.empty((0,), dtype=np.float32)
        self._heading = np.empty((0,), dtype=np.float32)
        self._dists_to_conflict = np.empty((0,), dtype=np.float32)
        self._stationary_speed = 0.1  # Default stationary speed threshold
        self._agent_to_conflict_point_max_distance = 0.5  # Default max distance to conflict point
        self._lane = np.empty((0,), dtype=np.float32)
        self._length = np.empty((0,), dtype=np.float32)
        self._width = np.empty((0,), dtype=np.float32)
        self._height = np.empty((0,), dtype=np.float32)
        self._agent_type = AgentType.TYPE_UNSET
