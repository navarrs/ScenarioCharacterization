from collections.abc import Callable
from enum import Enum
from typing import Annotated, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BeforeValidator

from characterization.utils.ad_types import AgentType

EPS = 1e-6
SUPPORTED_SCENARIO_TYPES = ["gt", "ho"]


# Validator factory
def validate_array(
    expected_dtype: Any,
    expected_ndim: int,
) -> Callable[[Any], NDArray]:  # pyright: ignore[reportMissingTypeArgument]
    def _validator(v: Any) -> NDArray:  # pyright: ignore[reportMissingTypeArgument]
        if not isinstance(v, np.ndarray):
            raise TypeError("Expected a numpy.ndarray")
        if v.dtype != expected_dtype:
            raise TypeError(f"Expected dtype {expected_dtype}, got {v.dtype}")
        if v.ndim != expected_ndim:
            raise ValueError(f"Expected {expected_ndim}D array, got {v.ndim}D")
        return v

    return _validator


# Reusable types
BooleanNDArray2D = Annotated[NDArray[np.bool_], BeforeValidator(validate_array(np.bool_, 2))]
BooleanNDArray3D = Annotated[NDArray[np.bool_], BeforeValidator(validate_array(np.bool_, 3))]
Float64NDArray3D = Annotated[NDArray[np.float64], BeforeValidator(validate_array(np.float64, 3))]
Float32NDArray3D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 3))]
Float32NDArray2D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 2))]
Float32NDArray1D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 1))]
Int32NDArray1D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 1))]
Int32NDArray2D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 2))]
Int64NDArray2D = Annotated[NDArray[np.int64], BeforeValidator(validate_array(np.int64, 2))]


class InteractionStatus(Enum):
    UNKNOWN = -1
    COMPUTED_OK = 0
    PARTIAL_INVALID_HEADING = 1
    MASK_NOT_VALID = 2
    DISTANCE_TOO_FAR = 3
    STATIONARY = 4


class ReturnCriterion(Enum):
    CRITICAL = 0
    AVERAGE = 1
    UNSET = -1


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

    _agent_trajectory: np.ndarray

    def __init__(self, trajectory: np.ndarray) -> None:
        """Initializes the AgentTrajectoryMasker with trajectory data.

        Args:
            trajectory (np.ndarray): The trajectory data of shape (N, T, D=10) or (T, D=10).
        """
        self._agent_trajectory = trajectory

    # Mask accessors
    @property
    def xyz_pos_mask(self) -> list[bool]:
        return self._TRAJECTORY_XYZ_POS

    @property
    def xy_pos_mask(self) -> list[bool]:
        return self._TRAJECTORY_XY_POS

    @property
    def xy_vel_mask(self) -> list[bool]:
        return self._TRAJECTORY_XY_VEL

    @property
    def heading_mask(self) -> list[bool]:
        return self._TRAJECTORY_HEADING

    # Trajectory accessors
    @property
    def agent_trajectories(self) -> np.ndarray:
        return self._agent_trajectory

    @property
    def agent_dims(self) -> np.ndarray:
        return self._agent_trajectory[..., self._TRAJECTORY_DIMS]

    @property
    def agent_lengths(self) -> np.ndarray:
        return self._agent_trajectory[..., self._TRAJECTORY_LENGTHS]

    @property
    def agent_widths(self) -> np.ndarray:
        return self._agent_trajectory[..., self._TRAJECTORY_WIDTHS]

    @property
    def agent_heights(self) -> np.ndarray:
        return self._agent_trajectory[..., self._TRAJECTORY_HEIGHTS]

    @property
    def agent_headings(self) -> np.ndarray:
        return self._agent_trajectory[..., self._TRAJECTORY_HEADING]

    @property
    def agent_xyz_pos(self) -> np.ndarray:
        return self._agent_trajectory[..., self._TRAJECTORY_XYZ_POS]

    @property
    def agent_xy_pos(self) -> np.ndarray:
        return self._agent_trajectory[..., self._TRAJECTORY_XY_POS]

    @property
    def agent_xy_vel(self) -> np.ndarray:
        return self._agent_trajectory[..., self._TRAJECTORY_XY_VEL]

    @property
    def agent_valid(self) -> np.ndarray:
        return self._agent_trajectory[..., self._TRAJECTORY_VALID]

    @property
    def agent_state(self) -> np.ndarray:
        return self._agent_trajectory[..., self._TRAJECTORY_STATE]


class InteractionAgent:
    """Class representing an agent for interaction feature computation."""

    def __init__(self):
        """Initializes an InteractionAgent and resets all attributes."""
        self.reset()

    @property
    def position(self) -> np.ndarray | None:
        """np.ndarray: The positions of the agent over time (shape: [T, 2])."""
        return self._position

    @position.setter
    def position(self, value: np.ndarray | None) -> None:
        """Sets the positions of the agent.

        Args:
            value (np.ndarray): The positions of the agent over time (shape: [T, 2]).
        """
        if value is not None:
            self._position = np.asarray(value, dtype=np.float32)
        else:
            self._position = None

    @property
    def speed(self) -> np.ndarray | None:
        """np.ndarray: The velocities of the agent over time (shape: [T,])."""
        return self._speed

    @speed.setter
    def speed(self, value: np.ndarray | None) -> None:
        """Sets the velocities of the agent.

        Args:
            value (np.ndarray): The velocities of the agent over time (shape: [T,]).
        """
        if value is not None:
            self._speed = np.asarray(value, dtype=np.float32)
        else:
            self._speed = None

    @property
    def heading(self) -> np.ndarray | None:
        """np.ndarray: The headings of the agent over time (shape: [T,])."""
        return self._heading

    @heading.setter
    def heading(self, value: np.ndarray | None) -> None:
        """Sets the headings of the agent.

        Args:
            value (np.ndarray): The headings of the agent over time (shape: [T,]).
        """
        if value is not None:
            self._heading = np.asarray(value, dtype=np.float32)
        else:
            self._heading = None

    @property
    def length(self) -> np.ndarray | None:
        """np.ndarray or None: The lengths of the agent over time (shape: [T,])."""
        return self._length

    @length.setter
    def length(self, value: np.ndarray | None) -> None:
        """Sets the lengths of the agent.

        Args:
            value (np.ndarray): The lengths of the agent over time (shape: [T,]).
        """
        if value is not None:
            self._length = np.asarray(value, dtype=np.float32)
        else:
            self._length = None

    @property
    def width(self) -> np.ndarray | None:
        """np.ndarray or None: The widths of the agent over time (shape: [T,])."""
        return self._width

    @width.setter
    def width(self, value: np.ndarray | None) -> None:
        """Sets the widths of the agent.

        Args:
            value (np.ndarray): The widths of the agent over time (shape: [T,]).
        """
        if value is not None:
            self._height = np.asarray(value, dtype=np.float32)
        else:
            self._width = None

    @property
    def height(self) -> np.ndarray | None:
        """np.ndarray or None: The heights of the agent over time (shape: [T,])."""
        return self._height

    @height.setter
    def height(self, value: np.ndarray | None) -> None:
        """Sets the heights of the agent.

        Args:
            value (np.ndarray): The heights of the agent over time (shape: [T,]).
        """
        if value is not None:
            self._height = np.asarray(value, dtype=np.float32)
        else:
            self._height = None

    @property
    def agent_type(self) -> AgentType | None:
        """str: The type of the agent."""
        return self._agent_type

    @agent_type.setter
    def agent_type(self, value: AgentType | None) -> None:
        """Sets the type of the agent.

        Args:
            value (str): The type of the agent.
        """
        if value is not None:
            self._agent_type = value
        else:
            self._agent_type = None

    @property
    def is_stationary(self) -> bool | None:
        """Bool or None: Whether the agent is stationary (True/False), or None if unknown."""
        if self.speed is None:
            self._is_stationary = None
        else:
            self._is_stationary = self.speed.mean() < self._stationary_speed
        return self._is_stationary

    @property
    def stationary_speed(self) -> float:
        """float: The speed threshold below which the agent is considered stationary."""
        return self._stationary_speed

    @stationary_speed.setter
    def stationary_speed(self, value: float | None) -> None:
        """Sets the stationary speed threshold.

        Args:
            value (float): The speed threshold below which the agent is considered stationary.
        """
        if value is not None:
            self._stationary_speed = float(value)
        else:
            self._stationary_speed = 0.1

    @property
    def in_conflict_point(self) -> bool:
        """bool: Whether the agent is in a conflict point."""
        if self._dists_to_conflict is None:
            self._in_conflict_point = False
        else:
            self._in_conflict_point = np.any(
                self._dists_to_conflict <= self._agent_to_conflict_point_max_distance,
            ).__bool__()
        return self._in_conflict_point

    @property
    def agent_to_conflict_point_max_distance(self) -> float | None:
        """float: The maximum distance to a conflict point."""
        return self._agent_to_conflict_point_max_distance

    @agent_to_conflict_point_max_distance.setter
    def agent_to_conflict_point_max_distance(self, value: float | None) -> None:
        """Sets the maximum distance to a conflict point.

        Args:
            value (float): The maximum distance to a conflict point.
        """
        if value is not None:
            self._agent_to_conflict_point_max_distance = float(value)
        else:
            self._agent_to_conflict_point_max_distance = 0.5  # Default value

    @property
    def dists_to_conflict(self) -> np.ndarray | None:
        """np.ndarray: The distances to conflict points (shape: [T,])."""
        return self._dists_to_conflict

    @dists_to_conflict.setter
    def dists_to_conflict(self, value: np.ndarray | None) -> None:
        """Sets the distances to conflict points.

        Args:
            value (np.ndarray | None): The distances to conflict points (shape: [T,]).
        """
        if value is not None:
            self._dists_to_conflict = np.asarray(value, dtype=np.float32)
        else:
            self._dists_to_conflict = None

    @property
    def lane(self) -> np.ndarray | None:
        """np.ndarray or None: The lane of the agent, if available."""
        return self._lane

    @lane.setter
    def lane(self, value: np.ndarray | None) -> None:
        """Sets the lane of the agent.

        Args:
            value (np.ndarray or None): The lane of the agent, if available.
        """
        if value is not None:
            self._lane = np.asarray(value, dtype=np.float32)
        else:
            self._lane = None

    def reset(self) -> None:
        """Resets all agent attributes to their default values."""
        self._position = None
        self._speed = None
        self._heading = None
        self._dists_to_conflict = None
        self._stationary_speed = 0.1  # Default stationary speed threshold
        self._agent_to_conflict_point_max_distance = 0.5  # Default max distance to conflict point
        self._lane = None
        self._length = None
        self._width = None
        self._height = None
        self._agent_type = None
