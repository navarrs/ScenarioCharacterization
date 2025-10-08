from collections.abc import Callable
from enum import Enum
from typing import Annotated, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BeforeValidator

from characterization.utils.ad_types import AgentType

SMALL_EPS = 1e-6
BIG_EPS = 1e6
SUPPORTED_SCENARIO_TYPES = ["gt", "ho"]
SUPPORTED_CRITERIA = ["critical", "average"]
MIN_VALID_POINTS = 2


def mph_to_ms(mph: float) -> float:
    """Converts miles per hour (mph) to meters per second (m/s).

    Args:
        mph (float): Speed in miles per hour.

    Returns:
        float: Speed in meters per second.
    """
    return mph * 0.44704  # 1 mph = 0.44704 m/s


# Validator factory
def validate_array(
    expected_dtype: Any,  # noqa: ANN401
    expected_ndim: int,
) -> Callable[[Any], NDArray]:  # pyright: ignore[reportMissingTypeArgument]
    """Factory function to create a validator for numpy arrays with specific dtype and ndim."""

    def _validator(v: Any) -> NDArray:  # noqa: ANN401 pyright: ignore[reportMissingTypeArgument]
        if not isinstance(v, np.ndarray):
            error_message = f"Expected a numpy.ndarray, got {type(v)}"
            raise TypeError(error_message)
        if v.dtype != expected_dtype:
            error_message = f"Expected dtype {expected_dtype}, got {v.dtype}"
            raise TypeError(error_message)
        if v.ndim != expected_ndim:
            error_message = f"Expected {expected_ndim}D array, got {v.ndim}D"
            raise ValueError(error_message)
        return v

    return _validator


# Reusable types
BooleanNDArray2D = Annotated[NDArray[np.bool_], BeforeValidator(validate_array(np.bool_, 2))]
BooleanNDArray3D = Annotated[NDArray[np.bool_], BeforeValidator(validate_array(np.bool_, 3))]
Float64NDArray3D = Annotated[NDArray[np.float64], BeforeValidator(validate_array(np.float64, 3))]
Float32NDArray4D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 4))]
Float32NDArray3D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 3))]
Float32NDArray2D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 2))]
Float32NDArray1D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 1))]
Int32NDArray1D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 1))]
Int32NDArray2D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 2))]
Int64NDArray2D = Annotated[NDArray[np.int64], BeforeValidator(validate_array(np.int64, 2))]


class InteractionStatus(Enum):
    """Enumeration for interaction status."""

    UNKNOWN = -1
    COMPUTED_OK = 0
    PARTIAL_INVALID_HEADING = 1
    MASK_NOT_VALID = 2
    DISTANCE_TOO_FAR = 3
    STATIONARY = 4


class ReturnCriterion(Enum):
    """Enumeration for return criteria."""

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

    _agent_trajectory: NDArray[np.float32]

    def __init__(self, trajectory: NDArray[np.float32]) -> None:
        """Initializes the AgentTrajectoryMasker with trajectory data.

        Args:
            trajectory (np.ndarray): The trajectory data of shape (N, T, D=10) or (T, D=10).
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


class LaneMasker:
    """Masks for indexing lane data from the reformatted by the dataloader classes.

    The class expects an input of shape (N, L, T, D=6) or (L, T, D=6) where N is the number of agents, L is the number
    of lanes, and T is the number of timesteps. D is the number of features per lane point, organized as follows:
    timesteps and D is the number of features per lane point, organized as follows:
        idx 0: closest lane distance to the agent in meters.
        idx 1: lane point index of the closest lane point to the agent.
        idx 2 to 4: the lane point's (x, y, z) coordinates.
        idx 5: lane index.
    """

    # Lane Distances
    _LANE_DISTS: ClassVar[list[bool]] = [True, False, False, False, False, False]

    # Lane Point Index
    _LANE_POINT_IDX: ClassVar[list[bool]] = [False, True, False, False, False, False]

    # Lane Point (x, y, z) position masks
    _LANE_POINT_XYZ_POS: ClassVar[list[bool]] = [False, False, True, True, True, False]
    _LANE_POINT_XY_POS: ClassVar[list[bool]] = [False, False, True, True, False, False]

    # Lane Index
    _LANE_IDX: ClassVar[list[bool]] = [False, False, False, False, False, True]

    # Lane and distance
    _LANE_DIST_AND_IDX: ClassVar[list[bool]] = [True, False, False, False, False, True]

    _lane_to_agent_metadata: NDArray[np.float32]

    def __init__(self, lane_to_agent_metadata: NDArray[np.float32]) -> None:
        """Initializes the LaneMasker with lane to agent metadata.

        Args:
            lane_to_agent_metadata (np.ndarray): The lane to agent metadata of shape (N, T, D=6) or (T, D=6).
        """
        self._lane_to_agent_metadata = lane_to_agent_metadata

    # Lane metadata accessors
    @property
    def lane_to_agent_metadata(self) -> NDArray[np.float32]:
        """Returns the lane to agent metadata."""
        return self._lane_to_agent_metadata

    @property
    def lane_dists(self) -> NDArray[np.float32]:
        """Returns the closest lane distances to the agent."""
        return self._lane_to_agent_metadata[..., self._LANE_DISTS]

    @property
    def lane_point_idx(self) -> NDArray[np.float32]:
        """Returns the lane point indices of the closest lane points to the agent."""
        return self._lane_to_agent_metadata[..., self._LANE_POINT_IDX]

    @property
    def lane_point_xyz_pos(self) -> NDArray[np.float32]:
        """Returns the (x, y, z) position of the closest lane points to the agent."""
        return self._lane_to_agent_metadata[..., self._LANE_POINT_XYZ_POS]

    @property
    def lane_point_xy_pos(self) -> NDArray[np.float32]:
        """Returns the (x, y) position of the closest lane points to the agent."""
        return self._lane_to_agent_metadata[..., self._LANE_POINT_XY_POS]

    @property
    def lane_idx(self) -> NDArray[int]:
        """Returns the lane indices of the closest lanes to the agent."""
        return self._lane_to_agent_metadata[..., self._LANE_IDX].astype(int)

    @property
    def lane_dist_and_idx(self) -> NDArray[np.float32]:
        """Returns the closest lane distances and lane indices to the agent."""
        return self._lane_to_agent_metadata[..., self._LANE_DIST_AND_IDX]


class InteractionAgent:
    """Class representing an agent for interaction feature computation."""

    def __init__(self) -> None:
        """Initializes an InteractionAgent and resets all attributes."""
        self.reset()

    @property
    def position(self) -> np.ndarray:
        """np.ndarray: The positions of the agent over time (shape: [T, 2])."""
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        """Sets the positions of the agent.

        Args:
            value (np.ndarray): The positions of the agent over time (shape: [T, 2]).
        """
        self._position = np.asarray(value, dtype=np.float32)

    @property
    def speed(self) -> np.ndarray:
        """np.ndarray: The velocities of the agent over time (shape: [T,])."""
        return self._speed

    @speed.setter
    def speed(self, value: np.ndarray) -> None:
        """Sets the velocities of the agent.

        Args:
            value (np.ndarray): The velocities of the agent over time (shape: [T,]).
        """
        self._speed = np.asarray(value, dtype=np.float32)

    @property
    def heading(self) -> np.ndarray:
        """np.ndarray: The headings of the agent over time (shape: [T,])."""
        return self._heading

    @heading.setter
    def heading(self, value: np.ndarray) -> None:
        """Sets the headings of the agent.

        Args:
            value (np.ndarray): The headings of the agent over time (shape: [T,]).
        """
        self._heading = np.asarray(value, dtype=np.float32)

    @property
    def length(self) -> np.ndarray:
        """np.ndarray: The lengths of the agent over time (shape: [T,])."""
        return self._length

    @length.setter
    def length(self, value: np.ndarray) -> None:
        """Sets the lengths of the agent.

        Args:
            value (np.ndarray): The lengths of the agent over time (shape: [T,]).
        """
        self._length = np.asarray(value, dtype=np.float32)

    @property
    def width(self) -> np.ndarray:
        """np.ndarray: The widths of the agent over time (shape: [T,])."""
        return self._width

    @width.setter
    def width(self, value: np.ndarray) -> None:
        """Sets the widths of the agent.

        Args:
            value (np.ndarray): The widths of the agent over time (shape: [T,]).
        """
        self._width = np.asarray(value, dtype=np.float32)

    @property
    def height(self) -> np.ndarray:
        """np.ndarray: The heights of the agent over time (shape: [T,])."""
        return self._height

    @height.setter
    def height(self, value: np.ndarray) -> None:
        """Sets the heights of the agent.

        Args:
            value (np.ndarray): The heights of the agent over time (shape: [T,]).
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
            self._dists_to_conflict <= self._agent_to_conflict_point_max_distance
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
    def dists_to_conflict(self) -> np.ndarray:
        """np.ndarray: The distances to conflict points (shape: [T,])."""
        return self._dists_to_conflict

    @dists_to_conflict.setter
    def dists_to_conflict(self, value: np.ndarray) -> None:
        """Sets the distances to conflict points.

        Args:
            value (np.ndarray | None): The distances to conflict points (shape: [T,]).
        """
        self._dists_to_conflict = np.asarray(value, dtype=np.float32)

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


class TrajectoryType(Enum):
    """Trajectory Types for WOMD."""

    TYPE_UNSET = -1
    TYPE_STATIONARY = 0
    TYPE_STRAIGHT = 1
    TYPE_STRAIGHT_RIGHT = 2
    TYPE_STRAIGHT_LEFT = 3
    TYPE_RIGHT_U_TURN = 4
    TYPE_RIGHT_TURN = 5
    TYPE_LEFT_U_TURN = 6
    TYPE_LEFT_TURN = 7


# Weights for different trajectory types are loosely set based on Figure 3 (a) of https://arxiv.org/pdf/2403.15098
# Weight per class is set as (100% - class frequency %) * 0.10
TRAJECTORY_TYPE_WEIGHTS = {
    TrajectoryType.TYPE_UNSET: 0.0,
    # Stationary agents correspond to less than 10% of the data
    TrajectoryType.TYPE_STATIONARY: 9.0,
    # Straight-moving agents correspond to ~50% of the data.
    TrajectoryType.TYPE_STRAIGHT: 5.0,
    # Straight-right agents correspond less than ~10% of the data.
    TrajectoryType.TYPE_STRAIGHT_RIGHT: 9.0,
    # Straight-left agents correspond to ~10% of the data.
    TrajectoryType.TYPE_STRAIGHT_LEFT: 9.0,
    # Right-turn agents correspond to less than 20% of the data.
    TrajectoryType.TYPE_RIGHT_TURN: 8.0,
    # Left-turn agents correspond to less than 20% of the data.
    TrajectoryType.TYPE_LEFT_TURN: 8.0,
    # Right-U-turn agents correspond to less than 10% of the data.
    TrajectoryType.TYPE_RIGHT_U_TURN: 9.0,
    # Left-U-turn agents correspond to less than 10% of the data.
    TrajectoryType.TYPE_LEFT_U_TURN: 9.0,
}
