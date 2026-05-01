"""Autonomous Driving (AD) map type definitions for WOMD."""

from enum import Enum
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


class LaneType(Enum):
    """Lane Types for WOMD."""

    TYPE_UNDEFINED = 0
    TYPE_FREEWAY = 1
    TYPE_SURFACE_STREET = 2
    TYPE_BIKE_LANE = 3


class RoadLineType(Enum):
    """Road line Types for WOMD."""

    TYPE_UNKNOWN = 0
    TYPE_BROKEN_SINGLE_WHITE = 1
    TYPE_SOLID_SINGLE_WHITE = 2
    TYPE_SOLID_DOUBLE_WHITE = 3
    TYPE_BROKEN_SINGLE_YELLOW = 4
    TYPE_BROKEN_DOUBLE_YELLOW = 5
    TYPE_SOLID_SINGLE_YELLOW = 6
    TYPE_SOLID_DOUBLE_YELLOW = 7
    TYPE_PASSING_DOUBLE_YELLOW = 8


class RoadEdgeType(Enum):
    """Road edge Types for WOMD."""

    TYPE_UNKNOWN = 0
    # Physical road boundary that doesn't have traffic on the other side (e.g.,
    # a curb or the k-rail on the right side of a freeway).
    TYPE_ROAD_EDGE_BOUNDARY = 1
    # Physical road boundary that separates the car from other traffic
    # (e.g. a k-rail or an island).
    TYPE_ROAD_EDGE_MEDIAN = 2


class PolylineType(Enum):
    """Polyline Types for WOMD."""

    # for lane
    TYPE_UNDEFINED = -1
    TYPE_FREEWAY = 1
    TYPE_SURFACE_STREET = 2
    TYPE_BIKE_LANE = 3
    # for roadline
    TYPE_BROKEN_SINGLE_WHITE = 6
    TYPE_SOLID_SINGLE_WHITE = 7
    TYPE_SOLID_DOUBLE_WHITE = 8
    TYPE_BROKEN_SINGLE_YELLOW = 9
    TYPE_BROKEN_DOUBLE_YELLOW = 10
    TYPE_SOLID_SINGLE_YELLOW = 11
    TYPE_SOLID_DOUBLE_YELLOW = 12
    TYPE_PASSING_DOUBLE_YELLOW = 13
    # for roadedge
    TYPE_ROAD_EDGE_BOUNDARY = 15
    TYPE_ROAD_EDGE_MEDIAN = 16
    # for stopsign
    TYPE_STOP_SIGN = 17
    # for crosswalk
    TYPE_CROSSWALK = 18
    # for speed bump
    TYPE_SPEED_BUMP = 19


class SignalState(Enum):
    """Traffic Signal States for WOMD."""

    LANE_STATE_UNKNOWN = 0
    # States for traffic signals with arrows.
    LANE_STATE_ARROW_STOP = 1
    LANE_STATE_ARROW_CAUTION = 2
    LANE_STATE_ARROW_GO = 3
    # Standard round traffic signals.
    LANE_STATE_STOP = 4
    LANE_STATE_CAUTION = 5
    LANE_STATE_GO = 6
    # Flashing light signals.
    LANE_STATE_FLASHING_STOP = 7
    LANE_STATE_FLASHING_CAUTION = 8


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
            lane_to_agent_metadata (NDArray[np.float32]): The lane to agent metadata of shape (N, T, D=6) or (T, D=6).
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
