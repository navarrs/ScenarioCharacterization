"""Defines Autonomoud Driving (AD) types."""

from enum import Enum


class AgentType(Enum):
    """Agent Types for WOMD."""

    TYPE_UNSET = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_CYCLIST = 3
    TYPE_OTHER = 4


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
    # // States for traffic signals with arrows.
    LANE_STATE_ARROW_STOP = 1
    LANE_STATE_ARROW_CAUTION = 2
    LANE_STATE_ARROW_GO = 3
    # // Standard round traffic signals.
    LANE_STATE_STOP = 4
    LANE_STATE_CAUTION = 5
    LANE_STATE_GO = 6
    # // Flashing light signals.
    LANE_STATE_FLASHING_STOP = 7
    LANE_STATE_FLASHING_CAUTION = 8
