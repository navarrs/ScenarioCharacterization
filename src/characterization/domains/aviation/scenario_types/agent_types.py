from enum import Enum
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from characterization.utils.common import XYZScale
from characterization.utils.constants import (
    FT_TO_KM,
    FT_TO_M,
    KM_TO_HM,
    KM_TO_M,
    KNOTS_TO_MS,
)
from characterization.utils.geometric_utils import wrap_angle


class AgentStateValidity(Enum):
    """Enumeration for agent state validity.

    Attributes:
        VALID: Indicates a valid state.
        INTERPOLATED: Indicates an interpolated state.
    """

    VALID = "[ORG]"
    INTERPOLATED = "[INT]"


class AgentType(Enum):
    """Enumeration for agent types.

    Attributes:
        AIRCRAFT: Represents an aircraft.
        VEHICLE: Represents a vehicle.
        UNKNOWN: Represents an unknown agent type.
    """

    AIRCRAFT = 0
    VEHICLE = 1
    UNKNOWN = 2

    # Special types for internal use
    TO_PREDICT = 3
    EGO_AGENT = 4


AGENT_COLORS = {
    AgentType.AIRCRAFT: "tomato",
    AgentType.VEHICLE: "royalblue",
    AgentType.UNKNOWN: "gray",
}

AGENT_TYPE_TO_STRING = {
    AgentType.AIRCRAFT: "AIRCRAFT",
    AgentType.VEHICLE: "VEHICLE",
    AgentType.UNKNOWN: "UNKNOWN",
}

STRING_TO_AGENT_TYPE = {
    "AIRCRAFT": AgentType.AIRCRAFT,
    "VEHICLE": AgentType.VEHICLE,
    "UNKNOWN": AgentType.UNKNOWN,
}

VALID_STATE_VALUE = 1
INVALID_STATE_VALUE = 0


def raw_to_agent_type(raw_type: str) -> AgentType:
    """Convert a raw agent-type string to :class:`AgentType`, falling back to ``UNKNOWN``.

    Args:
        raw_type: Raw agent type string as stored in scenario data.

    Returns:
        Corresponding :class:`AgentType`, or :attr:`AgentType.UNKNOWN` if unrecognised.
    """
    return STRING_TO_AGENT_TYPE.get(raw_type, AgentType.UNKNOWN)


class AgentPairType(Enum):
    """Agent pair types for aviation scenarios.

    Attributes:
        OTHER: Default / unrecognised combination.
        AIRCRAFT_AIRCRAFT: Two aircraft-like agents.
        AIRCRAFT_VEHICLE: One aircraft-like agent and one ground vehicle.
        VEHICLE_VEHICLE: Two ground vehicles.
    """

    OTHER = 0
    AIRCRAFT_AIRCRAFT = 1
    AIRCRAFT_VEHICLE = 2
    VEHICLE_VEHICLE = 3


_AIRCRAFT_LIKE: frozenset[AgentType] = frozenset({AgentType.AIRCRAFT, AgentType.EGO_AGENT, AgentType.TO_PREDICT})


def get_agent_pair_type(type_a: AgentType, type_b: AgentType) -> AgentPairType:
    """Determine the :class:`AgentPairType` for two aviation agents.

    Args:
        type_a: Type of the first agent.
        type_b: Type of the second agent.

    Returns:
        The :class:`AgentPairType` for the pair.
    """
    is_aircraft_a = type_a in _AIRCRAFT_LIKE
    is_aircraft_b = type_b in _AIRCRAFT_LIKE
    is_vehicle_a = type_a == AgentType.VEHICLE
    is_vehicle_b = type_b == AgentType.VEHICLE

    if is_aircraft_a and is_aircraft_b:
        return AgentPairType.AIRCRAFT_AIRCRAFT
    if (is_aircraft_a and is_vehicle_b) or (is_vehicle_a and is_aircraft_b):
        return AgentPairType.AIRCRAFT_VEHICLE
    if is_vehicle_a and is_vehicle_b:
        return AgentPairType.VEHICLE_VEHICLE
    return AgentPairType.OTHER


RAW_STATE_SIZE_NO_AIRCRAFT_TYPE = 13
RAW_STATE_SIZE_WITH_AIRCRAFT_TYPE = 14


class RawAgentTrajectory:
    """Masks for indexing trajectory data from the reformatted by the dataloader classes.

    The class expects a raw input of type (N, T, D) or (T, D), where N is the number of agents, T is the number of
    timesteps and D is the number of features per trajectory point.

    For D = 13, the features are organized as follows:
        Frame,ID,Altitude,Speed,Heading,Lat,Lon,Range,Bearing,Type,Interp,x,y
        idx 0: the frame ID
        idx 1: the agent ID
        idx 2: the agent's altitude in feet
        idx 3: the agent's speed in knots
        idx 4: the agent's heading in degrees
        idx 5: the agent's latitude in decimal degrees
        idx 6: the agent's longitude in decimal degrees
        idx 7: the agent's range from the airport in nautical miles
        idx 8: the agent's bearing from the airport in degrees
        idx 9: the agent type (Aircraft = 0, Vehicle = 1, Unknown=2)
        idx 10: valid mask (['ORG'] if valid ['INT'] if not)
        idx 11: the agent's x position in meters (local coordinates)
        idx 12: the agent's y position in meters (local coordinates)

    For D = 14, the features are organized as follows:
        Frame,ID,Altitude,Speed,Heading,Lat,Lon,Range,Bearing,Type,AcType,Interp,x,y,
        idx 0: the frame ID
        idx 1: the agent ID
        idx 2: the agent's altitude in feet
        idx 3: the agent's speed in knots
        idx 4: the agent's heading in degrees
        idx 5: the agent's latitude in decimal degrees
        idx 6: the agent's longitude in decimal degrees
        idx 7: the agent's range from the airport in nautical miles
        idx 8: the agent's bearing from the airport in degrees
        idx 9: the agent type (Aircraft = 0, Vehicle = 1, Unknown=2)
        idx 10: the aircraft type
        idx 11: valid mask (['ORG'] if valid ['INT'] if not)
        idx 12: the agent's x position in meters (local coordinates)
        idx 13: the agent's y position in meters (local coordinates)
    """

    # Agent information
    _frame_id: NDArray[np.float32]
    _agent_id: NDArray[np.float32]
    _agent_type: NDArray[np.object_]
    _aircraft_type: NDArray[np.object_] | None
    _agent_trajectory: NDArray[np.float32]

    # Information that gets handled separately

    # Frame ID index
    _TRAJECTORY_FRAME_ID: ClassVar[int] = 0

    # Agent ID index
    _TRAJECTORY_AGENT_ID: ClassVar[int] = 1

    # Agent type index
    _TRAJECTORY_AGENT_TYPE: ClassVar[int] = 9

    # Aircraft type index (only present when D=14)
    _TRAJECTORY_AIRCRAFT_TYPE: ClassVar[int] = 10

    # Raw agent state information (altitude, speed, heading, lat, lon, range, bearing, valid, x, y)
    # fmt: off
    _TRAJECTORY_RAW_STATE: ClassVar[list[bool]] = [
        False, False, True, True, True, True, True, True, True, False, True, True, True,
    ]

    # Raw agent state information (altitude, speed, heading, lat, lon, range, bearing, valid, x, y)
    # when an extra aircraft_type field is present at idx 10.
    _TRAJECTORY_RAW_STATE_WITH_AIRCRAFT_TYPE: ClassVar[list[bool]] = [
        False, False, True, True, True, True, True, True, True, False, False, True, True, True,
    ]

    _SUPPORTED_TRAJECTORY_DIMS: ClassVar[set[int]] = {
        RAW_STATE_SIZE_NO_AIRCRAFT_TYPE, RAW_STATE_SIZE_WITH_AIRCRAFT_TYPE,
    }
    # State masks based on the new order (x, y, altitude, heading, speed, lat, lon, range, bearing, valid
    # _STATE_ORDER reindexes the raw state columns: [x=8, y=9, alt=0, hdg=2, spd=1, lat=3, lon=4, rng=5, brg=6, valid=7]
    _STATE_ORDER: ClassVar[list[int]] = [8, 9, 0, 2, 1, 3, 4, 5, 6, 7]
    _TRAJECTORY_ALTITUDE: ClassVar[list[bool]] = [False, False, True, False, False, False, False, False, False, False]
    _TRAJECTORY_XY_POS: ClassVar[list[bool]] = [True, True, False, False, False, False, False, False, False, False]
    _TRAJECTORY_XYZ_POS: ClassVar[list[bool]] = [True, True, True, False, False, False, False, False, False, False]
    _TRAJECTORY_HEADING: ClassVar[list[bool]] = [False, False, False, True, False, False, False, False, False, False]
    _TRAJECTORY_SPEED: ClassVar[list[bool]] = [False, False, False, False, True, False, False, False, False, False]
    _TRAJECTORY_LAT: ClassVar[list[bool]] = [False, False, False, False, False, True, False, False, False, False]
    _TRAJECTORY_LON: ClassVar[list[bool]] = [False, False, False, False, False, False, True, False, False, False]
    _TRAJECTORY_LATLON: ClassVar[list[bool]] = [False, False, False, False, False, True, True, False, False, False]
    _TRAJECTORY_RANGE: ClassVar[list[bool]] = [False, False, False, False, False, False, False, True, False, False]
    _TRAJECTORY_BEARING: ClassVar[list[bool]] = [False, False, False, False, False, False, False, False, True, False]
    _TRAJECTORY_RANGE_BEARING: ClassVar[list[bool]] = [False, False, False, False, False, False, False, True, True, False] # noqa: E501
    _TRAJECTORY_VALID: ClassVar[list[bool]] = [False, False, False, False, False, False, False, False, False, True]
    # fmt: on

    def __init__(self, trajectory: NDArray[np.float32]) -> None:
        """Initialize the RawAgentTrajectory with trajectory data.

        Args:
            trajectory: Trajectory data of shape (N, T, D) or (T, D), with D in {13, 14}.
        """
        if trajectory.shape[-1] not in self._SUPPORTED_TRAJECTORY_DIMS:
            error_message = (
                f"Trajectory must be of size {RAW_STATE_SIZE_NO_AIRCRAFT_TYPE} or {RAW_STATE_SIZE_WITH_AIRCRAFT_TYPE}."
                f"Received: {trajectory.shape[-1]}"
            )
            raise ValueError(error_message)

        self._frame_id = trajectory[..., self._TRAJECTORY_FRAME_ID]
        self._agent_id = trajectory[..., self._TRAJECTORY_AGENT_ID]

        # One agent type per agent
        self._agent_type = np.array(
            [AgentType(agent_type) for agent_type in trajectory[..., self._TRAJECTORY_AGENT_TYPE].flatten()],
        )

        # One aircraft type per agent (only present when D=14)
        self._aircraft_type = None
        if trajectory.shape[-1] == RAW_STATE_SIZE_WITH_AIRCRAFT_TYPE:
            self._aircraft_type = trajectory[..., self._TRAJECTORY_AIRCRAFT_TYPE].flatten().astype(np.object_)

        raw_state_mask = (
            self._TRAJECTORY_RAW_STATE
            if trajectory.shape[-1] == RAW_STATE_SIZE_NO_AIRCRAFT_TYPE
            else self._TRAJECTORY_RAW_STATE_WITH_AIRCRAFT_TYPE
        )
        raw_trajectory = trajectory[..., raw_state_mask]
        # We return the state in the new order (x, y, altitude, heading, speed, lat, lon, range, bearing, valid)
        self._agent_trajectory = raw_trajectory[..., self._STATE_ORDER]

        # Map valid states to 1 and interpolated states to 0
        valid = self._agent_trajectory[..., self._TRAJECTORY_VALID]
        valid_mask = np.where(valid == AgentStateValidity.VALID.value, VALID_STATE_VALUE, INVALID_STATE_VALUE)
        self._agent_trajectory[..., self._TRAJECTORY_VALID] = valid_mask

        self._agent_trajectory = self._agent_trajectory.astype(np.float32)

    # Trajectory accessors
    @property
    def agent_state(self) -> NDArray[np.float32]:
        """Returns the agent state data."""
        return self._agent_trajectory

    @property
    def agent_state_size(self) -> int:
        """Returns the size of the agent state data."""
        return self._agent_trajectory.shape[-1]

    @property
    def altitude(self) -> NDArray[np.float32]:
        """Returns the altitude data."""
        return self._agent_trajectory[..., self._TRAJECTORY_ALTITUDE]

    @property
    def xy_position(self) -> NDArray[np.float32]:
        """Returns the x and y position data."""
        return self._agent_trajectory[..., self._TRAJECTORY_XY_POS]

    @property
    def xy_position_mask(self) -> list[bool]:
        """Returns the mask for the x and y position data."""
        return self._TRAJECTORY_XY_POS

    @property
    def xyz_position(self) -> NDArray[np.float32]:
        """Returns the altitude, x and y position data."""
        return self._agent_trajectory[..., self._TRAJECTORY_XYZ_POS]

    @property
    def xyz_position_mask(self) -> list[bool]:
        """Returns the mask for the x, y, and z position data."""
        return self._TRAJECTORY_XYZ_POS

    @property
    def heading(self) -> NDArray[np.float32]:
        """Returns the heading data."""
        return self._agent_trajectory[..., self._TRAJECTORY_HEADING]

    @property
    def heading_mask(self) -> list[bool]:
        """Returns the mask for the heading data."""
        return self._TRAJECTORY_HEADING

    @property
    def speed(self) -> NDArray[np.float32]:
        """Returns the speed data."""
        return self._agent_trajectory[..., self._TRAJECTORY_SPEED]

    @property
    def frame_id(self) -> NDArray[np.float32]:
        """Returns the frame ID data."""
        return self._frame_id

    @property
    def num_frames(self) -> int:
        """Returns the number of frames in the trajectory."""
        return self._frame_id.shape[0] if len(self._frame_id.shape) == 1 else self._frame_id.shape[1]

    @property
    def agent_id(self) -> NDArray[np.float32]:
        """Returns the agent ID data."""
        return self._agent_id

    @property
    def agent_type(self) -> NDArray[np.object_]:
        """Returns the agent type data."""
        return self._agent_type

    @property
    def aircraft_type(self) -> NDArray[np.object_] | None:
        """Returns the aircraft type data, or None if not present in the trajectory."""
        return self._aircraft_type

    @property
    def valid(self) -> NDArray[np.float32]:
        """Returns the valid mask data."""
        return self._agent_trajectory[..., self._TRAJECTORY_VALID]

    @property
    def valid_mask(self) -> list[bool]:
        """Returns the mask for the valid data."""
        return self._TRAJECTORY_VALID

    def normalize_altitude(self, max_altitude: float, min_altitude: float, *, use_minmax_scaling: bool = True) -> None:
        """Normalize the altitude data in the trajectory.

        Args:
            max_altitude: Maximum altitude for normalization.
            min_altitude: Minimum altitude for normalization.
            use_minmax_scaling: If True, applies min-max scaling. If False, applies zero-centering only.
        """
        altitudes = self._agent_trajectory[..., self._TRAJECTORY_ALTITUDE]
        normalized_altitudes = altitudes - min_altitude
        if use_minmax_scaling:
            normalized_altitudes = normalized_altitudes / (max_altitude - min_altitude)
        self._agent_trajectory[..., self._TRAJECTORY_ALTITUDE] = normalized_altitudes

    def update_imputed_state(self, new_agent_trajectory: NDArray[np.float32], new_frames: NDArray[np.float32]) -> None:
        """Update the entire agent state data in the trajectory.

        Args:
            new_agent_trajectory: New agent state data to set.
            new_frames: New frame ID data to set.
        """
        self._agent_trajectory = new_agent_trajectory
        self._frame_id = new_frames

    def unwrap_heading(self) -> None:
        """Convert heading from degrees to unwrapped radians to enable correct circular interpolation.

        Must be called before imputing missing data. After imputation, call standardize_state_units
        to re-wrap the heading back to [-pi, pi].
        """
        heading_rad = np.deg2rad(self._agent_trajectory[..., self._TRAJECTORY_HEADING])
        self._agent_trajectory[..., self._TRAJECTORY_HEADING] = np.unwrap(heading_rad)

    def standardize_state_units(self, scale_factor: XYZScale) -> None:
        """Standardize the units of the agent state data in the trajectory.

        Args:
            scale_factor: Scale factor to convert altitude from feet to the desired unit.
        """
        # Wrap heading back to [-pi, pi] (heading is already in radians from unwrap_heading)
        self._agent_trajectory[..., self._TRAJECTORY_HEADING] = wrap_angle(
            self._agent_trajectory[..., self._TRAJECTORY_HEADING],
        )
        match scale_factor:
            case XYZScale.KM:
                # Only need to scale the altitude since x and y are already in km
                self._agent_trajectory[..., self._TRAJECTORY_ALTITUDE] *= FT_TO_KM
            case XYZScale.HM:
                self._agent_trajectory[..., self._TRAJECTORY_ALTITUDE] *= FT_TO_KM * KM_TO_HM
                self._agent_trajectory[..., self._TRAJECTORY_XY_POS] *= KM_TO_HM
            case XYZScale.M:
                self._agent_trajectory[..., self._TRAJECTORY_ALTITUDE] *= FT_TO_M
                self._agent_trajectory[..., self._TRAJECTORY_XY_POS] *= KM_TO_M


class AgentTrajectory:
    """Masks for indexing trajectory data from the reformatted by the dataloader classes.

    The class expects a raw input of type (N, T, D=10) or (T, D=10) where N is the number of agents, T is the number of
    timesteps and D is the number of features per trajectory point, organized as follows:
        idx 0: the agent's x position in meters (local coordinates)
        idx 1: the agent's y position in meters (local coordinates)
        idx 2: the agent's altitude in meters
        idx 3: the agent's heading in radians
        idx 4: the agent's speed in knots
        idx 5: the agent's latitude in decimal degrees
        idx 6: the agent's longitude in decimal degrees
        idx 7: the agent's range from the airport in nautical miles
        idx 8: the agent's bearing from the airport in degrees
        idx 9: valid mask (1 if valid, 0 if not)
    """

    # Agent information
    _agent_trajectory: NDArray[np.float32]

    # State masks based on the new order (x, y, altitude, heading, speed, lat, lon, range, bearing, valid
    # fmt: off
    _TRAJECTORY_ALTITUDE: ClassVar[list[bool]] = [False, False, True, False, False, False, False, False, False, False]
    _TRAJECTORY_XY_POS: ClassVar[list[bool]] = [True, True, False, False, False, False, False, False, False, False]
    _TRAJECTORY_XYZ_POS: ClassVar[list[bool]] = [True, True, True, False, False, False, False, False, False, False]
    _TRAJECTORY_HEADING: ClassVar[list[bool]] = [False, False, False, True, False, False, False, False, False, False]
    _TRAJECTORY_SPEED: ClassVar[list[bool]] = [False, False, False, False, True, False, False, False, False, False]
    _TRAJECTORY_LAT: ClassVar[list[bool]] = [False, False, False, False, False, True, False, False, False, False]
    _TRAJECTORY_LON: ClassVar[list[bool]] = [False, False, False, False, False, False, True, False, False, False]
    _TRAJECTORY_LATLON: ClassVar[list[bool]] = [False, False, False, False, False, True, True, False, False, False]
    _TRAJECTORY_RANGE: ClassVar[list[bool]] = [False, False, False, False, False, False, False, True, False, False]
    _TRAJECTORY_BEARING: ClassVar[list[bool]] = [False, False, False, False, False, False, False, False, True, False]
    _TRAJECTORY_RANGE_BEARING: ClassVar[list[bool]] = [False, False, False, False, False, False, False, True, True, False] # noqa: E501
    _TRAJECTORY_VALID: ClassVar[list[bool]] = [False, False, False, False, False, False, False, False, False, True]
    _TRAJECTORY_STATE: ClassVar[list[bool]] = [True, True, True, True, True, True, True, True, True, False]
    # fmt: on

    def __init__(self, trajectory: NDArray[np.float32]) -> None:
        """Initialize the AgentTrajectory with trajectory data.

        Args:
            trajectory: Trajectory data of shape (N, T, D=10) or (T, D=10).
        """
        self._agent_trajectory = trajectory

    # Trajectory accessors
    @property
    def agent_trajectory(self) -> NDArray[np.float32]:
        """Returns the agent state data."""
        return self._agent_trajectory

    @property
    def agent_state(self) -> NDArray[np.float32]:
        """Returns the agent state data."""
        return self._agent_trajectory[..., self._TRAJECTORY_STATE]

    @property
    def agent_state_size(self) -> int:
        """Returns the size of the agent state data."""
        return self._agent_trajectory.shape[-1]

    @property
    def altitude(self) -> NDArray[np.float32]:
        """Returns the altitude data."""
        return self._agent_trajectory[..., self._TRAJECTORY_ALTITUDE]

    @property
    def xy_position(self) -> NDArray[np.float32]:
        """Returns the x and y position data."""
        return self._agent_trajectory[..., self._TRAJECTORY_XY_POS]

    @property
    def xy_position_mask(self) -> list[bool]:
        """Returns the mask for the x and y position data."""
        return self._TRAJECTORY_XY_POS

    @property
    def xyz_position(self) -> NDArray[np.float32]:
        """Returns the altitude, x and y position data."""
        return self._agent_trajectory[..., self._TRAJECTORY_XYZ_POS]

    @property
    def xyz_position_mask(self) -> list[bool]:
        """Returns the mask for the x, y, and z position data."""
        return self._TRAJECTORY_XYZ_POS

    @property
    def heading(self) -> NDArray[np.float32]:
        """Returns the heading data."""
        return self._agent_trajectory[..., self._TRAJECTORY_HEADING]

    @property
    def heading_mask(self) -> list[bool]:
        """Returns the mask for the heading data."""
        return self._TRAJECTORY_HEADING

    @property
    def speed(self) -> NDArray[np.float32]:
        """Returns the speed data."""
        return self._agent_trajectory[..., self._TRAJECTORY_SPEED]

    @property
    def speed_mask(self) -> list[bool]:
        """Returns the mask for the speed data."""
        return self._TRAJECTORY_SPEED

    @property
    def altitude_mask(self) -> list[bool]:
        """Returns the mask for the altitude field."""
        return self._TRAJECTORY_ALTITUDE

    @property
    def latlon_mask(self) -> list[bool]:
        """Returns the mask for the latitude and longitude fields."""
        return self._TRAJECTORY_LATLON

    @property
    def range_bearing_mask(self) -> list[bool]:
        """Returns the mask for the range and bearing fields."""
        return self._TRAJECTORY_RANGE_BEARING

    @property
    def valid_mask(self) -> list[bool]:
        """Returns the mask for the valid field."""
        return self._TRAJECTORY_VALID

    @property
    def latitude(self) -> NDArray[np.float32]:
        """Returns the latitude data."""
        return self._agent_trajectory[..., self._TRAJECTORY_LAT]

    @property
    def longitude(self) -> NDArray[np.float32]:
        """Returns the longitude data."""
        return self._agent_trajectory[..., self._TRAJECTORY_LON]

    @property
    def latlon(self) -> NDArray[np.float32]:
        """Returns the latitude and longitude data."""
        return self._agent_trajectory[..., self._TRAJECTORY_LATLON]

    @property
    def range(self) -> NDArray[np.float32]:
        """Returns the range data."""
        return self._agent_trajectory[..., self._TRAJECTORY_RANGE]

    @property
    def bearing(self) -> NDArray[np.float32]:
        """Returns the bearing data."""
        return self._agent_trajectory[..., self._TRAJECTORY_BEARING]

    @property
    def valid(self) -> NDArray[np.float32]:
        """Returns the valid mask data."""
        return self._agent_trajectory[..., self._TRAJECTORY_VALID]

    def get_valid_mask(self) -> NDArray[np.bool_]:
        """Returns the valid mask for the trajectory."""
        return self._agent_trajectory[..., self._TRAJECTORY_VALID].squeeze(-1).astype(bool)

    def get_valid_speeds(self, *, speed_to_ms: float = KNOTS_TO_MS) -> NDArray[np.float32]:
        """Return valid speed samples converted to m/s.

        Args:
            trajectory: Agent trajectory accessor.
            speed_to_ms: Conversion factor from the stored speed unit to m/s.

        Returns:
            Shape ``(T_valid,)`` array of speeds in m/s.
        """
        mask = self.get_valid_mask()
        speeds = self.speed.squeeze(-1)
        return (speeds[mask] * speed_to_ms).astype(np.float32)

    def get_valid_positions(self, *, scale: float = 1.0) -> NDArray[np.float32]:
        """Return valid 3-D positions converted to specified scale.

        Args:
            scale: Conversion factor from the stored position unit to specified scale.

        Returns:
            Shape ``(T_valid, 3)`` array of [x, y, z] positions in specified scale.
        """
        mask = self.get_valid_mask()
        return (self.xyz_position[mask] * scale).astype(np.float32)
