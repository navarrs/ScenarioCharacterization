from enum import Enum
from typing import ClassVar

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from characterization.utils.common import XYZScale
from characterization.utils.constants import EPSILON, LARGE_FLOAT, SCALE_FACTOR_FROM_KM


class MapType(Enum):
    """Enumeration for map types.

    Attributes:
        BOUNDARY: Represents a boundary.
        HOLD_SHORT_LINE: Represents a hold short line.
        EXIT: Represents an exit.
        RUNWAY: Represents a runway.
        TAXIWAY: Represents a taxiway.
        RAMP: Represents a ramp.
        UNKNOWN: Represents an unknown map type.
    """

    BOUNDARY = 0
    HOLD_SHORT_LINE = 1
    EXIT = 2
    RUNWAY = 3
    TAXIWAY = 4
    RAMP = 5
    UNKNOWN = -1


MAP_COLORS_LIGHT = {
    MapType.BOUNDARY: "black",
    MapType.HOLD_SHORT_LINE: "coral",
    MapType.EXIT: "magenta",
    MapType.RUNWAY: "dodgerblue",
    MapType.TAXIWAY: "limegreen",
    MapType.RAMP: "orchid",
    MapType.UNKNOWN: "gray",
}

MAP_COLORS_DARK = {
    MapType.BOUNDARY: "black",
    MapType.HOLD_SHORT_LINE: "orangered",
    MapType.EXIT: "darkmagenta",
    MapType.RUNWAY: "midnightblue",
    MapType.TAXIWAY: "darkgreen",
    MapType.RAMP: "rebeccapurple",
    MapType.UNKNOWN: "dimgray",
}

# Default to light mode
MAP_COLORS = MAP_COLORS_LIGHT


class MapInfo:
    """Masks for indexing map data from the reformatted by the dataloader classes.

    The class expects a raw input of hold short lines, polylines, and an airport surface movement graph.

    Hold short lines are organized as follows:
        idx 0: latitude
        idx 1: longitude
        idx 2: x position in meters (local coordinates)
        idx 3: y position in meters (local coordinates)
        idx 4: semantic type
        idx 5: id

    Polylines are organized as follows:
        idx 0: start latitude
        idx 1: start longitude
        idx 2: start x position in meters (local coordinates)
        idx 3: start y position in meters (local coordinates)
        idx 4: end latitude
        idx 5: end longitude
        idx 6: end x position in meters (local coordinates)
        idx 7: end y position in meters (local coordinates)
        idx 8: semantic type
        idx 9: id

    The graph is a NetworkX graph whose nodes carry x, y (km), lat, lon, and node_type attributes. The stored graph has
    x/y scaled to xyz_scale units so they match all other coordinate fields.
    """

    _hold_lines: NDArray[np.float32]
    _polylines: NDArray[np.float32]
    _graph: nx.Graph  # pyright: ignore[reportMissingTypeArgument]

    # Hold lines: (lat, lon, x, y, semantic_type, id)
    _HOLD_LINES_LATLON: ClassVar[list[bool]] = [True, True, False, False, False, False]
    _HOLD_LINES_XY: ClassVar[list[bool]] = [False, False, True, True, False, False]
    _HOLD_LINES_ID: ClassVar[list[bool]] = [False, False, False, False, False, True]

    # Polylines: (lat_start, lon_start, x_start, y_start, lat_end, lon_end, x_end, y_end, semantic_type, id)
    _POLYLINE_TYPE: ClassVar[list[bool]] = [False, False, False, False, False, False, False, False, True, False]
    _POLYLINE_ID: ClassVar[list[bool]] = [False, False, False, False, False, False, False, False, False, True]

    _POLYLINE_LATLON: ClassVar[list[bool]] = [True, True, False, False, True, True, False, False, False, False]
    _POLYLINE_XY: ClassVar[list[bool]] = [False, False, True, True, False, False, True, True, False, False]

    def __init__(
        self,
        hold_lines: NDArray[np.float32],
        polylines: NDArray[np.float32],
        graph: nx.Graph,  # pyright: ignore[reportMissingTypeArgument]
        xyz_scale: XYZScale = XYZScale.HM,
    ) -> None:
        """Initialize the MapInfo with hold short lines, polylines, and surface movement graph.

        Args:
            hold_lines: Hold short lines data.
            polylines: Polylines data.
            graph: NetworkX graph of the airport surface movement topology. Node x/y attributes
                are in km; they will be scaled to xyz_scale units in the stored graph copy.
            xyz_scale: Coordinate scale used for agent trajectories. xy values are rescaled to match.
        """
        self._hold_lines = hold_lines
        self._polylines = polylines
        # This is the factor to convert the raw xy (km) vales to the desired scale
        self._xy_scale: float = SCALE_FACTOR_FROM_KM[xyz_scale]
        self._polyline_types = self._polylines[..., self._POLYLINE_TYPE].squeeze(-1).astype(np.int32)

        self._graph = graph.copy()
        for _, data in self._graph.nodes(data=True):
            data["x"] = data["x"] * self._xy_scale
            data["y"] = data["y"] * self._xy_scale

        # Get boundaries
        self._boundaries = self._polylines[self._polyline_types == MapType.BOUNDARY.value]

        # Get exits
        self._exits = self._polylines[self._polyline_types == MapType.EXIT.value]

        # Get runways
        self._runways = self._polylines[self._polyline_types == MapType.RUNWAY.value]

        # Get taxiways
        self._taxiways = self._polylines[self._polyline_types == MapType.TAXIWAY.value]

        self._ramps = self._polylines[self._polyline_types == MapType.RAMP.value]

    def _polyline_xy_with_dir(self, subset: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return (M, 4) array as [x, y, x_dir, y_dir] for a subset of polylines.

        x, y is the segment start point in xyz_scale units. x_dir, y_dir is the unit direction from start to end.

        Args:
            subset: Polyline rows, shape (M, 10) in the raw MapInfo format.

        Returns:
            Array of shape (M, 4).
        """
        raw_xy = subset[..., self._POLYLINE_XY]  # (M, 4) as [x_s, y_s, x_e, y_e] in km
        start = raw_xy[:, :2] * self._xy_scale
        end = raw_xy[:, 2:] * self._xy_scale
        diff = end - start
        norm = np.linalg.norm(diff, axis=-1, keepdims=True)
        direction = diff / np.clip(norm, EPSILON, LARGE_FLOAT)
        return np.concatenate([start, direction], axis=-1).astype(np.float32)

    @property
    def graph(self) -> nx.Graph:  # pyright: ignore[reportMissingTypeArgument]
        """Returns the airport surface movement graph with x/y scaled to xyz_scale units."""
        return self._graph

    @property
    def hold_short_points_latlon(self) -> NDArray[np.float32]:
        """Returns the hold short lines lat/lon data."""
        return self._hold_lines[..., self._HOLD_LINES_LATLON]

    @property
    def hold_short_points_xy(self) -> NDArray[np.float32]:
        """Returns hold-short points as (K, 4) [x, y, x_dir, y_dir] in xyz_scale units. Direction is zero."""
        xy = self._hold_lines[..., self._HOLD_LINES_XY] * self._xy_scale  # (K, 2)
        zeros = np.zeros((xy.shape[0], 2), dtype=np.float32)
        return np.concatenate([xy, zeros], axis=-1).astype(np.float32)

    @property
    def hold_short_points_ids(self) -> NDArray[np.int64]:
        """Returns the hold short lines IDs data."""
        return self._hold_lines[..., self._HOLD_LINES_ID].squeeze(-1).astype(np.int64)

    @property
    def boundary_polylines_latlon(self) -> NDArray[np.float32]:
        """Returns the boundary lat/lon polylines data."""
        return self._boundaries[..., self._POLYLINE_LATLON]

    @property
    def boundary_polylines_xy(self) -> NDArray[np.float32]:
        """Returns boundary polylines as (M, 4) [x, y, x_dir, y_dir] in xyz_scale units."""
        return self._polyline_xy_with_dir(self._boundaries)

    @property
    def boundary_polylines_ids(self) -> NDArray[np.int64]:
        """Returns the boundary polyline IDs data."""
        return self._boundaries[..., self._POLYLINE_ID].squeeze(-1).astype(np.int64)

    @property
    def exit_polylines_latlon(self) -> NDArray[np.float32]:
        """Returns the exit lat/lon polylines data."""
        return self._exits[..., self._POLYLINE_LATLON]

    @property
    def exit_polylines_xy(self) -> NDArray[np.float32]:
        """Returns exit polylines as (M, 4) [x, y, x_dir, y_dir] in xyz_scale units."""
        return self._polyline_xy_with_dir(self._exits)

    @property
    def exit_polylines_ids(self) -> NDArray[np.int64]:
        """Returns the exit polyline IDs data."""
        return self._exits[..., self._POLYLINE_ID].squeeze(-1).astype(np.int64)

    @property
    def runway_polylines_latlon(self) -> NDArray[np.float32]:
        """Returns the runway lat/lon polylines data."""
        return self._runways[..., self._POLYLINE_LATLON]

    @property
    def runway_polylines_xy(self) -> NDArray[np.float32]:
        """Returns runway polylines as (M, 4) [x, y, x_dir, y_dir] in xyz_scale units."""
        return self._polyline_xy_with_dir(self._runways)

    @property
    def runway_polylines_ids(self) -> NDArray[np.int64]:
        """Returns the runway polyline IDs data."""
        return self._runways[..., self._POLYLINE_ID].squeeze(-1).astype(np.int64)

    @property
    def taxiway_polylines_latlon(self) -> NDArray[np.float32]:
        """Returns the taxiway lat/lon polylines data."""
        return self._taxiways[..., self._POLYLINE_LATLON]

    @property
    def taxiway_polylines_xy(self) -> NDArray[np.float32]:
        """Returns taxiway polylines as (M, 4) [x, y, x_dir, y_dir] in xyz_scale units."""
        return self._polyline_xy_with_dir(self._taxiways)

    @property
    def taxiway_polylines_ids(self) -> NDArray[np.int64]:
        """Returns the taxiway polyline IDs data."""
        return self._taxiways[..., self._POLYLINE_ID].squeeze(-1).astype(np.int64)

    @property
    def ramp_polylines_latlon(self) -> NDArray[np.float32]:
        """Returns the ramp lat/lon polylines data."""
        return self._ramps[..., self._POLYLINE_LATLON]

    @property
    def ramp_polylines_xy(self) -> NDArray[np.float32]:
        """Returns ramp polylines as (M, 4) [x, y, x_dir, y_dir] in xyz_scale units."""
        return self._polyline_xy_with_dir(self._ramps)

    @property
    def ramp_polylines_ids(self) -> NDArray[np.int64]:
        """Returns the ramp polyline IDs data."""
        return self._ramps[..., self._POLYLINE_ID].squeeze(-1).astype(np.int64)
