import pickle
from pathlib import Path
from typing import cast

import networkx as nx
from pydantic import BaseModel, computed_field
from pydantic_core import PydanticUndefined

from characterization.domains.aviation.schemas.airport_metadata import TimeMetadata
from characterization.domains.aviation.schemas.critical_probe import CriticalProbe
from characterization.schemas.scenario import BaseAgentData, BaseScenario, BaseScenarioMetadata
from characterization.schemas.types import (
    AgentTypeNDArray1D,
    BooleanNDArray1D,
    Float32NDArray1D,
    Float32NDArray2D,
    Int32NDArray1D,
    Int64NDArray1D,
    ObjectTypeNDArray1D,
)
from characterization.utils.common import SpeedUnits, XYZScale


class MapData(BaseModel):
    """Airport surface map data.

    All ``*_xy`` fields are in the same coordinate scale as agent trajectories (``xyz_scale`` at processing time:
    km or hm). The ``*_latlon`` fields are always in decimal degrees and are unaffected by scaling.

    Polyline ``*_xy`` fields store ``[x, y, x_dir, y_dir]`` where ``(x, y)`` is the segment midpoint and
    ``(x_dir, y_dir)`` is the unit direction vector from start to end. Hold-short points store ``[x, y, 0, 0]``
    (points have no direction).

    Attributes:
        airport_id: Airport identifier.
        hold_short_points_latlon: Shape (K, 2) hold short points as [lat, lon].
        hold_short_points_xy: Shape (K, 4) hold short points as [x, y, 0, 0] in xyz_scale units.
        hold_short_points_ids: Shape (K,) hold short point identifiers.
        boundary_polylines_latlon: Shape (M, 4) boundary polylines as [lat_start, lon_start, lat_end, lon_end].
        boundary_polylines_xy: Shape (M, 4) boundary polylines as [x, y, x_dir, y_dir] in xyz_scale units.
        boundary_polylines_ids: Shape (M,) boundary polyline identifiers.
        exit_polylines_latlon: Shape (L, 4) exit polylines as [lat_start, lon_start, lat_end, lon_end].
        exit_polylines_xy: Shape (L, 4) exit polylines as [x, y, x_dir, y_dir] in xyz_scale units.
        exit_polylines_ids: Shape (L,) exit polyline identifiers.
        runway_polylines_latlon: Shape (R, 4) runway polylines as [lat_start, lon_start, lat_end, lon_end].
        runway_polylines_xy: Shape (R, 4) runway polylines as [x, y, x_dir, y_dir] in xyz_scale units.
        runway_polylines_ids: Shape (R,) runway polyline identifiers.
        taxiway_polylines_latlon: Shape (T, 4) taxiway polylines as [lat_start, lon_start, lat_end, lon_end].
        taxiway_polylines_xy: Shape (T, 4) taxiway polylines as [x, y, x_dir, y_dir] in xyz_scale units.
        taxiway_polylines_ids: Shape (T,) taxiway polyline identifiers.
        ramp_polylines_latlon: Shape (P, 4) ramp polylines as [lat_start, lon_start, lat_end, lon_end].
        ramp_polylines_xy: Shape (P, 4) ramp polylines as [x, y, x_dir, y_dir] in xyz_scale units.
        ramp_polylines_ids: Shape (P,) ramp polyline identifiers.
        graph: NetworkX graph of the airport surface movement topology. Node x/y attributes are in xyz_scale units;
            lat/lon are in decimal degrees; node_type is an integer (1=taxiway junction, 3=runway, 4=other).
    """

    airport_id: str

    hold_short_points_latlon: Float32NDArray2D
    hold_short_points_xy: Float32NDArray2D
    hold_short_points_ids: Int64NDArray1D

    boundary_polylines_latlon: Float32NDArray2D
    boundary_polylines_xy: Float32NDArray2D
    boundary_polylines_ids: Int64NDArray1D

    exit_polylines_latlon: Float32NDArray2D
    exit_polylines_xy: Float32NDArray2D
    exit_polylines_ids: Int64NDArray1D

    runway_polylines_latlon: Float32NDArray2D
    runway_polylines_xy: Float32NDArray2D
    runway_polylines_ids: Int64NDArray1D

    taxiway_polylines_latlon: Float32NDArray2D
    taxiway_polylines_xy: Float32NDArray2D
    taxiway_polylines_ids: Int64NDArray1D

    ramp_polylines_latlon: Float32NDArray2D
    ramp_polylines_xy: Float32NDArray2D
    ramp_polylines_ids: Int64NDArray1D

    graph: nx.Graph  # pyright: ignore[reportMissingTypeArgument]

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    def to_pickle(self, save_dir: Path) -> None:
        """Serialize to a pickle file named ``{airport_id}.pkl`` inside ``save_dir``.

        Args:
            save_dir: Directory in which to write the pickle file.
        """
        scene_filepath = save_dir / f"{self.airport_id}.pkl"
        with scene_filepath.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, filepath: Path) -> "MapData":
        """Deserialize from a pickle file.

        Args:
            filepath: Path to the pickle file.

        Returns:
            The deserialized MapData instance.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)


class AgentData(BaseAgentData):
    """Agent data for aviation scenarios.

    Attributes:
        agent_ids: Agent identifiers including the ego agent.
        agent_types: Agent types (e.g. aircraft, vehicle).
        aircraft_types: Aircraft model/type for each agent (optional).
        agent_trajectories: Shape (N, T, 10) as [x, y, altitude, heading, speed, lat, lon, range, bearing, valid].
        agent_relevance: Shape (N,) relevance scores. If None, all agents are equally relevant.
    """

    agent_ids: Int32NDArray1D
    agent_types: AgentTypeNDArray1D
    aircraft_types: ObjectTypeNDArray1D | None = None

    @computed_field
    @property
    def num_agents(self) -> int:
        """Number of agents in the scenario."""
        return len(self.agent_ids)


class ScenarioMetadata(BaseScenarioMetadata):
    """Scenario metadata for aviation scenarios.

    Attributes:
        timestamps_seconds: Timestamps in seconds for each frame.
        time_metadata: Wall-clock time information for the scenario.
        ego_agent_id: Ego agent identifier.
        ego_agent_index: Ego agent index in the agent list.
        ego_selection_strategy: Strategy used to select the ego agent.
        xyz_scale: Coordinate unit of the x/y/z position fields in agent trajectories.
        speed_units: Unit of the speed field in agent trajectories.
    """

    timestamps_seconds: Float32NDArray1D
    time_metadata: TimeMetadata
    ego_agent_id: int
    ego_agent_index: int
    ego_selection_strategy: str

    xyz_scale: XYZScale = XYZScale.M
    speed_units: SpeedUnits = SpeedUnits.MS

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore from pickle, backfilling defaults for fields added after the pickle was created."""
        field_dict = cast("dict[str, object]", state.get("__dict__", state))
        for field_name, field_info in self.__class__.model_fields.items():
            if field_name not in field_dict and field_info.default is not PydanticUndefined:
                field_dict[field_name] = field_info.default
        super().__setstate__(state)

    @computed_field
    @property
    def duration_s(self) -> float:
        """Total duration of the scenario in seconds."""
        if self.timestamps_seconds is not None and len(self.timestamps_seconds) > 0:
            return float(self.timestamps_seconds[-1] - self.timestamps_seconds[0])
        err_msg = "timestamps_seconds is empty, cannot compute duration."
        raise ValueError(err_msg)


class AgentsToPredict(BaseModel):
    """Agents eligible for trajectory prediction in an aviation scenario.

    Attributes:
        agent_index: Indices (within the full agent array) for all agents valid to predict.
        agent_difficulty: Difficulty scores per agent.
        agent_type: Agent types.
        valid_to_be_ego: Boolean mask over agent_index indicating which agents are eligible for ego selection
            (i.e. have a valid observation at current_time_index).
        difficulty_scoring_strategy: Strategy used to score agent difficulty.
    """

    agent_index: Int32NDArray1D
    agent_difficulty: Float32NDArray1D
    agent_type: AgentTypeNDArray1D
    valid_to_be_ego: BooleanNDArray1D
    difficulty_scoring_strategy: str

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}


class Scenario(BaseScenario):
    """Aviation scenario with agents, trajectories, and airport map data.

    Inherits ``to_pickle``, ``from_pickle``, and ``__setstate__`` from ``BaseScenario``.

    Attributes:
        metadata: Scenario metadata.
        agent_data: Agent trajectories and types.
        agents_to_predict: Agents eligible for trajectory prediction.
        static_map_data: Airport surface map (optional).
        critical_probe: Counterfactual probe result (optional).
    """

    metadata: ScenarioMetadata
    agent_data: AgentData
    agents_to_predict: AgentsToPredict
    static_map_data: MapData | None = None
    critical_probe: CriticalProbe | None = None
