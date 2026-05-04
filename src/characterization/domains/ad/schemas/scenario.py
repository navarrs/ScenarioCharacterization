from typing import Any

from pydantic import BaseModel, NonNegativeInt, computed_field

from characterization.domains.ad.scenario_types import AgentType
from characterization.schemas.critical_probe import CriticalProbe
from characterization.schemas.scenario import BaseAgentData, BaseScenario, BaseScenarioMetadata
from characterization.schemas.types import (
    Float32NDArray1D,
    Float32NDArray2D,
    Float32NDArray3D,
    Float32NDArray4D,
    Int32NDArray1D,
    Int32NDArray2D,
)
from characterization.utils.common import XYZScale


class AgentData(BaseAgentData):
    """Agent data for autonomous driving scenarios.

    Attributes:
        agent_ids: All agent identifiers in the scenario, including the ego agent.
        agent_types: Type for each agent, e.g. vehicle or pedestrian.
        agent_trajectories: Shape (N, T, 10) where each point is
            (x, y, z, length, width, height, heading_rad, vx, vy, valid).
        agent_relevance: Shape (N,) relevance scores. If None, all agents are equally relevant.
    """

    agent_ids: list[NonNegativeInt]
    agent_types: list[AgentType]

    @computed_field
    @property
    def num_agents(self) -> int:
        """Number of agents in the scenario."""
        return len(self.agent_ids)


class ScenarioMetadata(BaseScenarioMetadata):
    """Scenario metadata for autonomous driving.

    Attributes:
        timestamps_seconds: Timestamps in seconds for each timestep.
        current_time_index: Index of the current timestep (required in AD).
        ego_vehicle_id: Identifier of the ego vehicle.
        ego_agent_index: Array index of the ego agent (vehicle).
        objects_of_interest: Identifiers of objects of interest.
        xyz_scale: Coordinate unit of the x/y/z position fields in agent trajectories.
        max_stationary_speed: Speed below which an agent is considered stationary (m/s).
        max_stationary_displacement: Max displacement for a stationary agent over the scenario (m).
        max_straight_lateral_displacement: Max lateral displacement for a straight trajectory (m).
        min_uturn_longitudinal_displacement: Min longitudinal displacement for a U-turn (m).
        max_straight_absolute_heading_diff: Max heading change for a straight trajectory (degrees).
        agent_to_agent_max_distance: Max distance to consider two agents interacting (m).
        agent_to_conflict_point_max_distance: Max distance from a conflict point to consider it relevant (m).
        agent_to_agent_distance_breach: Distance threshold for a close-call breach (m).
        heading_threshold: Max heading difference to consider agents co-directional (degrees).
        agent_max_deceleration: Maximum feasible deceleration (m/s²).
    """

    timestamps_seconds: list[float]
    current_time_index: int  # Required in AD; base allows None
    ego_vehicle_id: int
    ego_agent_index: int
    objects_of_interest: list[int]
    xyz_scale: XYZScale = XYZScale.M

    # Trajectory classification thresholds
    # Obtained from: https://github.com/vita-epfl/UniTraj/blob/main/unitraj/datasets/common_utils.py#L400
    max_stationary_speed: float = 2.0  # m/s
    max_stationary_displacement: float = 5.0  # m
    max_straight_lateral_displacement: float = 5.0  # m
    min_uturn_longitudinal_displacement: float = -5.0  # m
    max_straight_absolute_heading_diff: float = 30.0  # degrees

    # Interaction detection thresholds
    agent_to_agent_max_distance: float = 100.0  # m
    agent_to_conflict_point_max_distance: float = 10.0  # m
    agent_to_agent_distance_breach: float = 0.5  # m
    heading_threshold: float = 45.0  # degrees
    agent_max_deceleration: float = 15.0  # m/s²

    @computed_field
    @property
    def duration_s(self) -> float:
        """Total duration of the scenario in seconds."""
        if self.timestamps_seconds:
            return self.timestamps_seconds[-1] - self.timestamps_seconds[0]
        err_msg = "timestamps_seconds is empty, cannot compute duration."
        raise ValueError(err_msg)


class TracksToPredict(BaseModel):
    """Tracks selected for trajectory prediction.

    Attributes:
        track_index: Indices of the tracks to predict.
        difficulty: Difficulty level for each track.
        object_type: Agent type for each track.
    """

    track_index: list[NonNegativeInt]
    difficulty: list[NonNegativeInt]
    object_type: list[AgentType]

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}


class StaticMapData(BaseModel):
    """Static map data for an autonomous driving scenario.

    Attributes:
        map_polylines: Shape (P, 7) polyline points as (x, y, z, dx, dy, dz, type).
        lane_ids: Unique lane identifiers.
        lane_speed_limits_mph: Speed limit per lane in mph.
        lane_polyline_idxs: Shape (L, 2) start/end polyline indices per lane.
        road_line_ids: Road line identifiers.
        road_line_polyline_idxs: Shape (R, 2) polyline index ranges for road lines.
        road_edge_ids: Road edge identifiers.
        road_edge_polyline_idxs: Shape (E, 2) polyline index ranges for road edges.
        crosswalk_ids: Crosswalk identifiers.
        crosswalk_polyline_idxs: Shape (C, 2) polyline index ranges for crosswalks.
        speed_bump_ids: Speed bump identifiers.
        speed_bump_polyline_idxs: Shape (S, 2) polyline index ranges for speed bumps.
        stop_sign_ids: Stop sign identifiers.
        stop_sign_polyline_idxs: Shape (G, 2) polyline index ranges for stop signs.
        stop_sign_lane_ids: Lane IDs associated with each stop sign.
        map_conflict_points: Shape (C, 3) conflict point coordinates (x, y, z).
        agent_distances_to_conflict_points: Shape (N, C, T) distances from each agent to each conflict point.
        agent_closest_lanes: Shape (N, T, K, 6) K closest lanes per agent per timestep.
    """

    map_polylines: Float32NDArray2D | None = None
    lane_ids: Int32NDArray1D | None = None
    lane_speed_limits_mph: Float32NDArray1D | None = None
    lane_polyline_idxs: Int32NDArray2D | None = None
    road_line_ids: Int32NDArray1D | None = None
    road_line_polyline_idxs: Int32NDArray2D | None = None
    road_edge_ids: Int32NDArray1D | None = None
    road_edge_polyline_idxs: Int32NDArray2D | None = None
    crosswalk_ids: Int32NDArray1D | None = None
    crosswalk_polyline_idxs: Int32NDArray2D | None = None
    speed_bump_ids: Int32NDArray1D | None = None
    speed_bump_polyline_idxs: Int32NDArray2D | None = None
    stop_sign_ids: Int32NDArray1D | None = None
    stop_sign_polyline_idxs: Int32NDArray2D | None = None
    stop_sign_lane_ids: list[list[int]] | None = None
    map_conflict_points: Float32NDArray2D | None = None
    agent_distances_to_conflict_points: Float32NDArray3D | None = None
    agent_closest_lanes: Float32NDArray4D | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    @computed_field
    @property
    def num_polylines(self) -> int:
        """Number of polylines in the map."""
        return 0 if self.map_polylines is None else len(self.map_polylines)

    @computed_field
    @property
    def num_conflict_points(self) -> int:
        """Number of conflict points in the map."""
        return 0 if self.map_conflict_points is None else len(self.map_conflict_points)


class DynamicMapData(BaseModel):
    """Dynamic map data for an autonomous driving scenario (e.g. traffic light states).

    Attributes:
        stop_points: Dynamic stop points in the map.
        lane_ids: Lane identifiers associated with dynamic data.
        states: State information for each stop point.
    """

    stop_points: list[Any] | None = None
    lane_ids: list[Any] | None = None
    states: list[Any] | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    @computed_field
    @property
    def num_stop_points(self) -> int:
        """Number of dynamic stop points."""
        return 0 if self.stop_points is None else len(self.stop_points)


class Scenario(BaseScenario):
    """Autonomous driving scenario with agents, trajectories, and optional map data.

    Inherits ``to_pickle``, ``from_pickle``, and ``__setstate__`` from ``BaseScenario``.

    Attributes:
        metadata: Scenario metadata.
        agent_data: Agent trajectories and types.
        tracks_to_predict: Tracks selected for trajectory prediction (optional).
        static_map_data: Road network and map features (optional).
        dynamic_map_data: Traffic light states and dynamic elements (optional).
    """

    metadata: ScenarioMetadata
    agent_data: AgentData
    tracks_to_predict: TracksToPredict | None = None
    static_map_data: StaticMapData | None = None
    dynamic_map_data: DynamicMapData | None = None
    critical_probe: CriticalProbe | None = None

    def __setstate__(self, state: dict[str, object]) -> None:
        """Backfill fields missing from older pickles."""
        state.setdefault("critical_probe", None)
        self.__dict__.update(state)
