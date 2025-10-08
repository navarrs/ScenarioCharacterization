from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel, NonNegativeInt, computed_field

from characterization.utils.ad_types import AgentType
from characterization.utils.common import (
    Float32NDArray1D,
    Float32NDArray2D,
    Float32NDArray3D,
    Float32NDArray4D,
    Int32NDArray1D,
    Int32NDArray2D,
)

DType = TypeVar("DType", bound=np.generic)


class AgentData(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the data for agents in a scenario, including their identifiers, types, and trajectories.

    Attributes:
        agent_ids (list[NonNegativeInt]): list of all agent identifiers in the scenario, including the ego agent.
        agent_types (list[str]): list of types for each agent, e.g., "vehicle", "pedestrian", etc.
        agent_trajectories (Float32NDArray3D): 3D array of shape (N, T, D=10) where N is the number of agents, T is the
            number of timesteps and D is the number of features per trajectory point, organized as follows:
                idx 0 to 2: the agent's (x, y, z) center coordinates.
                idx 3 to 5: the agent's length, width and height in meters.
                idx 6: the agent's angle (heading) of the forward direction in radians
                idx 7 to 8: the agent's (x, y) velocity in meters/second
                idx 9: a flag indicating if the information is valid
            NOTE: For convenient masking see 'AgentTrajectoryMasker' in utils/common.py
        agent_relevance (Float32NDArray1D | None): Optional 1D array of shape (N,) representing relevance scores for
            each agent. Higher values indicate greater relevance, while NaN or negative values indicate
            irrelevance. If None, all agents are considered equally relevant.
    """

    agent_ids: list[NonNegativeInt]
    agent_types: list[AgentType]
    agent_trajectories: Float32NDArray3D
    agent_relevance: Float32NDArray1D | None = None  # Optional relevance scores for agents

    # To allow numpy and other arbitrary types in the model
    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    @computed_field
    @property
    def num_agents(self) -> int:
        """Returns the number of agents in the scenario."""
        return len(self.agent_ids)


class ScenarioMetadata(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates metadata for a scenario, including its ID, type, and timing information.

    Attributes:
        scenario_id (str): Unique identifier for the scenario.
        timestamps_seconds (list[float]): List of timestamps in seconds for each timestep in the scenario.
        frequency_hz (float): Frequency of the scenario data in Hertz (Hz).
        current_time_index (int): Current index in the timestamps list.
        ego_vehicle_id (int): Unique identifier for the ego vehicle in the scenario.
        ego_vehicle_index (int): Index of the ego vehicle in the scenario.
        objects_of_interest (list): List of objects of interest in the scenario.
        track_length (int): Length of the track in the scenario.
        dataset (str): Name of the dataset from which the scenario is derived.
        max_stationary_speed (float | None): Speed threshold below which an agent is considered stationary.
        max_stationary_displacement (float | None): Maximum displacement threshold for an agent to be considered
            stationary over the scenario duration.
        max_straight_lateral_displacement (float | None): Maximum lateral displacement threshold for an agent to be
            considered moving straight.
        min_uturn_longitudinal_displacement (float | None): Minimum longitudinal displacement threshold for an agent to
            be considered making a U-turn.
        max_straight_absolute_heading_diff (float | None): Maximum absolute heading difference threshold for an agent to
            be considered moving straight.
        agent_to_agent_max_distance (float | None): Maximum distance between agents to consider them for interaction.
        agent_to_conflict_point_max_distance (float | None): Maximum distance from an agent to a conflict point to
            consider it relevant.
        agent_to_agent_distance_breach (float | None): Distance threshold for considering an agent's distance to another
            agent as a breach / close-call.
        heading_threshold (float | None): Threshold for the heading difference between agents.
    """

    scenario_id: str
    timestamps_seconds: list[float]
    frequency_hz: float
    current_time_index: int
    ego_vehicle_id: int
    ego_vehicle_index: int
    objects_of_interest: list[int]
    track_length: int
    dataset: str

    # Thresholds.
    # Obtained from: https://github.com/vita-epfl/UniTraj/blob/main/unitraj/datasets/common_utils.py#L400
    max_stationary_speed: float = 2.0  # m/s
    max_stationary_displacement: float = 5.0  # meters
    max_straight_lateral_displacement: float = 5.0  # meters
    min_uturn_longitudinal_displacement: float = -5.0  # meters
    max_straight_absolute_heading_diff: float = 30.0  # degrees

    # Optional thresholds for scenario characterization
    agent_to_agent_max_distance: float = 50.0  # meters
    agent_to_conflict_point_max_distance: float = 2.0  # meters
    agent_to_agent_distance_breach: float = 1.0  # meters
    heading_threshold: float = 45.0  # degrees

    # To allow numpy and other arbitrary types in the model
    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}


class TracksToPredict(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the tracks to predict in a scenario, including the ego vehicle and other agents.

    Attributes:
        track_index (list[NonNegativeInt]): List of indices for the tracks.
        difficulty (list[NonNegativeInt]): List of difficulty levels for the tracks.
        object_type (list[AgentType]): List of types for each track, e.g., "vehicle", "pedestrian", etc.
    """

    track_index: list[NonNegativeInt]
    difficulty: list[NonNegativeInt]
    object_type: list[AgentType]

    # To allow numpy and other arbitrary types in the model
    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}


class StaticMapData(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates static map data for a scenario, including polylines and conflict points.

    Attributes:
        map_polylines (Float32NDArray2D | None): 2D array of shape (P, 7) representing the polylines in the map, where
            P is the number of polyline points. Each polyline point is represented by 7 features:
                idx 0 to 2: the (x, y, z) coordinates of the polyline point.
                idx 3 to 5: the polyline direction vector (dx, dy, dz).
                idx 6: the polyline type (e.g., lane, road line, road edge, crosswalk, speed bump, stop sign).
        lane_ids (Int32NDArray1D | None): 1D array of unique lane identifiers.
        lane_speed_limits_mph (Float32NDArray1D | None): 1D array of speed limits for each lane in miles per hour (mph).
        lane_polyline_idxs (Int32NDArray2D | None): 2D array of shape (L, 2) representing the start and end polyline
            indices in map_polylines for each lane. L is the number of lanes.
        road_line_ids (Int32NDArray1D | None): 1D array of unique road line identifiers.
        road_line_polyline_idxs (Int32NDArray2D | None): 2D array of shape (R, 2) representing the start and end
            polyline indices in map_polylines for each road line. R is the number of road lines.
        road_edge_ids (Int32NDArray1D | None): 1D array of unique road edge identifiers.
        road_edge_polyline_idxs (Int32NDArray2D | None): 2D array of shape (E, 2) representing the start and end
            polyline indices in map_polylines for each road edge. E is the number of road edges.
        crosswalk_ids (Int32NDArray1D | None): 1D array of unique crosswalk identifiers.
        crosswalk_polyline_idxs (Int32NDArray2D | None): 2D array of shape (C, 2) representing the start and end
            polyline indices in map_polylines for each crosswalk. C is the number of crosswalks.
        speed_bump_ids (Int32NDArray1D | None): 1D array of unique speed bump identifiers.
        speed_bump_polyline_idxs (Int32NDArray2D | None): 2D array of shape (S, 2) representing the start and end
            polyline indices in map_polylines for each speed bump. S is the number of speed bumps.
        stop_sign_ids (Int32NDArray1D | None): 1D array of unique stop sign identifiers.
        stop_sign_polyline_idxs (Int32NDArray2D | None): 2D array of shape (G, 2) representing the start and end
            polyline indices in map_polylines for each stop sign. G is the number of stop signs.
        stop_sign_lane_ids (list[list[int]] | None): List of lists, where each inner list contains lane IDs associated
            with a stop sign.
        map_conflict_points (Float32NDArray2D | None): 2D array of shape (C, 3) representing conflict regions in the
            map, where C is the number of conflict points. Each conflict point is represented as (x, y, z) coordinates.
        agent_distances_to_conflict_points (Float32NDArray3D | None): 3D array of shape (N, C, T) representing the
            distances from each agent to each conflict point at each timestep, where N is the number of agents, C is the
            number of conflict points, and T is the number of timesteps. Distances are in meters.
        agent_closest_lanes (Float32NDArray4D | None): 4D array of shape (N, T, K, 6) representing the K closest lanes
            to each agent at each timestep, where N is the number of agents, T is the number of timesteps, and K is the
            number of closest lanes.
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

    # Optional information that can be derived from existing map information
    map_conflict_points: Float32NDArray2D | None = None
    agent_distances_to_conflict_points: Float32NDArray3D | None = None
    agent_closest_lanes: Float32NDArray4D | None = None

    # To allow numpy and other arbitrary types in the model
    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    @computed_field
    @property
    def num_polylines(self) -> int:
        """Returns the number of polylines in the map."""
        return 0 if self.map_polylines is None else len(self.map_polylines)

    @computed_field
    @property
    def num_conflict_points(self) -> int:
        """Returns the number of conflict points in the map."""
        return 0 if self.map_conflict_points is None else len(self.map_conflict_points)


class DynamicMapData(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates dynamic map data for a scenario, including dynamic stop points.

    TODO: combine dynamic and static map data into a single map data class.

    Attributes:
        stop_points (list[Any] | None): List of dynamic stop points in the map.
        lane_ids (list[Any] | None): List of lane identifiers associated with the dynamic map data.
        states (list[Any] | None): Placeholder for state information, can be more specific if needed.
    """

    stop_points: list[Any] | None = None
    lane_ids: list[Any] | None = None
    states: list[Any] | None = None  # Placeholder for state information, can be more specific if needed

    # To allow numpy and other arbitrary types in the model
    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    @computed_field
    @property
    def num_stop_points(self) -> int:
        """Returns the number of dynamic stop points in the map."""
        return 0 if self.stop_points is None else len(self.stop_points)


class Scenario(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Represents a scenario containing information about agents, their trajectories, and the environment.

    This class is used to encapsulate all relevant data for a scenario, including agent states, map information,
    and scenario metadata. It is designed to be used in the context of autonomous driving or similar.

    Attributes:
        metadata (ScenarioMetadata): Metadata for the scenario, including IDs, timestamps, and thresholds.
        agent_data (AgentData): Data for agents in the scenario, including their trajectories, positions, and types.
        tracks_to_predict (TracksToPredict | None): Optional data for tracks to predict in the scenario.
        static_map_data (StaticMapData | None): Optional static map data for the scenario.
        dynamic_map_data (DynamicMapData | None): Optional dynamic map data for the scenario.
    """

    metadata: ScenarioMetadata
    agent_data: AgentData
    tracks_to_predict: TracksToPredict | None = None
    static_map_data: StaticMapData | None = None
    dynamic_map_data: DynamicMapData | None = None

    # To allow numpy and other arbitrary types in the model
    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}
