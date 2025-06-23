from typing import Annotated, Any, Callable, List, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, BeforeValidator, NonNegativeInt, PositiveInt

from characterization.utils.common import InteractionStatus

DType = TypeVar("DType", bound=np.generic)


# Validator factory
def validate_array(expected_dtype: Any, expected_ndim: int) -> Callable[[Any], NDArray]:
    def _validator(v: Any) -> NDArray:
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
Float32NDArray3D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 3))]
Float32NDArray2D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 2))]
Float32NDArray1D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 1))]
Int32NDArray1D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 1))]
Int32NDArray2D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 2))]


class Scenario(BaseModel):
    """Represents a scenario containing information about agents, their trajectories, and the environment.
    This class is used to encapsulate all relevant data for a scenario, including agent states, map information,
    and scenario metadata. It is designed to be used in the context of autonomous driving or similar.

    Attributes:
        scenario_id (str): Unique identifier for the scenario.
        total_timesteps (PositiveInt): Total number of timesteps in the scenario.
        last_observed_timestep (PositiveInt): Intermediate timestep for splitting the scenario into history and future.
        timestamps (Float32NDArray1D): Timestamps for each timestep in the scenario.
        num_agents (PositiveInt): Total number of agents in the scenario, including those with padded trajectories.
        ego_index (NonNegativeInt): Index of the ego agent (e.g., self-driving car) in the scenario.
        ego_id (PositiveInt): Unique identifier for the ego agent.
        agent_ids (List[NonNegativeInt]): List of all agent identifiers in the scenario, including the ego agent.
        agent_types (List[str]): List of types for each agent, e.g., "vehicle", "pedestrian", etc.
        agent_positions (Float32NDArray3D): 3D array of shape(num_agents, total_timesteps, 3) representing each agent's
            (x, y, z) positions at each timestep.
        agent_dimensions (Float32NDArray3D): 3D array of shape(num_agents, total_timesteps, 3) representing each agent's
            dimensions (length, width, height) at each timestep.
        agent_velocities (Float32NDArray3D): 3D array of shape(num_agents, total_timesteps, 2) representing each agent's
          (Vx, Vy) velocities at each timestep.
        agent_headings (Float32NDArray3D): 3D array of shape(num_agents, total_timesteps, 1) representing each agent's
            headings (in radians) at each timestep.
        agent_valid (BooleanNDArray3D): 3D boolean array of shape(num_agents, total_timesteps, 1) indicating whether an
            agent's information at a given timestep is valid.
        agent_relevance (Float32NDArray1D): 1D array of shape(num_agents) indicating the relevance [0, 1] of each agent
            in the scenario.
        num_conflict_points (NonNegativeInt): Number of conflict points in the map, if available.
        map_conflict_points (Float32NDArray2D | None): 2D array of shape(num_conflict_points, 3) representing the
            (x, y, z) positions for each conflict point in the map.
        agent_distances_to_conflict_points (Float32NDArray3D | None): 3D array of shape(num_conflict_points, num_agents,
            1)representing the closest distances from agents to conflict points, if available.
        num_polylines (NonNegativeInt): Number of polylines in the map, if available.
        map_polylines (Float32NDArray2D | None): 2D array of shape(num_polylines, 7) representing the polylines in the
            map, if available.
        lane_ids (Int32NDArray1D | None): 1D array of shape(num_polylines) representing the lane IDs for each polyline.
        lane_speed_limits_mph (Float32NDArray1D | None): 1D array of shape(num_polylines) representing the speed limits
            in miles per hour for each lane.
        lane_polyline_idxs (Int32NDArray2D | None): 2D array of shape(num_polylines, 2) representing the start and end
            indices of each lane in the polylines.
        road_line_ids (Int32NDArray1D | None): 1D array of shape(num_polylines) representing the road line IDs for each
            polyline.
        road_line_polyline_idxs (Int32NDArray2D | None): 2D array of shape(num_polylines, 2) representing the start and
            end indices of each road line in the polylines.
        road_edge_ids (Int32NDArray1D | None): 1D array of shape(num_polylines) representing the road edge IDs for each
            polyline.
        road_edge_polyline_idxs (Int32NDArray2D | None): 2D array of shape(num_polylines, 2) representing the start and
            end indices of each road edge in the polylines.
        crosswalk_ids (Int32NDArray1D | None): 1D array of shape(num_polylines) representing the crosswalk IDs for each
            polyline.
        crosswalk_polyline_idxs (Int32NDArray2D | None): 2D array of shape(num_polylines, 2) representing the start and
            end indices of each crosswalk in the polylines.
        speed_bump_ids (Int32NDArray1D | None): 1D array of shape(num_polylines) representing the speed bump IDs for
            each polyline.
        speed_bump_polyline_idxs (Int32NDArray2D | None): 2D array of shape(num_polylines, 2) representing the start and
            end indices of each speed bump in the polylines.
        stop_sign_ids (Int32NDArray1D | None): 1D array of shape(num_polylines) representing the stop sign IDs for each
            polyline.
        stop_sign_polyline_idxs (Int32NDArray2D | None): 2D array of shape(num_polylines, 2) representing the start and
            end indices of each stop sign in the polylines.
        stop_sign_lane_ids (list[list[int]] | None): List of lists representing the lane IDs associated with each stop
            sign, if available.
        num_dynamic_stop_points (NonNegativeInt): Number of dynamic stop points in the map, if available.
        dynamic_stop_points (Float32NDArray2D | None): 2D array of shape(num_dynamic_stop_points, 3) representing the
            (x, y, z) positions for each dynamic stop point in the map.
        dynamic_stop_points_lane_ids (Int32NDArray1D | None): 1D array of shape(num_dynamic_stop_points) representing
            the lane IDs for each dynamic stop point.
        stationary_speed (float): Speed threshold below which an agent is considered stationary.
        agent_to_agent_max_distance (float): Maximum distance between agents to consider them for interaction.
        agent_to_conflict_point_max_distance (float): Maximum distance from an agent to a conflict point to consider it
            relevant.
        agent_to_agent_distance_breach (float): Distance threshold for considering an agent's distance to another agent
            as a breach / close-call.
    """

    # Scenario Information
    scenario_id: str
    scenario_type: str
    total_timesteps: PositiveInt
    last_observed_timestep: PositiveInt
    timestamps: Float32NDArray1D
    last_timestep_to_consider: PositiveInt

    # Agent Information
    num_agents: PositiveInt
    ego_index: NonNegativeInt
    ego_id: PositiveInt
    agent_ids: List[NonNegativeInt]
    agent_types: List[str]

    agent_positions: Float32NDArray3D
    agent_velocities: Float32NDArray3D
    agent_lengths: Float32NDArray2D
    agent_widths: Float32NDArray2D
    agent_heights: Float32NDArray2D
    agent_headings: Float32NDArray2D
    agent_valid: BooleanNDArray2D
    agent_relevance: Float32NDArray1D

    # Map Information
    num_conflict_points: NonNegativeInt = 0
    map_conflict_points: Float32NDArray2D | None
    agent_distances_to_conflict_points: Float32NDArray3D | None
    num_polylines: NonNegativeInt = 0
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
    num_dynamic_stop_points: NonNegativeInt = 0
    dynamic_stop_points: Float32NDArray2D | None = None
    dynamic_stop_points_lane_ids: Int32NDArray1D | None = None

    # Thresholds
    stationary_speed: float
    agent_to_agent_max_distance: float
    agent_to_conflict_point_max_distance: float
    agent_to_agent_distance_breach: float
    heading_threshold: float

    # To allow numpy and other arbitrary types in the model
    model_config = {"arbitrary_types_allowed": True}


class ScenarioFeatures(BaseModel):
    """Represents the features extracted from a scenario, including individual agent features and interaction features.
    This class is used to encapsulate all relevant features for a scenario, including agent states, interaction metrics,
    and other characteristics that can be used for analysis or modeling.

    Attributes:
        scenario_id (str): Unique identifier for the scenario.
        num_agents (PositiveInt): Total number of agents in the scenario.
        valid_idxs (Int32NDArray1D | None): Indices of valid agents in the scenario.
        agent_types (List[str] | None): List of types for each agent, e.g., "vehicle", "pedestrian", etc.
        speed (Float32NDArray1D | None): Speed of each agent at each timestep.
        speed_limit_diff (Float32NDArray1D | None): Difference between agent speed and speed limit at each timestep.
        acceleration (Float32NDArray1D | None): Acceleration of each agent at each timestep.
        deceleration (Float32NDArray1D | None): Deceleration of each agent at each timestep.
        jerk (Float32NDArray1D | None): Jerk (rate of change of acceleration) of each agent at each timestep.
        waiting_period (Float32NDArray1D | None): Waiting period of each agent at each timestep.
        waiting_interval (Float32NDArray1D | None): Waiting interval of each agent at each timestep.
        waiting_distance (Float32NDArray1D | None): Waiting distance of each agent at each timestep.
        agent_to_agent_closest_dists (Float32NDArray2D | None):
            2D array of shape(num_agents, num_agents) representing the closest distances between agents.
        separation (Float32NDArray1D | None): Separation distance between agents at each timestep.
        intersection (Float32NDArray1D | None): Intersection distance between agents at each timestep.
        collision (Float32NDArray1D | None): Collision distance between agents at each timestep.
        mttcp (Float32NDArray1D | None): Minimum time to conflict point (mTTCP) for each agent at each timestep.
        interaction_status (List[InteractionStatus] | None):
            List of interaction statuses for each agent pair in the scenario.
        interaction_agent_indices (List[tuple[int, int]] | None):
            List of tuples representing the indices of interacting agents in the scenario.
        interaction_agent_types (List[tuple[str, str]] | None):
            List of tuples representing the types of interacting agents in the scenario.
    """

    scenario_id: str
    num_agents: PositiveInt

    # Individual Features
    valid_idxs: Int32NDArray1D | None = None
    agent_types: List[str] | None = None
    speed: Float32NDArray1D | None = None
    speed_limit_diff: Float32NDArray1D | None = None
    acceleration: Float32NDArray1D | None = None
    deceleration: Float32NDArray1D | None = None
    jerk: Float32NDArray1D | None = None
    waiting_period: Float32NDArray1D | None = None
    waiting_interval: Float32NDArray1D | None = None
    waiting_distance: Float32NDArray1D | None = None

    # Interaction Features
    agent_to_agent_closest_dists: Float32NDArray2D | None = None
    separation: Float32NDArray1D | None = None
    intersection: Float32NDArray1D | None = None
    collision: Float32NDArray1D | None = None
    mttcp: Float32NDArray1D | None = None
    thw: Float32NDArray1D | None = None
    ttc: Float32NDArray1D | None = None
    drac: Float32NDArray1D | None = None
    # leader_follower: Float32NDArray1D | None = None
    # valid_headings: Float32NDArray1D | None = None
    interaction_status: List[InteractionStatus] | None = None
    interaction_agent_indices: List[tuple[int, int]] | None = None
    interaction_agent_types: List[tuple[str, str]] | None = None

    model_config = {"arbitrary_types_allowed": True}


class ScenarioScores(BaseModel):
    """Represents the scores for a scenario, including individual agent scores, interaction scores, and combined.
    This class is used to encapsulate the results of scoring a scenario based on various criteria, such as safety and
    interaction quality.

    Attributes:
        scenario_id (str): Unique identifier for the scenario.
        num_agents (PositiveInt): Total number of agents in the scenario.
        individual_agent_scores (Float32NDArray1D | None): Individual scores for each agent in the scenario.
        individual_scene_score (float | None): Overall score for the scene based on individual agent scores.
        interaction_agent_scores (Float32NDArray1D | None): Interaction scores for each agent in the scenario.
        interaction_scene_score (float | None): Overall score for the scene based on interaction scores.
        safeshift_agent_scores (Float32NDArray1D | None): Combined scores for each agent in the scenario,
            incorporating both individual and interaction scores.
        safeshift_scene_score (float | None): Overall score for the scene based on combined scores.
    """

    scenario_id: str
    num_agents: PositiveInt

    # Individual Scores
    individual_agent_scores: Float32NDArray1D | None = None
    individual_scene_score: float | None = None

    # Interaction Scores
    interaction_agent_scores: Float32NDArray1D | None = None
    interaction_scene_score: float | None = None

    # Combined Scores
    safeshift_agent_scores: Float32NDArray1D | None = None
    safeshift_scene_score: float | None = None

    model_config = {"arbitrary_types_allowed": True}

    def __getitem__(self, key: str) -> Any:
        """
        Get the value of a key in the ScenarioScores object.

        Args:
            key (str): The key to retrieve.

        Returns:
            Any: The value associated with the key.
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found in ScenarioScores.")
