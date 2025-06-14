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
BooleanNDArray3D = Annotated[NDArray[np.bool_], BeforeValidator(validate_array(np.bool_, 3))]
Float32NDArray3D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 3))]
Float32NDArray2D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 2))]
Float32NDArray1D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 1))]
Int32NDArray1D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 1))]
Int32NDArray2D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 2))]


class Scenario(BaseModel):
    # Scenario Information
    scenario_id: str
    last_observed_timestep: PositiveInt
    total_timesteps: PositiveInt
    timestamps: Float32NDArray1D

    # Agent Information
    num_agents: PositiveInt

    ego_index: NonNegativeInt
    ego_id: PositiveInt
    agent_ids: List[NonNegativeInt]
    agent_types: List[str]
    agent_valid: BooleanNDArray3D
    agent_positions: Float32NDArray3D
    agent_dimensions: Float32NDArray3D
    agent_velocities: Float32NDArray3D
    agent_headings: Float32NDArray3D
    agent_relevance: Float32NDArray1D

    # Map Information
    static_map_info: dict[str, Any]
    dynamic_map_info: dict[str, Any]
    # map_polylines: Float32NDArray2D | None = None
    # polyline_idxs_lane: Int32NDArray2D | None = None
    # polyline_idxs_road_line: Int32NDArray2D | None = None
    # polyline_idxs_road_edge: Int32NDArray2D | None = None
    # polyline_idxs_crosswalk: Int32NDArray2D | None = None
    # polyline_idxs_speed_bump: Int32NDArray2D | None = None
    # polyline_idxs_stop_sign: Int32NDArray2D | None = None
    # map_stop_points: Float32NDArray3D | None = None

    map_conflict_points: Float32NDArray2D | None
    agent_distances_to_conflict_points: Float32NDArray3D | None

    # Thresholds
    stationary_speed: float
    agent_to_agent_max_distance: float
    agent_to_conflict_point_max_distance: float
    agent_to_agent_distance_breach: float

    model_config = {"arbitrary_types_allowed": True}


class ScenarioFeatures(BaseModel):
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
    interaction_status: List[InteractionStatus] | None = None
    interaction_agent_indices: List[tuple[int, int]] | None = None
    interaction_agent_types: List[tuple[str, str]] | None = None

    model_config = {"arbitrary_types_allowed": True}


class ScenarioScores(BaseModel):
    scenario_id: str
    num_agents: PositiveInt

    # Individual Scores
    individual_agent_scores: Float32NDArray1D | None = None
    individual_scene_score: float | None = None

    # Interaction Scores
    interaction_agent_scores: Float32NDArray1D | None = None
    interaction_scene_score: float | None = None

    # Combined Scores
    combined_agent_scores: Float32NDArray1D | None = None
    combined_scene_score: float | None = None

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
