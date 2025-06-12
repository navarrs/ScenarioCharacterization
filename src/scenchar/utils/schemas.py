from typing import Annotated, Any, Callable, List, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, BeforeValidator, NonNegativeInt, PositiveInt

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


class Scenario(BaseModel):
    num_agents: PositiveInt
    scenario_id: str
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
    last_observed_timestep: PositiveInt
    total_timesteps: PositiveInt
    stationary_speed: float
    agent_to_agent_max_distance: float
    agent_to_conflict_point_max_distance: float
    agent_to_agent_distance_breach: float
    timestamps: Float32NDArray1D
    map_conflict_points: Float32NDArray2D | None
    agent_distances_to_conflict_points: Float32NDArray3D | None

    model_config = {"arbitrary_types_allowed": True}
