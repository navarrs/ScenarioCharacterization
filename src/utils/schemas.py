from typing import Any, List

import numpy as np
from pydantic import BaseModel, GetCoreSchemaHandler, NonNegativeInt, PositiveInt
from pydantic_core import core_schema


class BooleanNDArray3D:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        def validate(value: Any) -> np.ndarray:
            if not isinstance(value, np.ndarray):
                raise TypeError("Value must be a numpy ndarray")
            if value.ndim != 3:
                raise ValueError("Array must be 3D")
            if value.dtype != np.bool_:
                raise ValueError("Array must be of boolean type")
            return value

        return core_schema.no_info_plain_validator_function(validate)


class Float32NDArray3D:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        def validate(value: Any) -> np.ndarray:
            if not isinstance(value, np.ndarray):
                raise TypeError("Value must be a numpy ndarray")
            if value.ndim != 3:
                raise ValueError("Array must be 3D")
            if value.dtype != np.float32:
                raise ValueError("Array must be of float32 type")
            return value

        return core_schema.no_info_plain_validator_function(validate)


class Float32NDArray2D:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        def validate(value: Any) -> np.ndarray:
            if not isinstance(value, np.ndarray):
                raise TypeError("Value must be a numpy ndarray")
            if value.ndim != 2:
                raise ValueError("Array must be 2D")
            if value.dtype != np.float32:
                raise ValueError("Array must be of float32 type")
            return value

        return core_schema.no_info_plain_validator_function(validate)


class Float32NDArray1D:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        def validate(value: Any) -> np.ndarray:
            if not isinstance(value, np.ndarray):
                raise TypeError("Value must be a numpy ndarray")
            if value.ndim != 1:
                raise ValueError("Array must be 1D")
            if value.dtype != np.float32:
                raise ValueError("Array must be of float32 type")
            return value

        return core_schema.no_info_plain_validator_function(validate)


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
    timestamps: Float32NDArray1D
    map_conflict_points: Float32NDArray2D | None
    agent_distances_to_conflict_points: Float32NDArray3D | None
