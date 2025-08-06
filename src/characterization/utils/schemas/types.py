from collections.abc import Callable
from typing import Annotated, Any
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from pydantic import BeforeValidator

# Validator factory
def validate_array(
    expected_dtype: Any,
    expected_ndim: int,
) -> Callable[[Any], NDArray]:  # pyright: ignore[reportMissingTypeArgument]
    def _validator(v: Any) -> NDArray:  # pyright: ignore[reportMissingTypeArgument]
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
Float64NDArray3D = Annotated[NDArray[np.float64], BeforeValidator(validate_array(np.float64, 3))]
Float32NDArray3D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 3))]
Float32NDArray2D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 2))]
Float32NDArray1D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 1))]
Int32NDArray1D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 1))]
Int32NDArray2D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 2))]
Int64NDArray2D = Annotated[NDArray[np.int64], BeforeValidator(validate_array(np.int64, 2))]


class AgentType(Enum):
    TYPE_UNSET = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_CYCLIST = 3
    TYPE_OTHER = 4
