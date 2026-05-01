from collections.abc import Callable
from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BeforeValidator

from characterization.domains.aviation.scenario_types import AgentType


# Validator factory
def validate_array(
    expected_dtype: Any,  # noqa: ANN401
    expected_ndim: int,
) -> Callable[[Any], NDArray]:  # pyright: ignore[reportMissingTypeArgument]
    """Factory function to create a validator for numpy arrays with specific dtype and ndim."""

    def _validator(
        v: Any,  # noqa: ANN401
    ) -> NDArray:  # pyright: ignore[reportMissingTypeArgument]
        if not isinstance(v, np.ndarray):
            error_message = f"Expected a numpy.ndarray, got {type(v)}"
            raise TypeError(error_message)
        if v.dtype != expected_dtype:
            error_message = f"Expected dtype {expected_dtype}, got {v.dtype}"
            raise TypeError(error_message)
        if v.ndim != expected_ndim:
            error_message = f"Expected {expected_ndim}D array, got {v.ndim}D"
            raise ValueError(error_message)
        return v

    return _validator


# Reusable types
BooleanNDArray1D = Annotated[NDArray[np.bool_], BeforeValidator(validate_array(np.bool_, 1))]
BooleanNDArray2D = Annotated[NDArray[np.bool_], BeforeValidator(validate_array(np.bool_, 2))]
BooleanNDArray3D = Annotated[NDArray[np.bool_], BeforeValidator(validate_array(np.bool_, 3))]
Float64NDArray3D = Annotated[NDArray[np.float64], BeforeValidator(validate_array(np.float64, 3))]
Float32NDArray4D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 4))]
Float32NDArray3D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 3))]
Float32NDArray2D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 2))]
Float32NDArray1D = Annotated[NDArray[np.float32], BeforeValidator(validate_array(np.float32, 1))]
Int32NDArray1D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 1))]
Int32NDArray2D = Annotated[NDArray[np.int32], BeforeValidator(validate_array(np.int32, 2))]
Int64NDArray1D = Annotated[NDArray[np.int64], BeforeValidator(validate_array(np.int64, 1))]
Int64NDArray2D = Annotated[NDArray[np.int64], BeforeValidator(validate_array(np.int64, 2))]
AgentTypeNDArray1D = Annotated[NDArray[AgentType], BeforeValidator(validate_array(AgentType, 1))]
ObjectTypeNDArray1D = Annotated[NDArray[object], BeforeValidator(validate_array(np.object_, 1))]
