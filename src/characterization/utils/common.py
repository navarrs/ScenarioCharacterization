from enum import Enum, StrEnum
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# General-purpose Enums
# ---------------------------------------------------------------------------


class DataSplits(Enum):
    """Enumeration of data splits."""

    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


class XYZScale(Enum):
    """Enumeration of XYZ coordinate scales."""

    KM = "km"
    HM = "hm"
    M = "m"


class SpeedUnits(Enum):
    """Enumeration of speed units."""

    KNOTS = "knots"
    MS = "ms"
    MPH = "mph"


class AccelerationUnits(Enum):
    """Enumeration of acceleration units."""

    MS2 = "ms2"


class PositionUnits(Enum):
    """Enumeration of position units for feature output."""

    M = "m"
    HM = "hm"
    KM = "km"


class FeatureType(Enum):
    """Enumeration for feature types."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class InteractionStatus(Enum):
    """Enumeration for interaction status."""

    UNKNOWN = -1
    COMPUTED_OK = 0
    PARTIAL_INVALID_HEADING = 1
    MASK_NOT_VALID = 2
    DISTANCE_TOO_FAR = 3
    STATIONARY = 4


class ReturnCriterion(StrEnum):
    """How to reduce a per-timestep feature array to a scalar, or return it unchanged."""

    CRITICAL = "critical"
    AVERAGE = "average"
    ALL = "all"


class TrajectoryType(Enum):
    """Trajectory Types for WOMD."""

    TYPE_UNSET = -1
    TYPE_STATIONARY = 0
    TYPE_STRAIGHT = 1
    TYPE_STRAIGHT_RIGHT = 2
    TYPE_STRAIGHT_LEFT = 3
    TYPE_RIGHT_U_TURN = 4
    TYPE_RIGHT_TURN = 5
    TYPE_LEFT_U_TURN = 6
    TYPE_LEFT_TURN = 7


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------


def return_by_criterion(
    values: NDArray[np.float32],
    criterion: ReturnCriterion,
    *,
    critical_is_min: bool = False,
) -> float | NDArray[np.float32]:
    """Reduce a 1-D array to a scalar, or return it unchanged, depending on the criterion.

    Args:
        values: Array of values to reduce or return.
        criterion: Aggregation strategy.
        critical_is_min: When ``True``, ``CRITICAL`` returns the minimum instead of the maximum.
            Use this for safety-relevant time features (TTC, THW) where smaller means more dangerous.
            Ignored when ``criterion`` is ``ALL``.

    Returns:
        The original array when ``criterion`` is ``ALL``; a scalar otherwise.
        Returns ``float("nan")`` for empty arrays with scalar criteria.
    """
    if len(values) == 0:
        return float("nan")

    match criterion:
        case ReturnCriterion.ALL:
            return values
        case ReturnCriterion.CRITICAL:
            return float(np.nanmin(values)) if critical_is_min else float(np.nanmax(values))
        case ReturnCriterion.AVERAGE:
            return float(np.nanmean(values))


def categorize_from_thresholds(value: float, threshold_values: list[float]) -> int:
    """Categorizes a value based on provided ranges.

    Args:
        value (float): The value to categorize.
        threshold_values (list[float]): A list of threshold values defining the ranges.

    Returns:
        int: The category index (1, 2, ..., n+1) based on the ranges.
    """
    num_thresholds = len(threshold_values)
    assert num_thresholds >= 1, "At least one range must be provided."

    # If there is only one category, return 1 or 2 based on the value
    if num_thresholds < 2:  # noqa: PLR2004
        return 1 if value <= threshold_values[0] else 2

    # If value is below the lowest range, return 1
    if value <= threshold_values[0]:
        return 1

    # Categorize based on ranges, starting from category 2
    for category, (lower_bound, upper_bound) in enumerate(pairwise(threshold_values)):
        if lower_bound < value <= upper_bound:
            return category + 2

    # If value is above the highest range
    return num_thresholds + 1


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class ValueClipper(BaseModel):
    """Bounds for clipping a value."""

    min: float = 0.0
    max: float = float("inf")

    def clip(self, value: float) -> float:
        """Clip the input value to the configured bounds."""
        return float(np.clip(value, self.min, self.max))

    def clip_array(self, values: NDArray[np.float32]) -> NDArray[np.float32]:
        """Clip the input array to the configured bounds."""
        return np.clip(values, self.min, self.max).astype(np.float32)


SUPPORTED_SCENARIO_TYPES = ["gt", "ho"]
SUPPORTED_CRITERIA = ["critical", "average"]
SUPPORTED_SCORERS = ["individual", "interaction", "safeshift"]
MAX_DECELERATION = 10.0  # m/s^2
STATIONARY_SPEED_THRESHOLD = 0.5  # m/s
