from enum import Enum
from itertools import pairwise

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


class ReturnCriterion(Enum):
    """Enumeration for return criteria."""

    CRITICAL = 0
    AVERAGE = 1
    UNSET = -1


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
# Shared utility function
# ---------------------------------------------------------------------------


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

SUPPORTED_SCENARIO_TYPES = ["gt", "ho"]
SUPPORTED_CRITERIA = ["critical", "average"]
SUPPORTED_SCORERS = ["individual", "interaction", "safeshift"]
MAX_DECELERATION = 10.0  # m/s^2
STATIONARY_SPEED_THRESHOLD = 0.5  # m/s
