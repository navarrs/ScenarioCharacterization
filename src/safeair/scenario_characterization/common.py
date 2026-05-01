"""Shared enums and utilities for scenario characterization."""

from enum import Enum, StrEnum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from characterization.domains.aviation.scenario_types import STRING_TO_AGENT_TYPE, AgentType
from safeair.schemas import Scenario


class ProbeValidity(Enum):
    """Validity of a counterfactual probe.

    Attributes:
        VALID: The probe produced a valid scenario with a meaningful score delta.
        INVALID: The probe produced an invalid scenario (e.g. with NaNs or empty trajectories) or no meaningful
            score delta.
    """

    VALID = 0
    INVALID = -1


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


class ReturnCriterion(StrEnum):
    """How to reduce a per-timestep feature array to a scalar, or return it unchanged.

    Attributes:
        CRITICAL: Return the critical (most extreme) feature value.
        AVERAGE: Return the mean value.
        ALL: Return the full per-timestep array without reduction.
    """

    CRITICAL = "critical"
    AVERAGE = "average"
    ALL = "all"


class TrajectoryType(StrEnum):
    """Classification of an agent's trajectory shape.

    Attributes:
        STATIONARY: The agent did not move meaningfully.
        TAKEOFF: The agent is taking off — either climbing or accelerating on the ground before liftoff.
        LANDING: The agent lost significant altitude but had not reached the ground by the end of the trajectory.
        LANDED: The agent lost significant altitude and reached the ground by the end of the trajectory.
        STRAIGHT: The agent moved in a roughly straight line.
        TURNING: The agent turned.
        U_TURN: The agent reversed direction.
    """

    STATIONARY = "STATIONARY"
    TAKEOFF = "TAKEOFF"
    LANDING = "LANDING"
    LANDED = "LANDED"
    STRAIGHT = "STRAIGHT"
    TURNING = "TURNING"
    U_TURN = "U_TURN"


class AgentPairType(StrEnum):
    """Type of an agent pair, derived from the types of the two agents.

    Attributes:
        AIRCRAFT_AIRCRAFT: Both agents are aircraft.
        AIRCRAFT_VEHICLE: One aircraft and one ground vehicle.
        AIRCRAFT_UNKNOWN: One aircraft and one unknown agent.
        VEHICLE_VEHICLE: Both agents are ground vehicles.
        VEHICLE_UNKNOWN: One ground vehicle and one unknown agent.
        UNKNOWN_UNKNOWN: Both agents are of unknown type.
    """

    AIRCRAFT_AIRCRAFT = "AIRCRAFT_AIRCRAFT"
    AIRCRAFT_VEHICLE = "AIRCRAFT_VEHICLE"
    AIRCRAFT_UNKNOWN = "AIRCRAFT_UNKNOWN"
    VEHICLE_VEHICLE = "VEHICLE_VEHICLE"
    VEHICLE_UNKNOWN = "VEHICLE_UNKNOWN"
    UNKNOWN_UNKNOWN = "UNKNOWN_UNKNOWN"


# Ordered pair → AgentPairType mapping (canonical order: lower int value first)
_PAIR_TYPE_MAP: dict[tuple[int, int], AgentPairType] = {
    (AgentType.AIRCRAFT.value, AgentType.AIRCRAFT.value): AgentPairType.AIRCRAFT_AIRCRAFT,
    (AgentType.AIRCRAFT.value, AgentType.VEHICLE.value): AgentPairType.AIRCRAFT_VEHICLE,
    (AgentType.AIRCRAFT.value, AgentType.UNKNOWN.value): AgentPairType.AIRCRAFT_UNKNOWN,
    (AgentType.VEHICLE.value, AgentType.VEHICLE.value): AgentPairType.VEHICLE_VEHICLE,
    (AgentType.VEHICLE.value, AgentType.UNKNOWN.value): AgentPairType.VEHICLE_UNKNOWN,
    (AgentType.UNKNOWN.value, AgentType.UNKNOWN.value): AgentPairType.UNKNOWN_UNKNOWN,
}


def raw_to_agent_type(raw_type: str) -> AgentType:
    """Convert raw agent type string to AgentType enum, with fallback to UNKNOWN.

    Args:
        raw_type: The raw agent type string from scenario data.

    Returns:
        Corresponding AgentType enum value, or AgentType.UNKNOWN if the raw type is unrecognized.
    """
    try:
        agent_type = AgentType(STRING_TO_AGENT_TYPE[raw_type])
    except (ValueError, TypeError):
        agent_type = AgentType.UNKNOWN
    return agent_type


def get_conflict_points_from_scenario(scenario: Scenario, *, scale: float = 1.0) -> NDArray[np.float32] | None:
    """Extract conflict points from scenario map data, if available.

    Currently, only hold-short lines are used as conflict points. These are represented as 2D coordinates in the
    scenario's static map data, so we convert them to 3D points with a z-value of 0 and apply the scale factor.

    Args:
        scenario: The scenario from which to extract conflict points.
        scale: The scale factor to apply to the conflict points.

    Returns:
        An array of shape (num_conflict_points, 3) containing the conflict point coordinates or ``None`` if no conflict
        points are available.
    """
    conflict_points: NDArray[np.float32] | None = None
    if scenario.static_map_data is not None and len(scenario.static_map_data.hold_short_points_xy) > 0:
        hold_short_points_xy = scenario.static_map_data.hold_short_points_xy[:, :2]
        zeros_z = np.zeros((hold_short_points_xy.shape[0], 1), dtype=hold_short_points_xy.dtype)
        hold_short_points = np.concatenate((hold_short_points_xy, zeros_z), axis=1)
        conflict_points = (hold_short_points * scale).astype(np.float32)
    return conflict_points


def get_agent_pair_type(type_a: AgentType, type_b: AgentType) -> AgentPairType:
    """Determine the pair type from two agent types, order-independent.

    Special internal types (TO_PREDICT, EGO_AGENT) are normalised to their base type before lookup.

    Args:
        type_a: Type of the first agent.
        type_b: Type of the second agent.

    Returns:
        The corresponding AgentPairType.
    """

    def _normalise(t: AgentType) -> AgentType:
        if t in (AgentType.TO_PREDICT, AgentType.EGO_AGENT):
            return AgentType.UNKNOWN
        return t

    a, b = _normalise(type_a), _normalise(type_b)
    key = (min(a.value, b.value), max(a.value, b.value))
    return _PAIR_TYPE_MAP.get(key, AgentPairType.UNKNOWN_UNKNOWN)


def moving_average(arr: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    """Apply a simple box moving average.

    Uses ``np.convolve`` with ``mode='same'``, which avoids boundary artefacts in the output length while still
    smoothing interior values.

    Args:
        arr: 1-D input array.
        window: Number of samples to average. Clamped to ``len(arr)`` if larger.

    Returns:
        Smoothed array of the same length as ``arr``.
    """
    if len(arr) == 0:
        return arr

    w = min(window, len(arr))
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(arr, kernel, mode="same").astype(np.float32)


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
        critical_is_min: When ``True``, ``CRITICAL`` returns the minimum value instead of the maximum. Use this for
            safety-relevant time features (TTC, THW, MTTCP, separation) where a smaller value indicates a more dangerous
            scenario. Ignored when ``criterion`` is ``ALL``.

    Returns:
        The original array when ``criterion`` is ``ALL``; a scalar otherwise. Returns ``float("nan")`` for empty
        arrays with scalar criteria.
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
