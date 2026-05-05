"""Runway-based predicates for counterfactual loss-of-separation rule checking.

Implements the agent-runway and runway-runway predicates defined in COUNTERFACTUAL_RULES.md,
translated to work with trajectory data and airport map (graph + polylines).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString

from safeair.scenario_characterization.common import TrajectoryType
from safeair.scenario_characterization.features.individual_utils import classify_trajectory_type
from safeair.schemas.scenario import MapData
from safeair.schemas.scenario_features import CharacterizationParameters
from safeair.utils.constants import EPSILON
from safeair.utils.scenario_types import AgentTrajectory

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RunwayGeometry:
    """Geometric representation of a single logical runway derived from map data.

    Attributes:
        runway_id: Runway name string derived from the OSM ``ref`` edge attribute, e.g. ``"4L/22R"``.
        segments_m: Shape ``(K, 4)`` array of ``[x_start, y_start, x_end, y_end]`` in meters.
        direction: Shape ``(2,)`` dominant unit direction vector for the runway centerline.
        thresholds: Shape ``(2, 2)`` array of the two outermost endpoints ``[near_xy, far_xy]`` in meters.
    """

    runway_id: str
    segments_m: NDArray[np.float32]
    direction: NDArray[np.float32]
    thresholds: NDArray[np.float32]


@dataclass
class AgentRunwayPredicates:
    """Truth values of the four agent-runway predicates for one agent.

    Each dict maps a ``runway_id`` string to a boolean indicating whether the predicate holds
    for that runway.

    Attributes:
        landing_runway: True if the agent is landing on the runway and within 1 NM of a threshold.
        takeoff_runway: True if the agent is taking off from the runway.
        cross_runway: True if the agent's path crosses the runway without being heading-aligned.
        holding_on_runway: True if the agent is on the runway at low speed. This covers both
            Line Up and Wait (LUAW — aircraft lined up on the runway waiting for takeoff clearance)
            and post-landing scenarios where the aircraft has not yet vacated the runway.
    """

    landing_runway: dict[str, bool] = field(default_factory=dict)
    takeoff_runway: dict[str, bool] = field(default_factory=dict)
    cross_runway: dict[str, bool] = field(default_factory=dict)
    holding_on_runway: dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry helpers (private)
# ---------------------------------------------------------------------------


def _point_to_segment_perpendicular_distance(
    points: NDArray[np.float32],
    segment_start: NDArray[np.float32],
    segment_end: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Perpendicular distance from each point to the nearest location on a finite segment.

    Args:
        points: Shape ``(T, 2)`` array of 2-D points in meters.
        segment_start: Shape ``(2,)`` segment start in meters.
        segment_end: Shape ``(2,)`` segment end in meters.

    Returns:
        Shape ``(T,)`` distance array in meters.
    """
    segment_vector = segment_end - segment_start
    segment_length_sq = float(np.dot(segment_vector, segment_vector))
    if segment_length_sq < EPSILON:
        return np.linalg.norm(points - segment_start[None], axis=-1).astype(np.float32)
    # Project each point onto the segment, clamping to [0, 1] to stay within the segment bounds.
    projection = np.clip(((points - segment_start[None]) @ segment_vector) / segment_length_sq, 0.0, 1.0)
    closest = segment_start[None] + projection[:, None] * segment_vector[None]
    return np.linalg.norm(points - closest, axis=-1).astype(np.float32)


def _min_perpendicular_dist_to_runway(
    pos_xy_m: NDArray[np.float32],
    runway: RunwayGeometry,
) -> NDArray[np.float32]:
    """Minimum perpendicular distance from each trajectory point to any segment of the runway.

    Args:
        pos_xy_m: Shape ``(T, 2)`` XY positions in meters.
        runway: Runway geometry.

    Returns:
        Shape ``(T,)`` minimum distance array in meters.
    """
    min_dist = np.full(len(pos_xy_m), np.inf, dtype=np.float32)
    for row in runway.segments_m:
        segment_start = row[:2].astype(np.float32)
        segment_end = row[2:].astype(np.float32)
        dist = _point_to_segment_perpendicular_distance(pos_xy_m, segment_start, segment_end)
        min_dist = np.minimum(min_dist, dist)
    return min_dist


def _heading_alignment_mask(
    heading_rad: NDArray[np.float32],
    runway: RunwayGeometry,
    threshold_deg: float,
) -> NDArray[np.bool_]:
    """Return True where heading is aligned with the runway direction (within threshold, mod 180°).

    Alignment is checked mod 180° so that traffic traveling in either direction along the runway
    is considered aligned. For example, on runway "4L/22R", both heading ~40° and ~220° are aligned.

    Args:
        heading_rad: Shape ``(T,)`` headings in radians.
        runway: Runway geometry (``direction`` field is used).
        threshold_deg: Max heading difference in degrees for alignment.

    Returns:
        Shape ``(T,)`` boolean array.
    """
    runway_heading = float(np.arctan2(runway.direction[1], runway.direction[0]))
    diff_rad = heading_rad - runway_heading
    diff_deg = np.abs(np.degrees(np.arctan2(np.sin(diff_rad), np.cos(diff_rad))))
    # Fold [0, 180] → [0, 90] so both runway directions map to the same range.
    diff_folded = np.minimum(diff_deg, 180.0 - diff_deg)
    return diff_folded < threshold_deg


def _is_agent_on_runway_mask(
    pos_xy_m: NDArray[np.float32],
    heading_rad: NDArray[np.float32],
    runway: RunwayGeometry,
    corridor_width_m: float,
    heading_threshold_deg: float,
) -> NDArray[np.bool_]:
    """Return a per-timestep boolean mask: True where the agent is on the runway corridor.

    Combines :func:`_min_perpendicular_dist_to_runway` (position check) and
    :func:`_heading_alignment_mask` (heading check). Both conditions must hold simultaneously.

    Args:
        pos_xy_m: Shape ``(T, 2)`` XY positions in meters.
        heading_rad: Shape ``(T,)`` headings in radians.
        runway: Runway geometry.
        corridor_width_m: Half-width of the runway corridor in meters.
        heading_threshold_deg: Max heading difference (mod 180°) for alignment.

    Returns:
        Shape ``(T,)`` boolean array.
    """
    within_corridor: NDArray[np.bool_] = _min_perpendicular_dist_to_runway(pos_xy_m, runway) < corridor_width_m
    heading_aligned = _heading_alignment_mask(heading_rad, runway, heading_threshold_deg)
    return within_corridor & heading_aligned


def _compute_runway_thresholds(
    segments_m: NDArray[np.float32],
    direction: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Return the two outermost runway endpoints along the centerline direction.

    All segment endpoints are projected onto ``direction``; the points with the smallest and
    largest projections become the two runway thresholds.

    Args:
        segments_m: Shape ``(K, 4)`` array of segment endpoints in meters.
        direction: Shape ``(2,)`` unit direction vector for the runway.

    Returns:
        Shape ``(2, 2)`` array of ``[start_xy, end_xy]`` in meters.
    """
    endpoints = np.vstack([segments_m[:, :2], segments_m[:, 2:]])  # (2K, 2)
    projections = endpoints @ direction
    return np.stack([endpoints[int(np.argmin(projections))], endpoints[int(np.argmax(projections))]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Runway geometry builder
# ---------------------------------------------------------------------------


def build_runway_geometries(map_data: MapData, scale_to_m: float) -> dict[str, RunwayGeometry]:
    """Build a mapping from runway ID to its geometry.

    The airport surface movement graph stores one edge per runway polyline segment. Each edge carries:
    - ``osmid``: matches the segment's index in ``runway_polylines_ids``.
    - ``ref``: OpenStreetMap runway name, e.g. ``"4L/22R"`` (used as the ``runway_id``).
    - ``length``: segment length in meters.

    The function groups all segments belonging to the same runway by their ``ref`` value, then
    computes the dominant direction (average unit vector, aligned to the same half-plane), and
    the two threshold endpoints (outermost segment endpoints along the centerline).

    Segment endpoint reconstruction:
        ``runway_polylines_xy`` stores ``[x_start, y_start, x_dir, y_dir]`` where ``x_dir, y_dir``
        is a unit direction vector. The endpoint is:
        ``end = start + edge_length_m * direction``

    Args:
        map_data: Scenario map data containing runway polylines and airport graph.
        scale_to_m: Conversion factor from scenario coordinate units to meters.

    Returns:
        Dict mapping each runway ID string to its :class:`RunwayGeometry`. Returns an empty dict
        if the map has no graph or no runway polylines with matching graph edges.
    """
    # Build a lookup from OSM edge ID to the corresponding row in runway_polylines_ids.
    osmid_to_row: dict[int, int] = {int(osmid): row_idx for row_idx, osmid in enumerate(map_data.runway_polylines_ids)}

    # Collect segments per runway ID by iterating all graph edges.
    runway_id_to_segments: dict[str, list[NDArray[np.float32]]] = {}
    for _u, _v, edge_data in map_data.graph.edges(data=True):
        osmid = edge_data.get("osmid")
        runway_id = edge_data.get("ref")
        length_m = edge_data.get("length")
        # Skip edges that lack any required attribute or don't correspond to a runway polyline.
        if osmid is None or runway_id is None or length_m is None or int(osmid) not in osmid_to_row:
            continue

        row_idx = osmid_to_row[int(osmid)]
        # Runway polylines are stored in scenario units; convert to meters for geometry computations.
        start_m = (map_data.runway_polylines_xy[row_idx, :2] * scale_to_m).astype(np.float32)
        unit_direction = map_data.runway_polylines_xy[row_idx, 2:].astype(np.float32)
        end_m = start_m + float(length_m) * unit_direction
        runway_id_to_segments.setdefault(runway_id, []).append(
            np.array([start_m[0], start_m[1], end_m[0], end_m[1]], dtype=np.float32)
        )

    # Build a RunwayGeometry for each runway ID.
    runway_geometries: dict[str, RunwayGeometry] = {}
    for runway_id, segments in runway_id_to_segments.items():
        segments_m = np.array(segments, dtype=np.float32)  # (K, 4)

        # Compute the dominant direction as the mean of all segment unit vectors, after aligning
        # them all to the same half-plane (opposite directions get flipped before averaging).
        raw_directions = segments_m[:, 2:4] - segments_m[:, 0:2]
        norms = np.linalg.norm(raw_directions, axis=-1, keepdims=True)
        valid = norms.squeeze(-1) > EPSILON
        if not np.any(valid):
            continue

        unit_directions = raw_directions[valid] / norms[valid]
        flip = (unit_directions @ unit_directions[0]) < 0.0
        unit_directions[flip] = -unit_directions[flip]
        mean_direction = unit_directions.mean(axis=0)
        norm = float(np.linalg.norm(mean_direction))
        if norm < EPSILON:
            continue

        direction = (mean_direction / norm).astype(np.float32)
        thresholds = _compute_runway_thresholds(segments_m, direction)
        runway_geometries[runway_id] = RunwayGeometry(
            runway_id=runway_id, segments_m=segments_m, direction=direction, thresholds=thresholds
        )

    return runway_geometries


# ---------------------------------------------------------------------------
# Agent-runway predicate evaluators
# ---------------------------------------------------------------------------


def eval_landing_runway(
    traj: AgentTrajectory,
    runway: RunwayGeometry,
    scale_to_m: float,
    speed_to_ms: float,
    config: CharacterizationParameters,
) -> bool:
    """Return True if the agent is landing on this runway.

    Conditions (all must hold):
    - Trajectory classified as LANDING or LANDED.
    - Agent is within the runway corridor AND heading-aligned at some valid timestep.
    - At some such timestep the agent is within 1 NM (``landing_max_threshold_dist_m``) of either
      runway threshold. The 1 NM criterion matches the ``landing_runway`` predicate definition in
      COUNTERFACTUAL_RULES.md ("within 1 mile or closer").

    Args:
        traj: Agent trajectory.
        runway: Target runway geometry.
        scale_to_m: Conversion factor from scenario units to meters.
        speed_to_ms: Conversion factor from stored speed units to m/s.
        config: Characterization parameters.

    Returns:
        Boolean predicate value.
    """
    positions = traj.get_valid_positions(scale=scale_to_m)
    speeds = traj.get_valid_speeds(speed_to_ms=speed_to_ms)
    if len(positions) == 0:
        return False

    trajectory_type = classify_trajectory_type(positions, speeds, config=config)
    if trajectory_type not in (TrajectoryType.LANDING, TrajectoryType.LANDED):
        return False

    pos_xy_m = (traj.xy_position * scale_to_m).astype(np.float32)
    heading_rad = traj.heading.squeeze(-1).astype(np.float32)
    valid_mask = traj.get_valid_mask()

    on_runway_valid = valid_mask & _is_agent_on_runway_mask(
        pos_xy_m, heading_rad, runway, config.runway_corridor_width_m, config.heading_threshold
    )
    if not np.any(on_runway_valid):
        return False

    positions_on_runway = pos_xy_m[on_runway_valid]
    return any(
        bool(
            np.any(
                np.linalg.norm(positions_on_runway - threshold_m[None], axis=-1) < config.landing_max_threshold_dist_m
            )
        )
        for threshold_m in runway.thresholds
    )


def eval_takeoff_runway(
    traj: AgentTrajectory,
    runway: RunwayGeometry,
    scale_to_m: float,
    speed_to_ms: float,
    config: CharacterizationParameters,
) -> bool:
    """Return True if the agent is taking off from this runway.

    Conditions (all must hold):
    - Trajectory classified as TAKEOFF.
    - Agent is within the runway corridor AND heading-aligned at some valid timestep.

    Args:
        traj: Agent trajectory.
        runway: Target runway geometry.
        scale_to_m: Conversion factor from scenario units to meters.
        speed_to_ms: Conversion factor from stored speed units to m/s.
        config: Characterization parameters.

    Returns:
        Boolean predicate value.
    """
    positions = traj.get_valid_positions(scale=scale_to_m)
    speeds = traj.get_valid_speeds(speed_to_ms=speed_to_ms)
    if len(positions) == 0:
        return False

    if classify_trajectory_type(positions, speeds, config=config) != TrajectoryType.TAKEOFF:
        return False

    pos_xy_m = (traj.xy_position * scale_to_m).astype(np.float32)
    heading_rad = traj.heading.squeeze(-1).astype(np.float32)
    valid_mask = traj.get_valid_mask()
    on_runway = _is_agent_on_runway_mask(
        pos_xy_m, heading_rad, runway, config.runway_corridor_width_m, config.heading_threshold
    )
    return bool(np.any(valid_mask & on_runway))


def eval_cross_runway(
    traj: AgentTrajectory,
    runway: RunwayGeometry,
    scale_to_m: float,
    speed_to_ms: float,  # noqa: ARG001
    config: CharacterizationParameters,
) -> bool:
    """Return True if the agent crosses the runway without being heading-aligned.

    An agent is crossing the runway when it enters the corridor at some valid timestep with a heading
    that is NOT aligned with the runway direction (i.e. taxiing perpendicular across the runway,
    rather than operating along it). Two conditions must both hold:
    1. Per-timestep check: at least one valid timestep is within the corridor AND not heading-aligned.
    2. Shapely check: the full valid trajectory line geometrically intersects the runway centerline.
       This second check filters out agents that merely graze the corridor edge without truly crossing.

    Args:
        traj: Agent trajectory.
        runway: Target runway geometry.
        scale_to_m: Conversion factor from scenario units to meters.
        speed_to_ms: Unused; present for API symmetry with other predicate evaluators.
        config: Characterization parameters.

    Returns:
        Boolean predicate value.
    """
    pos_xy_m = (traj.xy_position * scale_to_m).astype(np.float32)
    heading_rad = traj.heading.squeeze(-1).astype(np.float32)
    valid_mask = traj.get_valid_mask()

    within_corridor: NDArray[np.bool_] = (
        _min_perpendicular_dist_to_runway(pos_xy_m, runway) < config.runway_corridor_width_m
    )
    not_aligned = ~_heading_alignment_mask(heading_rad, runway, config.heading_threshold)

    if not np.any(valid_mask & within_corridor & not_aligned):
        return False

    valid_positions = pos_xy_m[valid_mask]
    if len(valid_positions) < 2:  # noqa: PLR2004
        return False
    trajectory_line = LineString(valid_positions.tolist())
    runway_centerline = LineString(runway.thresholds.tolist())
    return bool(trajectory_line.intersects(runway_centerline))


def eval_holding_on_runway(
    traj: AgentTrajectory,
    runway: RunwayGeometry,
    scale_to_m: float,
    speed_to_ms: float,
    config: CharacterizationParameters,
) -> bool:
    """Return True if the agent is holding on the runway.

    This covers two cases:
    - Line Up and Wait (LUAW): aircraft lined up on the runway awaiting takeoff clearance.
    - Post-landing hold: aircraft that has landed but not yet vacated the runway.

    Conditions (all must hold at some valid timestep):
    - Agent is within the runway corridor AND heading-aligned.
    - Agent speed is below ``holding_max_speed_ms``.

    The speed threshold is set higher than ``max_stationary_speed`` because a LUAW aircraft may
    still be moving slowly while taxiing into position.

    Args:
        traj: Agent trajectory.
        runway: Target runway geometry.
        scale_to_m: Conversion factor from scenario units to meters.
        speed_to_ms: Conversion factor from stored speed units to m/s.
        config: Characterization parameters.

    Returns:
        Boolean predicate value.
    """
    pos_xy_m = (traj.xy_position * scale_to_m).astype(np.float32)
    heading_rad = traj.heading.squeeze(-1).astype(np.float32)
    speeds_ms = (traj.speed.squeeze(-1) * speed_to_ms).astype(np.float32)
    valid_mask = traj.get_valid_mask()

    on_runway = _is_agent_on_runway_mask(
        pos_xy_m, heading_rad, runway, config.runway_corridor_width_m, config.heading_threshold
    )
    slow: NDArray[np.bool_] = speeds_ms < config.holding_max_speed_ms
    return bool(np.any(valid_mask & on_runway & slow))


# ---------------------------------------------------------------------------
# Runway-runway predicate evaluators
# ---------------------------------------------------------------------------


def are_same_runway(runway_id_1: str, runway_id_2: str) -> bool:
    """Return True if both runway IDs name the same logical runway."""
    return runway_id_1 == runway_id_2


def are_parallel_runways(
    runway_1: RunwayGeometry,
    runway_2: RunwayGeometry,
    threshold_deg: float = 20.0,
) -> bool:
    """Return True if the two runways are parallel (heading differs by less than ``threshold_deg`` mod 180°).

    Args:
        runway_1: First runway geometry.
        runway_2: Second runway geometry.
        threshold_deg: Maximum heading difference in degrees (mod 180°).

    Returns:
        Boolean.
    """
    cos_angle = float(abs(np.dot(runway_1.direction, runway_2.direction)))
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0))))
    return angle_deg < threshold_deg


def are_intersecting_runways(runway_1: RunwayGeometry, runway_2: RunwayGeometry) -> bool:
    """Return True if the two runway centerlines geometrically intersect.

    Args:
        runway_1: First runway geometry.
        runway_2: Second runway geometry.

    Returns:
        Boolean.
    """
    centerline_1 = LineString(runway_1.thresholds.tolist())
    centerline_2 = LineString(runway_2.thresholds.tolist())
    return bool(centerline_1.intersects(centerline_2))


# ---------------------------------------------------------------------------
# Aggregate predicate computation
# ---------------------------------------------------------------------------


def compute_agent_runway_predicates(
    traj: AgentTrajectory,
    runway_geometries: dict[str, RunwayGeometry],
    scale_to_m: float,
    speed_to_ms: float,
    config: CharacterizationParameters,
) -> AgentRunwayPredicates:
    """Compute all four agent-runway predicates for one agent against every runway.

    Args:
        traj: Agent trajectory.
        runway_geometries: Dict mapping runway IDs to geometry (from :func:`build_runway_geometries`).
        scale_to_m: Conversion factor from scenario units to meters.
        speed_to_ms: Conversion factor from stored speed units to m/s.
        config: Characterization parameters.

    Returns:
        :class:`AgentRunwayPredicates` with one entry per runway ID for each predicate.
    """
    landing: dict[str, bool] = {}
    takeoff: dict[str, bool] = {}
    cross: dict[str, bool] = {}
    holding: dict[str, bool] = {}

    for runway_id, runway in runway_geometries.items():
        landing[runway_id] = eval_landing_runway(traj, runway, scale_to_m, speed_to_ms, config)
        takeoff[runway_id] = eval_takeoff_runway(traj, runway, scale_to_m, speed_to_ms, config)
        cross[runway_id] = eval_cross_runway(traj, runway, scale_to_m, speed_to_ms, config)
        holding[runway_id] = eval_holding_on_runway(traj, runway, scale_to_m, speed_to_ms, config)

    return AgentRunwayPredicates(
        landing_runway=landing,
        takeoff_runway=takeoff,
        cross_runway=cross,
        holding_on_runway=holding,
    )
