"""Utility functions for pairwise agent interaction feature computation."""

import numpy as np
from numpy.typing import NDArray

from characterization.utils.common import ReturnCriterion, return_by_criterion
from characterization.utils.constants import EPSILON


def get_joint_valid_mask(valid_a: NDArray[np.float32], valid_b: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Return the mask of timesteps where both agents have valid observations.

    Args:
        valid_a: Shape ``(T,)`` valid mask for agent A (1 = valid, 0 = interpolated).
        valid_b: Shape ``(T,)`` valid mask for agent B.

    Returns:
        Shape ``(T,)`` boolean array.
    """
    return (valid_a.squeeze(-1).astype(bool)) & (valid_b.squeeze(-1).astype(bool))


def compute_separation(
    pos_a: NDArray[np.float32],
    pos_b: NDArray[np.float32],
    joint_valid: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return per-timestep lateral and vertical separations for joint-valid timesteps.

    Args:
        pos_a: Shape ``(T, 3)`` positions of agent A in meters.
        pos_b: Shape ``(T, 3)`` positions of agent B in meters.
        joint_valid: Shape ``(T,)`` joint-validity mask.

    Returns:
        ``(lateral_dists, vertical_dists)``, each shape ``(N_valid,)`` in meters.
        Returns a pair of empty arrays if no joint-valid timestep exists.
    """
    if not np.any(joint_valid):
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    lateral_dists = np.linalg.norm(pos_a[joint_valid, :2] - pos_b[joint_valid, :2], axis=1).astype(np.float32)
    vertical_dists = np.abs(pos_a[joint_valid, 2] - pos_b[joint_valid, 2]).astype(np.float32)
    return lateral_dists, vertical_dists


def compute_separation_1d(
    pos_a: NDArray[np.float32],
    pos_b: NDArray[np.float32],
    joint_valid: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Return per-joint-valid-timestep 2D Euclidean separation distances.

    Args:
        pos_a: Shape ``(T, 3)`` positions of agent A.
        pos_b: Shape ``(T, 3)`` positions of agent B.
        joint_valid: Shape ``(T,)`` joint-validity mask.

    Returns:
        Shape ``(N_valid,)`` array of 2D distances. Empty if no joint-valid timesteps.
    """
    if not np.any(joint_valid):
        return np.empty(0, dtype=np.float32)
    return np.linalg.norm(pos_a[joint_valid, :2] - pos_b[joint_valid, :2], axis=1).astype(np.float32)


def compute_mttcp(
    pos_a: NDArray[np.float32],
    pos_b: NDArray[np.float32],
    speeds_a_ms: NDArray[np.float32],
    speeds_b_ms: NDArray[np.float32],
    conflict_points_m: NDArray[np.float32],
    joint_valid: NDArray[np.bool_],
    agent_to_conflict_point_max_distance: float,
    criterion: ReturnCriterion,
) -> float | NDArray[np.float32]:
    """Compute the aggregated time-to-conflict-point (MTTCP) difference between two agents.

    For each conflict point and each joint-valid timestep the function estimates the time for each agent to reach that
    point (distance / speed) and records |t_a - t_b|.

    Args:
        pos_a: Shape ``(T, 3)`` positions of agent A in meters.
        pos_b: Shape ``(T, 3)`` positions of agent B in meters.
        speeds_a_ms: Shape ``(T,)`` speeds of agent A in m/s.
        speeds_b_ms: Shape ``(T,)`` speeds of agent B in m/s.
        conflict_points_m: Shape ``(K, 3)`` conflict-point positions in meters.
        joint_valid: Shape ``(T,)`` joint-validity mask.
        agent_to_conflict_point_max_distance: Distance threshold in meters; only consider timesteps where at least one
            agent is within this distance of a conflict point.
        criterion: How to reduce the per-timestep |t_a - t_b| values. ``CRITICAL`` returns the minimum (simultaneous
            arrival = most dangerous); ``AVERAGE`` returns the mean.

    Returns:
        Aggregated MTTCP in seconds, or ``inf`` if no relevant interactions found.
        Returns ``nan`` if ``conflict_points_m`` is empty.
    """
    if len(conflict_points_m) == 0:
        return float("nan")
    if not np.any(joint_valid):
        return float("inf")

    pa = pos_a[joint_valid]  # (N, 3)
    pb = pos_b[joint_valid]
    sa = speeds_a_ms[joint_valid]
    sb = speeds_b_ms[joint_valid]

    all_diffs: list[NDArray[np.float32]] = []
    for cp in conflict_points_m:
        dist_a = np.linalg.norm(pa - cp, axis=1)  # (N,)
        dist_b = np.linalg.norm(pb - cp, axis=1)

        near = (dist_a <= agent_to_conflict_point_max_distance) | (dist_b <= agent_to_conflict_point_max_distance)
        if not np.any(near):
            continue

        t_a = dist_a[near] / (sa[near] + EPSILON)
        t_b = dist_b[near] / (sb[near] + EPSILON)
        all_diffs.append(np.abs(t_a - t_b).astype(np.float32))

    if not all_diffs:
        return float("inf")

    combined = np.concatenate(all_diffs)
    return return_by_criterion(combined, criterion, critical_is_min=True)


def _find_leader(
    pos_a: NDArray[np.float32],
    pos_b: NDArray[np.float32],
    headings_a: NDArray[np.float32],
    _headings_b: NDArray[np.float32],
) -> NDArray[np.int8]:
    """Determine the leading agent at each timestep.

    The leader is the agent that is further ahead along its own heading direction. If the angle from A to B projected
    onto A's heading is positive, B is ahead (A follows B). Otherwise A leads. Only the xy components of the positions
    are used because heading is defined on the horizontal plane.

    Args:
        pos_a: Shape ``(N, 3)`` positions.
        pos_b: Shape ``(N, 3)`` positions.
        headings_a: Shape ``(N,)`` headings of A in radians.
        _headings_b: Shape ``(N,)`` headings of B in radians (unused; reserved for future use).

    Returns:
        Shape ``(N,)`` int8 array; 0 = A leads, 1 = B leads.
    """
    diff_xy = pos_b[:, :2] - pos_a[:, :2]  # (N, 2) — horizontal displacement only
    heading_vec = np.stack([np.cos(headings_a), np.sin(headings_a)], axis=1)  # (N, 2)
    proj = np.sum(diff_xy * heading_vec, axis=1)  # positive → B is ahead
    return np.where(proj >= 0, np.int8(1), np.int8(0)).astype(np.int8)


def compute_thw(
    pos_a: NDArray[np.float32],
    pos_b: NDArray[np.float32],
    speeds_a_ms: NDArray[np.float32],
    speeds_b_ms: NDArray[np.float32],
    headings_a: NDArray[np.float32],
    headings_b: NDArray[np.float32],
    joint_valid: NDArray[np.bool_],
    heading_threshold_deg: float,
    criterion: ReturnCriterion,
) -> float | NDArray[np.float32]:
    """Compute the aggregated time headway (THW) between two agents.

    THW = separation / follower_speed. Only computed at timesteps where the agents are co-directional
    (|heading difference| <= ``heading_threshold_deg``).

    Args:
        pos_a: Shape ``(T, 3)`` positions of agent A in meters.
        pos_b: Shape ``(T, 3)`` positions of agent B in meters.
        speeds_a_ms: Shape ``(T,)`` speeds of agent A in m/s.
        speeds_b_ms: Shape ``(T,)`` speeds of agent B in m/s.
        headings_a: Shape ``(T,)`` headings of agent A in radians.
        headings_b: Shape ``(T,)`` headings of agent B in radians.
        joint_valid: Shape ``(T,)`` joint-validity mask.
        heading_threshold_deg: Maximum heading difference in degrees to consider co-directional.
        criterion: How to reduce per-timestep THW values. ``CRITICAL`` returns the minimum (smallest headway =
            most dangerous); ``AVERAGE`` returns the mean.

    Returns:
        Aggregated THW in seconds, or ``inf`` if no co-directional follower pair is found.
    """
    if not np.any(joint_valid):
        return float("inf")

    pa, pb = pos_a[joint_valid], pos_b[joint_valid]
    sa, sb = speeds_a_ms[joint_valid], speeds_b_ms[joint_valid]
    ha, hb = headings_a[joint_valid], headings_b[joint_valid]

    hdiff = np.abs(np.degrees(np.arctan2(np.sin(ha - hb), np.cos(ha - hb))))
    co_directional = hdiff <= heading_threshold_deg
    if not np.any(co_directional):
        return float("inf")

    pa_cd, pb_cd = pa[co_directional], pb[co_directional]
    sa_cd, sb_cd = sa[co_directional], sb[co_directional]
    ha_cd, hb_cd = ha[co_directional], hb[co_directional]

    leader = _find_leader(pa_cd, pb_cd, ha_cd, hb_cd)  # 0=A leads, 1=B leads
    separation = np.linalg.norm(pa_cd - pb_cd, axis=1)

    follower_speed = np.where(leader == 1, sa_cd, sb_cd)  # leader==1 means B leads → A follows
    with np.errstate(divide="ignore", invalid="ignore"):
        thw = separation / (follower_speed + EPSILON)

    valid_thw = thw[follower_speed > EPSILON].astype(np.float32)
    if len(valid_thw) == 0:
        return float("inf")
    return return_by_criterion(valid_thw, criterion, critical_is_min=True)


def velocity_components_2d(
    speeds_ms: NDArray[np.float32],
    headings_rad: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Decompose scalar speeds into (east, north) velocity components using heading.

    Args:
        speeds_ms: Per-timestep scalar speeds in m/s.
        headings_rad: Per-timestep headings in radians, measured clockwise from north.

    Returns:
        ``(vx, vy)`` where vx is the east component and vy is the north component, both shape ``(N,)``.
    """
    return speeds_ms * np.sin(headings_rad), speeds_ms * np.cos(headings_rad)


def closing_rate_2d(
    pa: NDArray[np.float32],
    pb: NDArray[np.float32],
    vx_a: NDArray[np.float32],
    vy_a: NDArray[np.float32],
    vx_b: NDArray[np.float32],
    vy_b: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Horizontal closing rate as the component of relative velocity along the A→B direction.

    Equivalent to the time-derivative of horizontal separation, negated (positive = shrinking). Only uses the xy
    (horizontal) plane; altitude is ignored.

    Args:
        pa: Shape ``(N, 3)`` positions of A (only xy used).
        pb: Shape ``(N, 3)`` positions of B (only xy used).
        vx_a: East velocity component of A in m/s.
        vy_a: North velocity component of A in m/s.
        vx_b: East velocity component of B in m/s.
        vy_b: North velocity component of B in m/s.

    Returns:
        Shape ``(N,)`` array. Positive = agents closing; negative = separating.
    """
    d = pb[:, :2] - pa[:, :2]
    dist = np.linalg.norm(d, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_hat = d / (dist[:, None] + EPSILON)
    v_rel = np.stack([vx_a - vx_b, vy_a - vy_b], axis=1)
    return np.einsum("ni,ni->n", v_rel, d_hat)


def projected_cpa_dist_2d(
    pa: NDArray[np.float32],
    pb: NDArray[np.float32],
    vx_a: NDArray[np.float32],
    vy_a: NDArray[np.float32],
    vx_b: NDArray[np.float32],
    vy_b: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Horizontal closest-point-of-approach distance under constant-velocity assumption.

    For each timestep, extrapolates both agents at constant velocity and returns the minimum horizontal separation they
    will reach. Used to suppress TTC/DRAC for pairs whose paths do not converge within a meaningful threshold (e.g.
    parallel runways, non-intersecting paths), and as a candidate-level filter in pair selection.

    When agents have the same velocity (v_rel ≈ 0), separation stays constant, so CPA = current separation (t_cpa
    clamped to 0).

    Args:
        pa: Positions of A (only xy used), shape ``(N, 2)`` or ``(N, 3)``.
        pb: Positions of B (only xy used), shape ``(N, 2)`` or ``(N, 3)``.
        vx_a: East velocity component of A in m/s.
        vy_a: North velocity component of A in m/s.
        vx_b: East velocity component of B in m/s.
        vy_b: North velocity component of B in m/s.

    Returns:
        Shape ``(N,)`` projected horizontal CPA distances in meters.
    """
    d = pa[:, :2] - pb[:, :2]
    v_rel = np.stack([vx_a - vx_b, vy_a - vy_b], axis=1)
    v_rel_sq = np.sum(v_rel**2, axis=1)
    t_cpa = np.where(v_rel_sq > EPSILON, -np.sum(d * v_rel, axis=1) / np.maximum(v_rel_sq, EPSILON), 0.0)
    t_cpa = np.maximum(t_cpa, 0.0)
    r_cpa = d + v_rel * t_cpa[:, None]
    return np.linalg.norm(r_cpa, axis=1).astype(np.float32)


def compute_pair_gate(
    pos_a: NDArray[np.float32],
    pos_b: NDArray[np.float32],
    speeds_a_ms: NDArray[np.float32],
    speeds_b_ms: NDArray[np.float32],
    headings_a_rad: NDArray[np.float32],
    headings_b_rad: NDArray[np.float32],
    joint_valid: NDArray[np.bool_],
    max_cpa_dist: float,
    max_vertical_separation: float,
) -> NDArray[np.bool_]:
    """Return a refined joint-validity mask gating out timesteps where TTC/DRAC are not meaningful.

    Two per-timestep checks are applied on top of the existing ``joint_valid`` mask:

    - **CPA gate**: suppresses timesteps where the constant-velocity projected closest-point-of-approach (CPA) distance
      exceeds ``max_cpa_dist``.
    - **Altitude gate**: suppresses timesteps where the vertical separation ``|z_a - z_b|`` exceeds
      ``max_vertical_separation``.

    Args:
        pos_a: Shape ``(T, 3)`` positions of A in meters.
        pos_b: Shape ``(T, 3)`` positions of B in meters.
        speeds_a_ms: Shape ``(T,)`` speeds of A in m/s.
        speeds_b_ms: Shape ``(T,)`` speeds of B in m/s.
        headings_a_rad: Shape ``(T,)`` headings of A in radians (clockwise from north).
        headings_b_rad: Shape ``(T,)`` headings of B in radians (clockwise from north).
        joint_valid: Shape ``(T,)`` existing joint-validity mask.
        max_cpa_dist: CPA gate threshold in meters.
        max_vertical_separation: Altitude gate threshold in meters. Pass ``float("inf")`` to disable.

    Returns:
        Shape ``(T,)`` boolean mask that is a subset of ``joint_valid``.
    """
    if not np.any(joint_valid):
        return joint_valid
    pa, pb = pos_a[joint_valid], pos_b[joint_valid]
    vx_a, vy_a = velocity_components_2d(speeds_a_ms[joint_valid], headings_a_rad[joint_valid])
    vx_b, vy_b = velocity_components_2d(speeds_b_ms[joint_valid], headings_b_rad[joint_valid])
    altitude_ok = np.abs(pa[:, 2] - pb[:, 2]) <= max_vertical_separation
    cpa_ok = projected_cpa_dist_2d(pa, pb, vx_a, vy_a, vx_b, vy_b) <= max_cpa_dist
    result = joint_valid.copy()
    result[joint_valid] = altitude_ok & cpa_ok
    return result


def compute_ttc(
    pos_a: NDArray[np.float32],
    pos_b: NDArray[np.float32],
    speeds_a_ms: NDArray[np.float32],
    speeds_b_ms: NDArray[np.float32],
    headings_a_rad: NDArray[np.float32],
    headings_b_rad: NDArray[np.float32],
    joint_valid: NDArray[np.bool_],
    criterion: ReturnCriterion,
) -> float | NDArray[np.float32]:
    """Compute the aggregated time-to-collision (TTC) using velocity-vector closing rate.

    TTC at each valid timestep = horizontal_separation / closing_rate, where the closing rate is the projection of the
    relative velocity (v_a - v_b) onto the A->B unit vector. Only timesteps with a positive closing rate (agents
    approaching) contribute. ``joint_valid`` should already incorporate any geometry-based filtering (e.g. via
    ``compute_pair_gate``) before being passed here.

    Args:
        pos_a: Shape ``(T, 3)`` positions of agent A in meters.
        pos_b: Shape ``(T, 3)`` positions of agent B in meters.
        speeds_a_ms: Shape ``(T,)`` speeds of agent A in m/s.
        speeds_b_ms: Shape ``(T,)`` speeds of agent B in m/s.
        headings_a_rad: Shape ``(T,)`` headings of A in radians, clockwise from north.
        headings_b_rad: Shape ``(T,)`` headings of B in radians, clockwise from north.
        joint_valid: Shape ``(T,)`` mask — True where both agents are observed and the geometry is meaningful.
        criterion: How to reduce per-timestep TTC values. ``CRITICAL`` returns the minimum (soonest collision =
            most dangerous); ``AVERAGE`` returns the mean. ``ALL`` returns a shape-``(T,)`` array with ``nan``
            at joint-invalid and non-closing timesteps.

    Returns:
        Aggregated TTC in seconds, shape-``(T,)`` array when ``criterion`` is ``ALL``, or ``inf`` if agents are not
            approaching (scalar criteria only).
    """
    n_t = len(joint_valid)
    if not np.any(joint_valid):
        return np.full(n_t, np.nan, dtype=np.float32) if criterion == ReturnCriterion.ALL else float("inf")

    pa, pb = pos_a[joint_valid], pos_b[joint_valid]
    sa, sb = speeds_a_ms[joint_valid], speeds_b_ms[joint_valid]
    ha, hb = headings_a_rad[joint_valid], headings_b_rad[joint_valid]

    vx_a, vy_a = velocity_components_2d(sa, ha)
    vx_b, vy_b = velocity_components_2d(sb, hb)

    dist = np.linalg.norm(pb[:, :2] - pa[:, :2], axis=1)
    closing_rate = closing_rate_2d(pa, pb, vx_a, vy_a, vx_b, vy_b)
    closing = closing_rate > 0

    if criterion == ReturnCriterion.ALL:
        result = np.full(n_t, np.nan, dtype=np.float32)
        if np.any(closing):
            valid_indices = np.where(joint_valid)[0]
            result[valid_indices[closing]] = dist[closing] / (closing_rate[closing] + EPSILON)
        return return_by_criterion(result, criterion, critical_is_min=True)

    if not np.any(closing):
        return float("inf")

    ttc_vals = (dist[closing] / (closing_rate[closing] + EPSILON)).astype(np.float32)
    return return_by_criterion(ttc_vals, criterion, critical_is_min=True)


def compute_drac(
    pos_a: NDArray[np.float32],
    pos_b: NDArray[np.float32],
    speeds_a_ms: NDArray[np.float32],
    speeds_b_ms: NDArray[np.float32],
    headings_a_rad: NDArray[np.float32],
    headings_b_rad: NDArray[np.float32],
    joint_valid: NDArray[np.bool_],
    agent_max_deceleration: float,
    criterion: ReturnCriterion,
) -> float | NDArray[np.float32]:
    """Compute the aggregated deceleration rate to avoid collision (DRAC).

    DRAC at each valid timestep = closing_rate^2 / (2 * horizontal_separation), where closing_rate is the magnitude
    of the closing component of the relative velocity vector. Per-timestep values are clamped to
    ``agent_max_deceleration`` before aggregation. ``joint_valid`` should incorporate geometry-based filtering (via
    ``compute_pair_gate``) before being passed here.

    Args:
        pos_a: Shape ``(T, 3)`` positions of agent A in meters.
        pos_b: Shape ``(T, 3)`` positions of agent B in meters.
        speeds_a_ms: Shape ``(T,)`` speeds of agent A in m/s.
        speeds_b_ms: Shape ``(T,)`` speeds of agent B in m/s.
        headings_a_rad: Shape ``(T,)`` headings of A in radians, clockwise from north.
        headings_b_rad: Shape ``(T,)`` headings of B in radians, clockwise from north.
        joint_valid: Shape ``(T,)`` mask after geometry gating.
        agent_max_deceleration: Maximum feasible deceleration in m/s² (used as clip upper bound).
        criterion: How to reduce per-timestep DRAC values. ``CRITICAL`` returns the maximum (highest required
            braking = most dangerous); ``AVERAGE`` returns the mean. ``ALL`` returns a shape-``(T,)`` array with
            ``nan`` at joint-invalid timesteps.

    Returns:
        Aggregated DRAC in m/s², shape-``(T,)`` array when ``criterion`` is ``ALL``, or ``nan`` if not computable
            (scalar criteria only).
    """
    n_t = len(joint_valid)
    if not np.any(joint_valid):
        return np.full(n_t, np.nan, dtype=np.float32) if criterion == ReturnCriterion.ALL else float("nan")

    pa, pb = pos_a[joint_valid], pos_b[joint_valid]
    sa, sb = speeds_a_ms[joint_valid], speeds_b_ms[joint_valid]
    ha, hb = headings_a_rad[joint_valid], headings_b_rad[joint_valid]

    vx_a, vy_a = velocity_components_2d(sa, ha)
    vx_b, vy_b = velocity_components_2d(sb, hb)

    separation = np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1)
    closing_rate = closing_rate_2d(pa, pb, vx_a, vy_a, vx_b, vy_b)
    closing_positive = np.maximum(closing_rate, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        drac = closing_positive**2 / (2 * separation + EPSILON)

    drac_clipped = np.clip(drac, 0.0, agent_max_deceleration).astype(np.float32)

    if criterion == ReturnCriterion.ALL:
        result = np.full(n_t, np.nan, dtype=np.float32)
        result[joint_valid] = drac_clipped
        return return_by_criterion(result, criterion)

    return return_by_criterion(drac_clipped, criterion)
