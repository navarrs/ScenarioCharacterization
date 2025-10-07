import itertools
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from characterization.schemas import Scenario
from characterization.utils.common import SMALL_EPS, AgentTrajectoryMasker


def compute_dists_to_conflict_points(
    conflict_points: NDArray[np.float32] | None, trajectories: NDArray[np.float32]
) -> NDArray[np.float32] | None:
    """Computes distances from agent trajectories to conflict points.

    Args:
        conflict_points: Array of conflict points with shape [num_conflict_points, 3] or None if no conflict points.
        trajectories: Array of agent trajectories with shape [num_agents, num_time_steps, 3].

    Returns:
        Distances from each agent at each timestep to each conflict point with shape [num_agents, num_time_steps,
            num_conflict_points], or None if conflict_points is None.
    """
    if conflict_points is None:
        return None
    diff = conflict_points[None, None, :] - trajectories[:, :, None, :]
    return np.linalg.norm(diff, axis=-1)  # shape (num_agents, num_time_steps, num_conflict_points)


def compute_agent_to_agent_closest_dists(positions: NDArray[np.float32]) -> NDArray[np.float32]:
    """Computes the closest distance between each agent and any other agent over their trajectories.

    Args:
        positions: Array of agent positions over time with shape [num_agents, num_time_steps, 3].

    Returns:
        Minimum distance from each agent to any other agent over time with shape [num_agents, num_time_steps]. NaN
            values are replaced with infinity.
    """
    # shape of dists is (num_agents, num_agents, num_time_steps)
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)

    # Replace self-distances (zero) with np.inf to ignore them in the min computation
    for t in range(dists.shape[-1]):
        np.fill_diagonal(dists[:, :, t], np.inf)

    # Return the minimum distance to any other agent over time, replacing NaNs with np.inf
    return np.nan_to_num(np.nanmin(dists, axis=-1), nan=np.inf).astype(np.float32)


def find_conflict_points(
    scenario: Scenario, ndim: int = 3, resample_factor: int = 3, intersection_threshold: float = 0.5
) -> dict[str, Any] | None:
    """Finds the conflict points in the map for a scenario.

    Args:
        scenario: The scenario for which to find conflict points.
        ndim: Number of dimensions to consider (2 or 3). Defaults to 3.
        resample_factor: Factor to resample polylines for better intersection detection. Defaults to 3.
        intersection_threshold: Distance threshold in meters to consider two lanes intersecting. Defaults to 0.5.

    Returns:
        Dictionary containing the conflict points in the map with keys:
            - 'static': Static conflict points (crosswalks, speed bumps, stop signs)
            - 'dynamic': Dynamic conflict points (traffic lights)
            - 'lane_intersections': Lane intersection points
            - 'all_conflict_points': All conflict points concatenated
            - 'agent_distances_to_conflict_points': Distances from each agent to each conflict point
        Returns None if no static map data is available.
    """
    if scenario.static_map_data is None:
        return None

    # polylines = static_map_info["all_polylines"]
    polylines = scenario.static_map_data.map_polylines
    if polylines is None or polylines.shape[0] == 0:
        return None

    agent_trajectories = AgentTrajectoryMasker(scenario.agent_data.agent_trajectories)
    agent_positions = agent_trajectories.agent_xyz_pos

    # Static Conflict Points: Crosswalks, Speed Bumps and Stop Signs
    static_conflict_points_list = []
    crosswalks_idxs = scenario.static_map_data.crosswalk_polyline_idxs
    speed_bumps_idxs = scenario.static_map_data.speed_bump_polyline_idxs
    stop_signs_idxs = scenario.static_map_data.stop_sign_polyline_idxs
    conflict_idxs = np.concatenate(
        [
            crosswalks_idxs if crosswalks_idxs is not None else np.empty((0, 2), dtype=int),
            speed_bumps_idxs if speed_bumps_idxs is not None else np.empty((0, 2), dtype=int),
            stop_signs_idxs if stop_signs_idxs is not None else np.empty((0, 2), dtype=int),
        ],
        axis=0,
    )
    for start, end in conflict_idxs:
        points = polylines[start:end][:, :ndim]
        points = resample(points, points.shape[0] * resample_factor)
        static_conflict_points_list.append(points)
    static_conflict_points = (
        np.concatenate(static_conflict_points_list) if len(static_conflict_points_list) > 0 else np.empty((0, ndim))
    )

    # Lane Intersections
    lane_intersections_list = []
    lane_idxs = scenario.static_map_data.lane_polyline_idxs
    if lane_idxs is not None:
        num_lanes = len(lane_idxs)

        lane_combinations = list(itertools.combinations(range(num_lanes), 2))
        for i, j in lane_combinations:
            lane_i_idxs, lane_j_idxs = lane_idxs[i], lane_idxs[j]
            lane_i = polylines[lane_i_idxs[0] : lane_i_idxs[1]][:, :ndim]
            lane_j = polylines[lane_j_idxs[0] : lane_j_idxs[1]][:, :ndim]

            dists_ij = np.linalg.norm(lane_i[:, None] - lane_j, axis=-1)

            i_idx, j_idx = np.where(dists_ij < intersection_threshold)
            i_idx, j_idx = np.unique(i_idx), np.unique(j_idx)

            # TODO: determine if two lanes are consecutive, but not entry/exit lanes. If this is the
            # case there'll be an intersection that is not a conflict point.
            # start_i, end_i = i_idx[:min_timesteps], i_idx[-min_timesteps:]
            # start_j, end_j = j_idx[:min_timesteps], j_idx[-min_timesteps:]
            # if (np.any(start_i < min_timesteps) and np.any(end_j > lane_j.shape[0] - min_timesteps)) or (
            #     np.any(start_j < min_timesteps) and np.any(end_i > lane_i.shape[0] - min_timesteps)
            # ):
            #     lanes_i_ee = lane_infos[i]["entry_lanes"] + lane_infos[i]["exit_lanes"]
            #     lanes_j_ee = lane_infos[j]["entry_lanes"] + lane_infos[j]["exit_lanes"]
            #     if j not in lanes_i_ee and i not in lanes_j_ee:
            #         continue

            if i_idx.shape[0] > 0:
                lane_intersections_list.append(lane_i[i_idx])

            if j_idx.shape[0] > 0:
                lane_intersections_list.append(lane_j[j_idx])

    lane_intersections = (
        np.concatenate(lane_intersections_list) if len(lane_intersections_list) > 0 else np.empty((0, 3))
    )

    # Dynamic Conflict Points: Traffic Lights
    dynamic_conflict_points = np.empty((0, ndim))
    if scenario.dynamic_map_data is not None:
        stops = (
            scenario.dynamic_map_data.stop_points
            if scenario.dynamic_map_data.stop_points is not None
            else np.empty((0, ndim))
        )
        if len(stops) > 0 and len(stops[0]) > 0 and stops[0].shape[1] == ndim:
            dynamic_conflict_points = np.concatenate(stops[0])

    # Concatenate all conflict points into a single array if they are not empty
    conflict_point_list = []
    if static_conflict_points.shape[0] > 0:
        conflict_point_list.append(static_conflict_points)
    if dynamic_conflict_points.shape[0] > 0:
        conflict_point_list.append(dynamic_conflict_points)
    if lane_intersections.shape[0] > 0:
        conflict_point_list.append(lane_intersections)

    conflict_points = np.concatenate(conflict_point_list, dtype=np.float32) if conflict_point_list else None

    dists_to_conflict_points = (
        compute_dists_to_conflict_points(conflict_points, agent_positions) if conflict_points is not None else None
    )

    return {
        "static": static_conflict_points,
        "dynamic": dynamic_conflict_points,
        "lane_intersections": lane_intersections,
        "all_conflict_points": conflict_points,
        "agent_distances_to_conflict_points": dists_to_conflict_points,
    }


def compute_k_closest_lanes(
    trajectory: NDArray[np.float32],
    mask: NDArray[np.bool_],
    lanes: NDArray[np.float32],
    k_lanes: int = 16,
    threshold: float = 10,
) -> NDArray[np.float32]:
    """Compute k closest lanes for each timestep in trajectory with projection onto lane segments.

    Args:
        trajectory: Agent trajectory with shape [num_timesteps, 3].
        mask: Valid timesteps mask with shape [num_timesteps].
        lanes: Lane polylines with shape [num_lanes, num_points, 3].
        k_lanes: Number of closest lanes to keep. Defaults to 16.
        threshold: Distance threshold in meters for lane consideration. Defaults to 10.

    Returns:
        Lane metadata for k closest lanes with shape [num_timesteps, k_lanes, 6]. Metadata format per lane:
            [distance, lane_point_idx, proj_x, proj_y, proj_z, lane_idx].
    """
    num_lanes, num_timesteps = lanes.shape[0], trajectory.shape[0]
    valid_traj = trajectory[mask]

    if len(valid_traj) == 0:
        return np.full((num_timesteps, k_lanes, 6), np.inf, dtype=np.float32)

    # Initialize lane metadata: [distance, lane_point_idx, proj_x, proj_y, proj_z, lane_idx]
    lane_meta = np.full((num_lanes, num_timesteps, 6), np.inf, dtype=np.float32)

    for lane_idx in range(num_lanes):
        lane = lanes[lane_idx]  # (num_lane_points, D=xyz)
        # Remove invalid points
        valid_lane = lane[~np.isinf(lane).any(axis=1)]
        if len(valid_lane) == 0:
            continue

        # Compute distances from valid trajectory points to all lane points (num_valid_timesteps, num_valid_lane_points)
        dists = np.linalg.norm(valid_traj[:, None] - valid_lane[None, :], axis=2)

        closest_point_idxs = dists.argmin(axis=1)
        closest_dists = dists.min(axis=1)

        # Early exit if lane is too far
        if closest_dists.min() > threshold:
            lane_meta[lane_idx, mask, 0] = closest_dists
            lane_meta[lane_idx, mask, 1] = closest_point_idxs.astype(np.float32)
            lane_meta[lane_idx, mask, 2:5] = valid_lane[closest_point_idxs]
            continue

        # Project points onto lane segments for better accuracy
        projected_dists, projected_points, projected_idxs = _project_onto_lane_segments(
            valid_traj, valid_lane, closest_point_idxs, closest_dists
        )

        # Store results
        lane_meta[lane_idx, mask, 0] = projected_dists
        lane_meta[lane_idx, mask, 1] = projected_idxs
        lane_meta[lane_idx, mask, 2:5] = projected_points

    # Select k closest lanes for each timestep
    return _select_k_closest_lanes(lane_meta, k_lanes, mask)


def _project_onto_lane_segments(
    trajectory_points: NDArray[np.float32],
    lane_points: NDArray[np.float32],
    closest_idxs: NDArray[np.int_],
    closest_dists: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Project trajectory points onto lane segments for more accurate distance computation.

    Args:
        trajectory_points: Valid trajectory points with shape [num_valid_timesteps, 3].
        lane_points: Valid lane points with shape [num_valid_lane_points, 3].
        closest_idxs: Indices of closest lane points for each trajectory point with shape [num_valid_timesteps].
        closest_dists: Distances to closest lane points with shape [num_valid_timesteps].

    Returns:
        Tuple containing:
            - projected_dists: Distances to projected points with shape [num_valid_timesteps]
            - projected_points: Projected points on lane segments with shape [num_valid_timesteps, 3]
            - projected_idxs: Indices (possibly fractional) of projected points along lane with shape
              [num_valid_timesteps]
    """
    projected_dists = closest_dists.copy()
    projected_points = lane_points[closest_idxs].copy()
    projected_idxs = closest_idxs.astype(np.float32)

    for i, point_idx in enumerate(closest_idxs):
        # Determine segment endpoints
        if point_idx == 0 and len(lane_points) > 1:
            seg_start, seg_end = 0, 1
        elif point_idx == len(lane_points) - 1 and len(lane_points) > 1:
            seg_start, seg_end = len(lane_points) - 2, len(lane_points) - 1
        elif len(lane_points) > 1:
            # Choose segment based on which neighbor is closer
            prev_dist = np.linalg.norm(trajectory_points[i] - lane_points[point_idx - 1])
            next_dist = np.linalg.norm(trajectory_points[i] - lane_points[point_idx + 1])
            if prev_dist < next_dist:
                seg_start, seg_end = point_idx - 1, point_idx
            else:
                seg_start, seg_end = point_idx, point_idx + 1
        else:
            continue  # Single point lane, keep original

        # Project onto segment
        seg_vec = lane_points[seg_end] - lane_points[seg_start]
        seg_len_sq = np.sum(seg_vec**2)

        if seg_len_sq < SMALL_EPS:
            continue  # Degenerate segment

        # Parameter t for projection (0 <= t <= 1 means point is on segment)
        point_vec = trajectory_points[i] - lane_points[seg_start]
        t = np.dot(point_vec, seg_vec) / seg_len_sq

        # Compute projected point and distance
        proj_point = lane_points[seg_start] + t * seg_vec
        proj_dist = np.linalg.norm(trajectory_points[i] - proj_point)

        # Update if projection is better and within segment bounds
        if 0 <= t <= 1 and proj_dist < projected_dists[i]:
            projected_dists[i] = proj_dist
            projected_points[i] = proj_point
            projected_idxs[i] = seg_start + t

    return projected_dists, projected_points, projected_idxs


def _select_k_closest_lanes(
    lane_meta: NDArray[np.float32], k_lanes: int, mask: NDArray[np.bool_]
) -> NDArray[np.float32]:
    """Select k closest lanes for each timestep.

    Args:
        lane_meta: Lane metadata with shape [num_lanes, num_timesteps, 6].
        k_lanes: Number of closest lanes to select.
        mask: Valid timesteps mask with shape [num_timesteps].

    Returns:
        Selected lane metadata with shape [num_timesteps, k_lanes, 6].
    """
    num_timesteps = lane_meta.shape[1]

    # Get indices of k closest lanes for each timestep
    k_lane_idxs = lane_meta[:, :, 0].argsort(axis=0)[:k_lanes]  # (k_lanes, num_timesteps)

    # Initialize output array
    k_lanes_meta = np.full((num_timesteps, k_lanes, 6), np.inf, dtype=np.float32)

    # Fill results for each timestep
    for t in range(num_timesteps):
        if mask[t]:
            selected_lanes = k_lane_idxs[:, t]
            k_lanes_meta[t] = lane_meta[selected_lanes, t]
            k_lanes_meta[t, :, -1] = selected_lanes  # Store lane indices
    return k_lanes_meta


def find_closest_lanes(
    scenario: Scenario, ndim: int = 3, k_closest: int = 10, threshold_distance: float = 10, subsample_factor: int = 1
) -> dict[str, Any] | None:
    """Finds the closest lanes to each agent in a scenario.

    Args:
        scenario: The scenario for which to find closest lanes.
        ndim: Number of dimensions to consider (2 or 3). Defaults to 3.
        k_closest: Number of closest lanes to consider for each agent. Defaults to 10.
        threshold_distance: Distance threshold in meters to consider a lane as close. Defaults to 10.
        subsample_factor: Factor to subsample lane polylines for efficiency. Defaults to 1 (no subsampling).

    Returns:
        Dictionary containing the closest lanes to each agent with key:
            - 'agent_closest_lanes': Closest lane metadata for each agent at each timestep. Metadata format
              per lane: [distance, lane_point_idx, proj_x, proj_y, proj_z, lane_idx].
        Returns None if no static map data is available.
    """
    if scenario.static_map_data is None:
        return None

    # Get lane data
    polylines = scenario.static_map_data.map_polylines
    lane_idxs = scenario.static_map_data.lane_polyline_idxs
    if polylines is None or polylines.shape[0] == 0 or lane_idxs is None or lane_idxs.shape[0] == 0:
        return None

    max_lane_points = np.max(lane_idxs[:, 1] - lane_idxs[:, 0])
    lanes = np.full((len(lane_idxs), max_lane_points, ndim), np.inf, dtype=np.float32)
    for i, (start, end) in enumerate(lane_idxs):
        lane_points = polylines[start:end][:, :ndim]
        lanes[i, : lane_points.shape[0], :] = lane_points
    lanes = lanes[:, ::subsample_factor]

    # Agent info
    agent_data = scenario.agent_data
    agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
    agent_positions = agent_trajectories.agent_xyz_pos
    num_timesteps = agent_positions.shape[1]
    agent_valid = agent_trajectories.agent_valid

    closest_lanes_data = np.full(
        shape=(agent_data.num_agents, num_timesteps, k_closest, 6), fill_value=np.inf, dtype=np.float32
    )
    for n in range(agent_data.num_agents):
        agent_closest_lanes = compute_k_closest_lanes(
            agent_positions[n],
            agent_valid[n].astype(bool).squeeze(-1),
            lanes,
            k_lanes=k_closest,
            threshold=threshold_distance,
        )
        closest_lanes_data[n] = agent_closest_lanes
    return {"agent_closest_lanes": closest_lanes_data}
