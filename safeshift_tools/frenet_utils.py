import numpy as np
import pandas as pd

def simple_cartesian2frenet(traj, lane):
    points = traj[:, :2].astype(np.float64)
    ref_lane = lane[:, :2].astype(np.float64)
    # t is percentage of (ref_lane segment distance^2)

    ref_lane_segments = np.array([np.array(x) for x in zip(ref_lane[:-1], ref_lane[1:])])
    segment_lengths = np.linalg.norm(ref_lane_segments[:, 1] - ref_lane_segments[:, 0], axis=-1)
    # Frenet s for being at the start of each segment
    segment_frenet_s = np.array([0, *np.cumsum(segment_lengths)[:-1]])
    segment_dists = np.linalg.norm(points[:, np.newaxis, np.newaxis, :] - ref_lane_segments, axis=-1).mean(axis=-1)
    # Shape is N x 2 x 2, i.e. N x segment
    closest_segment_idxs = segment_dists.argmin(axis=-1)
    closest_segments = ref_lane_segments[closest_segment_idxs]
    eps = 1e-8

    l2 = np.sum((closest_segments[:, 1]-closest_segments[:, 0])**2, axis=-1)
    l2[l2 < eps] = eps

    t = np.sum((points - closest_segments[:, 0]) * (closest_segments[:, 1] - closest_segments[:, 0]), axis=-1) / l2
    proj_points = closest_segments[:, 0] + t[:, np.newaxis] * (closest_segments[:, 1] - closest_segments[:, 0])
    lane_point_dists = np.linalg.norm(points[:, np.newaxis, :] - ref_lane, axis=-1)
    closest_point_idxs = lane_point_dists.argmin(axis=-1)
    closest_points = ref_lane[closest_point_idxs]
    # if inner_proj_err.any():
    # Cases: 
    # 1. new t spills out beyond edges 0 or 1 -> use border segments
    # 2. new t spills slightly into neighbor -> use them as a segment
    #     breakpoint()
    early_points = (t < 0) & (closest_segment_idxs == 0)
    late_points = (t > 1) & (closest_segment_idxs == len(ref_lane_segments) - 1)
    # Non-differentiable points
    inner_proj_err = (~early_points) & (~late_points) & ((t < 0) | (t > 1))

    proj_points[inner_proj_err] = closest_points[inner_proj_err]
    closest_left = np.array([lane_point_dists[i, idx] for i, idx in enumerate(closest_segment_idxs)])[inner_proj_err]
    closest_right = np.array([lane_point_dists[i, idx + 1] for i, idx in enumerate(closest_segment_idxs)])[inner_proj_err]
    new_t = np.argmin(np.stack([closest_left, closest_right], axis=-1), axis=-1).astype(np.float64)
    new_t[new_t == 1] = 1 - eps
    t[inner_proj_err] = new_t

    # See: https://stackoverflow.com/a/1560510/10101616

    AB = closest_segments[:, 1] - closest_segments[:, 0]
    AX = points - closest_segments[:, 0]
    indicators = AB[:, 0]*AX[:, 1] - AB[:, 1]*AX[:, 0]
    indicators[indicators < 0] = -1
    indicators[indicators >= 0] = 1
    frenet_d = indicators * np.linalg.norm(proj_points - points, axis=-1)
    frenet_s = segment_frenet_s[closest_segment_idxs] + t*(l2**0.5)
    ret = np.stack([frenet_s, frenet_d], axis=-1)
    # plt.plot(points[:, 0], points[:, 1], marker='o', label='Traj')
    # plt.plot(ref_lane[:, 0], ref_lane[:, 1], marker='o', label='Lane')
    # plt.plot(proj_points[:, 0], proj_points[:, 1], marker='.', label='Proj Points')
    return ret, inner_proj_err, closest_segments

def simple_frenet2cartesian(points, lane):
    points = points[:, :2].astype(np.float64)
    lane = lane[:, :2].astype(np.float64)

    ref_lane = lane
    ref_lane_segments = np.array([np.array(x) for x in zip(ref_lane[:-1], ref_lane[1:])])
    segment_lengths = np.linalg.norm(ref_lane_segments[:, 1] - ref_lane_segments[:, 0], axis=-1)
    # Frenet s for being at the start of each segment
    segment_frenet_s = np.array([0, *np.cumsum(segment_lengths)[:-1]])
    segment_idxs = np.array([0 if point[0] <= 0 else \
                            len(segment_frenet_s) - 1 if point[0] >= segment_frenet_s[-1] else \
                            np.argmax(segment_frenet_s > point[0]) - 1 for point in points])

    t = points[:, 0] - segment_frenet_s[segment_idxs]
    closest_segments = ref_lane_segments[segment_idxs]
    l2 = np.sum((closest_segments[:, 1]-closest_segments[:, 0])**2, axis=-1)

    eps = 1e-8
    l2[l2 < eps] = eps
    proj_points = closest_segments[:, 0] + (t/(l2**0.5))[:, np.newaxis] * (closest_segments[:, 1] - closest_segments[:, 0])
    segment_dirs = closest_segments[:, 1] - closest_segments[:, 0]

    segment_dirs[segment_dirs == 0] = 1e-100
    segment_slopes = segment_dirs[:, 1]/segment_dirs[:, 0]
    segment_perp = -1/segment_slopes
    perp_ints = proj_points[:, 1] - segment_perp*proj_points[:, 0]
    perp_vec = np.stack([proj_points[:, 0], proj_points[:, 1] - perp_ints], axis=-1)
    perp_vec = perp_vec / np.linalg.norm(perp_vec, axis=-1)[:, np.newaxis]
    d = points[:, 1]
    points_pos = proj_points + perp_vec*d[:, np.newaxis]
    points_neg = proj_points - perp_vec*d[:, np.newaxis]

    # See: https://stackoverflow.com/a/1560510/10101616
    AB = segment_dirs
    AX_pos = points_pos - closest_segments[:, 0]
    # AX_neg = points_neg - closest_segments[:, 0]
    indicators_pos = AB[:, 0]*AX_pos[:, 1] - AB[:, 1]*AX_pos[:, 0]
    indicators_pos[indicators_pos < 0] = -1
    indicators_pos[indicators_pos >= 0] = 1

    correct_signs = np.sign(points[:, 1])
    correct_signs[correct_signs == 0] = 1
    xy_out = np.stack([points_pos[i] if indicators_pos[i] == sign else points_neg[i] \
                for i, sign in enumerate(correct_signs)], axis=0)
    # plt.plot(xy_out[:, 0], xy_out[:, 1], marker='.', label='Recovered')
    return xy_out

def cartesian2frenet(points, lane, ignore_bounds=False):
    """  Returns 2 dimensional with (s, d), where:
         s = distance along entire lane,
         d = signed orthogonal distance,
         Also returns eps_mask for segments which are ok to numerically project (eps dist >= 1e-8, and 0 <= segment_t <= 1)
    """
    points = points[:, :2]
    lane = lane[:, :2]
    ref_lane = lane
    #ref_lane_meta = lanes_meta[int(lane_idx)]

    ref_lane_segments = np.array([np.array(x) for x in zip(ref_lane[:-1], ref_lane[1:])])
    segment_lengths = np.linalg.norm(ref_lane_segments[:, 1] - ref_lane_segments[:, 0], axis=-1)
    # Frenet s for being at the start of each segment
    segment_frenet_s = np.array([0, *np.cumsum(segment_lengths)[:-1]])
    segment_dists = np.linalg.norm(points[:, np.newaxis, np.newaxis, :] - ref_lane_segments, axis=-1).mean(axis=-1)
    # Shape is N x 2 x 2, i.e. N x segment
    closest_segment_idxs = segment_dists.argmin(axis=-1)
    closest_segments = ref_lane_segments[closest_segment_idxs]

    # Now actually do the projections
    # Following this: https://arxiv.org/pdf/2305.17965.pdf
    # and also this: https://stackoverflow.com/a/61343727/10101616
    l2 = np.sum((closest_segments[:, 1]-closest_segments[:, 0])**2, axis=-1)
    eps = 1e-8
    # simply use a mask
    if not ignore_bounds:
        eps_mask = (l2 > eps) & (
            (np.abs(closest_segments[:, 1] - closest_segments[:, 0])[:, 0] > eps) | \
            (np.abs(closest_segments[:, 1] - closest_segments[:, 0])[:, 1] > eps)
        )
    else:
        eps_mask = np.ones_like(l2).astype(bool)
        l2[l2 < eps] = eps
    
    if eps_mask.any():
        # t should be between (0, 1) in order to fall within the segment, but it's okay to be outside for now
        closest_segments = closest_segments[eps_mask]
        closest_segment_idxs = closest_segment_idxs[eps_mask]
        points = points[eps_mask]
        l2 = l2[eps_mask]

        # t should be between 0 and 1, and represent the proportion along the distance from segment[0] to segment[1]
        t = np.sum((points - closest_segments[:, 0]) * (closest_segments[:, 1] - closest_segments[:, 0]), axis=-1) / l2
        proj_points = closest_segments[:, 0] + t[:, np.newaxis] * (closest_segments[:, 1] - closest_segments[:, 0])

        # See: https://stackoverflow.com/a/1560510/10101616
        AB = closest_segments[:, 1] - closest_segments[:, 0]
        AX = points - closest_segments[:, 0]
        indicators = AB[:, 0]*AX[:, 1] - AB[:, 1]*AX[:, 0]
        indicators[indicators < 0] = -1
        indicators[indicators >= 0] = 1
        frenet_d = indicators * np.linalg.norm(proj_points - points, axis=-1)
        frenet_s = segment_frenet_s[closest_segment_idxs] + t*(l2**0.5)
        if not ignore_bounds:
            valid_t = (t >= 0) & (t <= 1)
        else:
            valid_t = np.ones_like(t).astype(bool)
        t = t[valid_t]
        eps_idxs = np.arange(len(points))
        val_idxs = eps_idxs[eps_mask][valid_t]
        eps_mask[:] = False
        eps_mask[val_idxs] = True
        ret = np.stack([frenet_s[valid_t], frenet_d[valid_t]], axis=-1)
    else:
        ret = np.zeros((0, 2), dtype=np.float64)
    return ret, eps_mask, t

def frenet2cartesian(points, lane):
    """  Returns 2 dimensional with (x, y) """
    points = points[:, :2]
    lane = lane[:, :2]
    if not len(points):
        return np.zeros((0, 2), dtype=points.dtype)

    ref_lane = lane
    #ref_lane_meta = lanes_meta[int(lane_idx)]
    ref_lane_segments = np.array([np.array(x) for x in zip(ref_lane[:-1], ref_lane[1:])])
    segment_lengths = np.linalg.norm(ref_lane_segments[:, 1] - ref_lane_segments[:, 0], axis=-1)
    # Frenet s for being at the start of each segment
    segment_frenet_s = np.array([0, *np.cumsum(segment_lengths)[:-1]])
    segment_idxs = np.array([0 if point[0] <= 0 else \
                            len(segment_frenet_s) - 1 if point[0] >= segment_frenet_s[-1] else \
                            np.argmax(segment_frenet_s > point[0]) - 1 for point in points])

    t = points[:, 0] - segment_frenet_s[segment_idxs]
    closest_segments = ref_lane_segments[segment_idxs]
    # Now actually do the projections
    # Following this: https://arxiv.org/pdf/2305.17965.pdf
    # and also this: https://stackoverflow.com/a/61343727/10101616
    l2 = np.sum((closest_segments[:, 1]-closest_segments[:, 0])**2, axis=-1)
    eps = 1e-8
    eps_mask = (l2 > eps) & (
        (np.abs(closest_segments[:, 1] - closest_segments[:, 0])[:, 0] > eps) | \
        (np.abs(closest_segments[:, 1] - closest_segments[:, 0])[:, 1] > eps)
    )
    xy_out = np.zeros((len(points), 2), dtype=np.float64)
    xy_out[~eps_mask] = np.nan
    if eps_mask.any():
        # t should be between (0, 1) in order to fall within the segment, but it's okay to be outside for now
        closest_segments = closest_segments[eps_mask]
        # closest_segment_idxs = segment_idxs[eps_mask]
        points = points[eps_mask]
        l2 = l2[eps_mask]
        t = t[eps_mask]
        proj_points = closest_segments[:, 0] + (t/(l2**0.5))[:, np.newaxis] * (closest_segments[:, 1] - closest_segments[:, 0])
        segment_dirs = closest_segments[:, 1] - closest_segments[:, 0]
        # Explanation:
        # 1. Calculate perpendicular line to proj_point along segment, keep in 2D
        # 2. Keep the z value of proj_point unmodified
        # 3. Produce both possible points in either direction along perpendicular line
        # 4. Calculate indicator for both; choose the one with the correct sign
        
        # Avoid divide by zero
        segment_dirs[segment_dirs == 0] = 1e-100
        segment_slopes = segment_dirs[:, 1]/segment_dirs[:, 0]
        segment_perp = -1/segment_slopes
        perp_ints = proj_points[:, 1] - segment_perp*proj_points[:, 0]
        perp_vec = np.stack([proj_points[:, 0], proj_points[:, 1] - perp_ints], axis=-1)
        perp_vec = perp_vec / np.linalg.norm(perp_vec, axis=-1)[:, np.newaxis]
        d = points[:, 1]
        points_pos = proj_points + perp_vec*d[:, np.newaxis]
        points_neg = proj_points - perp_vec*d[:, np.newaxis]

        # See: https://stackoverflow.com/a/1560510/10101616
        AB = segment_dirs
        AX_pos = points_pos - closest_segments[:, 0]
        # AX_neg = points_neg - closest_segments[:, 0]
        indicators_pos = AB[:, 0]*AX_pos[:, 1] - AB[:, 1]*AX_pos[:, 0]
        indicators_pos[indicators_pos < 0] = -1
        indicators_pos[indicators_pos >= 0] = 1

        correct_signs = np.sign(points[:, 1])
        correct_signs[correct_signs == 0] = 1
        xy_out[eps_mask] = np.stack([points_pos[i] if indicators_pos[i] == sign else points_neg[i] \
                                      for i, sign in enumerate(correct_signs)], axis=0)
    return 

def do_interp(traj_segment, traj_vel, lane, linear_only=False):
    if lane is not None and not linear_only:
        # First, interpolate within (s, d) coordinates
        tmp = traj_segment[:, 3:6]
        tmp[tmp == np.inf] = np.nan
        sd_interp = pd.DataFrame(tmp).interpolate(
            limit_direction='both', limit_area='inside').to_numpy()
        to_fill = np.isnan(sd_interp[:, 0])
        assert len(np.unique(sd_interp[:, -1][~to_fill])) <= 1, 'Multiple lanes in a single segment'
        # Now, extrapolate but with a fixed d value
        to_fill = np.isnan(sd_interp[:, 0])
        if to_fill[0] or to_fill[-1]:
            # Also need to special case for when length is 1:
            #     use traj_vel to get next cartesian coordinate, figure out direction on lane
            if np.sum(~to_fill) > 1:
                start_speed = sd_interp[~to_fill][1] - sd_interp[~to_fill][0]
                end_speed = sd_interp[~to_fill][-1] - sd_interp[~to_fill][-2]
            elif np.sum(~to_fill) == 1:
                point_xyz = traj_segment[~to_fill, :3]
                point_xyz[:, :2] += traj_vel[~to_fill]/10.0
                assert not np.isnan(point_xyz).any(), 'NaN snuck in'
                sd_next_point = cartesian2frenet(point_xyz, lane)
                if len(sd_next_point[0]):
                    start_speed = (sd_next_point[0][:, :2] - sd_interp[~to_fill][:, :2]).squeeze()
                    end_speed = start_speed
                else:
                    start_speed, end_speed = None, None
            else:
                start_speed, end_speed = None, None

            # Constant velocity propagation forward
            if end_speed is not None and to_fill[-1]:
                idxs = np.arange(len(traj_segment))
                last_idx = idxs[~to_fill][-1]
                sd_interp[last_idx:, 1:] = sd_interp[last_idx, 1:]
                #sd_interp[:first_idx+1, 1] = [
                # sd_interp[first_idx, 1] - start_speed[1]*(first_idx - i) for i in range(first_idx+1)]
                sd_interp[last_idx:, 0] = [
                    sd_interp[last_idx, 0] + end_speed[0]*i for i in range(len(traj_segment) - last_idx)]
                
            if start_speed is not None and to_fill[0]:
                idxs = np.arange(len(traj_segment))
                first_idx = idxs[~to_fill][0]
                sd_interp[:first_idx+1, 1:] = sd_interp[first_idx, 1:]
                #sd_interp[:first_idx+1, 1] = [
                # sd_interp[first_idx, 1] - start_speed[1]*(first_idx - i) for i in range(first_idx+1)]
                sd_interp[:first_idx+1, 0] = [
                    sd_interp[first_idx, 0] - start_speed[0]*(first_idx - i) for i in range(first_idx+1)]
        
        # At this point, we've interpolated/extrapolated in frenet as appropraite into sd_interp
        # Now, we just need to convert back to cartesian
        interp_mask = traj_segment[:, -1] == 0
        to_project = sd_interp[interp_mask]
        proj_mask = ~np.isnan(to_project[:, 0])
        to_project = to_project[proj_mask]

        projected_xy = frenet2cartesian(to_project, lane)
        proj_idxs = np.arange(len(traj_segment))[interp_mask][proj_mask]
        traj_segment[proj_idxs, :2] = projected_xy
        traj_segment[proj_idxs, 3:6] = sd_interp[proj_idxs]
            
    # If we still need to do linear interpolation... i.e. from eps_mask not being all() or lane missing
    if np.isnan(traj_segment[:, :3]).any():
        xyz_interp = pd.DataFrame(traj_segment[:, :3]).interpolate(
            limit_direction='both', limit_area='inside').to_numpy()
        to_fill_xy = np.isnan(xyz_interp[:, 0])
        if (~to_fill_xy).sum() == 0:
            return traj_segment
        if to_fill_xy[-1]:
            idxs = np.arange(len(traj_segment))
            last_idx = idxs[~to_fill_xy][-1]
            end_speed = traj_vel[~to_fill_xy][-1]/10.0
            xyz_interp[last_idx:, :2] = [xyz_interp[last_idx, :2] + end_speed*i for i in range(len(traj_segment) - last_idx)]
        if to_fill_xy[0]:
            idxs = np.arange(len(traj_segment))
            first_idx = idxs[~to_fill_xy][0]
            start_speed = traj_vel[~to_fill_xy][0]/10.0
            xyz_interp[:first_idx+1, :2] = [xyz_interp[first_idx, :2] - start_speed*(first_idx - i) for i in range(first_idx+1)]

        # Handle z specially when at ends or beginnings
        to_fill_z = np.isnan(xyz_interp[:, -1])
        if (~to_fill_z).sum() == 0:
            return traj_segment
        if to_fill_z.any():
            idxs = np.arange(len(traj_segment))
            good_idxs = idxs[~to_fill_z]
            if to_fill_z[-1]:
                last_idx = good_idxs[-1]
                xyz_interp[last_idx:, 2] = xyz_interp[last_idx, 2]
            if to_fill_z[0]:
                first_idx = good_idxs[0]
                xyz_interp[:first_idx+1, 2] = xyz_interp[first_idx, 2]
            # Turns out, extrapolating z linearly is odd
            # if to_fill_z[-1]:
            #     last_idx = good_idxs[-1]
            #     end_speed = 0 if len(good_idxs) == 1 else traj_segment[good_idxs[-1], 2] - traj_segment[good_idxs[-2], 2]
            #     xyz_interp[last_idx:, 2] = [xyz_interp[last_idx, 2] + end_speed*i for i in range(len(traj_segment) - last_idx)]
            # if to_fill_z[0]:
            #     first_idx = good_idxs[0]
            #     start_speed = 0 if len(good_idxs) == 1 else traj_segment[good_idxs[1], 2] - traj_segment[good_idxs[0], 2]
            #     xyz_interp[:first_idx+1, 2] = [xyz_interp[first_idx, 2] - start_speed*(first_idx - i) for i in range(first_idx+1)]
        traj_segment[:, :3] = xyz_interp
    return traj_segment

def process_traj(
    traj, 
    traj_vel, 
    mask, 
    lane_info, 
    lanes, 
    lanes_meta, 
    closest_meta, 
    linear_only,         
    frenet_t_lower = 0, 
    frenet_t_upper = 1, 
    frenet_d_max = 2.5
):
    # i.e. No valid points in traj, typically only happens with hist_only flag Valid lane sequence 
    # not found, basically only when timeout is hit
    if lane_info is None or lane_info['sequence'] is None:
        # 7d: xyz, sd, lane_idx, valid (interpolation needed)
        ret = np.zeros((len(traj), 7), dtype=np.float64) * np.nan
        possible_idxs = np.arange(len(traj))
        real_idxs = possible_idxs[mask]
        ret[real_idxs, :3] = traj[real_idxs]
        ret[:, -2] = np.inf
        ret[:, -1] = 0
        ret[real_idxs, -1] = True
        ret = do_interp(ret, traj_vel, lane=None, linear_only=True)
        return ret

    lane_sequence = lane_info['sequence']

    # How to extrapolate in frenet, when not given a speed? Maybe look at simple linear interpolation, to start
    # What if actually stopped? i.e. in parking lot, side of street parked, traffic light, etc.
    # if len(traj[mask]) == 1 and traj_vel[mask].sum() != 0:
    #     pass

    lanes = [x.astype(np.float64) for x in lanes]
    possible_idxs = np.arange(len(mask))

    # Process:
    # 1. If lane_idx is valid, do interpolation if needed in Frenet space; otherwise do linear interp
    #    a. Interpolate between points
    #    b. If extrapolation needed at beginning or end of overall sequence, do it
    #    c. If next segement exists and its start isn't contiguous, interpolate to it in current lane
    # 2. Get the closest lanes for the interpolated points
    # 3. Get frenet of *all* points, for later usage of checking for possible conflicts, etc.
    segments_out = []
    for segment_idx, lane_segment in enumerate(lane_sequence):
        real_idxs = possible_idxs[mask][lane_segment.begin:lane_segment.end]

        interpolate_inner = len(real_idxs) != real_idxs[-1] - real_idxs[0] + 1
        next_index = possible_idxs[mask][lane_sequence[segment_idx + 1].begin] if segment_idx != len(lane_sequence) - 1 else None
        interpolate_next = segment_idx != len(lane_sequence) - 1 and real_idxs[-1] + 1 != next_index
        extrapolate_begin = segment_idx == 0 and real_idxs[0] != 0
        extrapolate_end = segment_idx == len(lane_sequence) - 1 and real_idxs[-1] != len(mask) - 1
        fill_required = np.any([interpolate_inner, interpolate_next, extrapolate_begin, extrapolate_end])

        points = traj[real_idxs]
        # Need to interpolate to next_index and such, so project that into frenet if required
        if interpolate_next:
            points_next = np.concatenate([points, traj[next_index][np.newaxis, :]], axis=0)
        else:
            points_next = points

        lane_idx = lane_segment.data[0]
        if lane_idx != np.inf:
            cur_frenet, eps_mask, frenet_t = cartesian2frenet(points_next, lanes[int(lane_idx)])
            # orig = points_next[eps_mask][:, :2]
            # back = frenet2cartesian(cur_frenet, lanes[int(lane_idx)])
            # if np.linalg.norm((back - orig) > 1e-8):
            #     breakpoint()
        else:
            cur_frenet, eps_mask, frenet_t = None, None, None

        if fill_required:
            # 7d: xyz, sd, lane_idx, valid (interpolation needed)
            start_idx = 0 if extrapolate_begin else real_idxs[0]
            end_idx = len(traj) if extrapolate_end else next_index + 1 if interpolate_next else real_idxs[-1] + 1
            new_traj_segment = np.zeros((end_idx - start_idx, 7), dtype=np.float64) * np.nan
            new_traj_segment[:, -1] = 0
            new_traj_segment[:, -2] = np.inf
            if extrapolate_begin:
                new_traj_segment[real_idxs, :3] = traj[real_idxs]
                new_traj_segment[real_idxs, -1] = True
                if interpolate_next:
                    new_traj_segment[next_index, :3] = points_next[-1]
                    new_traj_segment[next_index, -1] = True
            else:
                new_traj_segment[real_idxs - real_idxs[0], :3] = traj[real_idxs]
                new_traj_segment[real_idxs - real_idxs[0], -1] = True
                if interpolate_next:
                    new_traj_segment[next_index - real_idxs[0], :3] = points_next[-1]
                    new_traj_segment[next_index - real_idxs[0], -1] = True

            if cur_frenet is not None:
                if interpolate_next:
                    all_idxs = np.concatenate([real_idxs, [next_index]], axis=-1)
                else:
                    all_idxs = real_idxs
                # frenet_idxs = all_idxs[eps_mask]

                # Allow for one extra segment distance in each direction at most
                t_good = (frenet_t >= frenet_t_lower) & (frenet_t <= frenet_t_upper)
                # Harsher lane threshold of 1m at most
                d_good = np.abs(cur_frenet[:, 1]) <= frenet_d_max

                good_mask = (t_good) & (d_good)
                good_idxs = all_idxs[eps_mask][good_mask]
                if extrapolate_begin:
                    new_traj_segment[good_idxs, 3:5] = cur_frenet[good_mask]
                    new_traj_segment[good_idxs, -2] = lane_idx
                else:
                    new_traj_segment[good_idxs - real_idxs[0], 3:5] = cur_frenet[good_mask]
                    new_traj_segment[good_idxs - real_idxs[0], -2] = lane_idx
            new_traj_segment = do_interp(
                new_traj_segment, 
                traj_vel[start_idx:end_idx], 
                lanes[int(lane_idx)] if lane_idx != np.inf else None, 
                linear_only)
        else:
            # 7d: xyz, sd, lane_idx, valid (interpolation needed)
            new_traj_segment = np.zeros((len(real_idxs), 7), dtype=np.float64) * np.nan
            new_traj_segment[:, -1] = 0
            new_traj_segment[:, -2] = np.inf
            new_traj_segment[real_idxs - real_idxs[0], :3] = traj[real_idxs]
            new_traj_segment[real_idxs - real_idxs[0], -1] = True
            if cur_frenet is not None:
                # frenet_idxs = real_idxs[eps_mask]

                # Allow for one extra segment distance in each direction at most
                t_good = (frenet_t >= frenet_t_lower) & (frenet_t <= frenet_t_upper)
                # Harsher lane threshold of 1m at most
                d_good = np.abs(cur_frenet[:, 1]) <= frenet_d_max
                good_mask = (t_good) & (d_good)
                good_idxs = real_idxs[eps_mask][good_mask]

                new_traj_segment[good_idxs - real_idxs[0], 3:5] = cur_frenet[good_mask]
                new_traj_segment[good_idxs - real_idxs[0], -2] = lane_idx

        segment_out = new_traj_segment[:] if not interpolate_next else new_traj_segment[:-1]
        segments_out.append(segment_out)

    full_out = np.concatenate(segments_out, axis=0)
    assert len(full_out) == len(traj), 'Concatenation of segments failed'
    return full_out