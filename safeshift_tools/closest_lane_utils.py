import heapq
import numpy as np
import pandas as pd
import time 

from intervaltree import IntervalTree, Interval
from skspatial.objects import Vector

def do_resample(lane, resample_level=1):
    df = pd.DataFrame(lane)
    df.index = [pd.to_datetime(1e9 * x * resample_level) for x in df.index]
    df = df.resample('1s').interpolate()
    return df.to_numpy()

def build_lane_graph(lanes):
    """ Map lane index to lane ID and build a connected graph using entry and exit lanes. """
    id_to_idx = {}
    for l, lane in enumerate(lanes):
        id_to_idx[lane['id']] = l

    graph = {}
    for l, lane in enumerate(lanes):
        # Should always be able to connect back to self
        connected_lanes = [lane['id']] + lane['entry_lanes'] + lane['exit_lanes']
        # Make unique just in case
        connected_lanes = list(set(connected_lanes))
        graph[id_to_idx[lane['id']]] = [id_to_idx[c] for c in connected_lanes]
    return graph

def build_lane_sequences(
    closest_lanes_per_timestep, 
    lane_graph, 
    lane_segments,
    dist_threshold, 
    prob_threshold, 
    angle_thresh,
    traj, 
    traj_vel, 
    traj_heading, 
    track_infos, 
    agent_n, 
    timeout
):
    start_time = time.time()
    T, K, D = closest_lanes_per_timestep.shape
    obj_type = track_infos['object_type'][agent_n]
    #follow_rules = obj_type in ['TYPE_VEHICLE']
    #print(f'\t{agent_n}, threshold: {dist_threshold}')

    lane_probs = closest_lanes_per_timestep[..., 0]
    lane_probs = np.maximum(0, 1 - lane_probs/dist_threshold)
    # Assign list of valid regions
    possible_lanes = []
    # in 2d at least
    def get_angle(current_pos, current_heading, current_vel, lane: int):
        lane_segment = lane_segments[lane]
        segment_dists = np.linalg.norm(current_pos[np.newaxis, np.newaxis, :] - lane_segment, axis=-1).mean(axis=-1)
        # i.e. only one point in lane, unable to determine direction or projection things
        if not len(segment_dists):
            return 0, False
        closest_segment = lane_segment[segment_dists.argmin(axis=-1)]
        segment_dir = closest_segment[1] - closest_segment[0]
        current_heading_vec = Vector(np.array([np.cos(current_heading), np.sin(current_heading)]).squeeze())
        angle_diff = np.rad2deg(current_heading_vec.angle_signed(segment_dir[:2]))
        possible_trans = np.abs(angle_diff) < angle_thresh or np.abs(angle_diff) > (180 - angle_thresh)
        # TODO: assess if this is an okay transition if at slow speeds
        return angle_diff, (possible_trans or np.linalg.norm(current_vel) < 1.0)
        # return angle_diff, possible_trans

    for t in range(T):
        dists = {}
        for lane_info in closest_lanes_per_timestep[t, lane_probs[t] > prob_threshold]:
            angle_diff, possible_trans = get_angle(traj[t], traj_heading[t], traj_vel[t], int(lane_info[-1]))
            #if possible_trans or not follow_rules:
            if possible_trans:
                dists[lane_info[-1]] = lane_info
            # dists[lane_info[-1]] = lane_info
        possible_lanes.append(dists)

    # Build interval tree
    active_intervals = {}
    interval_list = []
    for t in range(T):
        # Check for interval ends first
        to_remove = []
        for active, (start, lane_id, lane_infos) in active_intervals.items():
            if active in possible_lanes[t]:
                lane_infos.append(possible_lanes[t][active])
            if active not in possible_lanes[t]:
                end = t
                # half-open interval, start inclusive, end exclusive
                interval_list.append([start, end, (lane_id, np.stack(lane_infos))])
                to_remove.append(active)
        for x in to_remove:
            active_intervals.pop(x)
        
        for lane_id, lane_info in possible_lanes[t].items():
            if lane_id not in active_intervals:
                active_intervals[lane_id] = (t, lane_id, [lane_info])
    for active, (start, lane_id, lane_infos) in active_intervals.items():
        end = T
        interval_list.append([start, end, (lane_id, np.stack(lane_infos))])
    tree = IntervalTree()
    for interval in interval_list:
        start, end = interval[0], interval[1]
        #sub_intervals = np.array_split(np.arange(start, end), factor)
        sub_intervals = np.array_split(np.arange(start, end), 1)
        for sub_interval in sub_intervals:
            if not len(sub_interval):
                break
            sub_start, sub_end = sub_interval[0], sub_interval[-1] + 1
            sub_data = (interval[2][0], interval[2][1][sub_start-start:sub_end-start])
            tree[sub_start:sub_end] = sub_data
    
    # Perform DFS
    # greedy i.e. so that the first one found is heuristically the "best" one
    # PQ = PriorityQueue()
    PQ = []
    # Priority = dist so far + dist if you were to add current interval
    def enqueue_item(interval, cur_seq):
        tot_dist = 0
        tot_n = 0
        if len(cur_seq):
            tot_n += sum([x.end - x.begin for x in cur_seq])
            tot_dist += sum([x.data[1][:, 0].sum() for x in cur_seq])
        tot_n += (interval.end - interval.begin)
        tot_dist += interval.data[1][:, 0].sum()
        # Add time to de-conflict order
        heapq.heappush(PQ, (tot_dist, time.time(), (interval, cur_seq)))
    def enqueue_skip(begin, cur_seq):
        new_data = np.zeros((1, 2), dtype=np.float64)
        new_data[0, 0] = np.inf
        new_data[0, 1] = 100000
        new_data = (np.inf, new_data)
        enqueue_item(Interval(begin, begin+1, new_data), cur_seq.copy())

    for interval in tree[0]:
        enqueue_item(interval, [])
    if not len(PQ):
        enqueue_skip(0, [])

    #queue = deque([(interval, []) for interval in tree[0]])
    valid_seqs = []
    valid_dists = []
    longest_seq_t = -1
    longest_seqs = None
    n_expanded = 0
    #while not PQ.empty():
    while PQ:
        tot_dist, _, (interval, cur_seq) = heapq.heappop(PQ)
        n_expanded += 1
        if timeout > 0 and time.time() - start_time > timeout:
            break
        end = interval.end
        cur_seq.append(interval)
        # Greedy approach, only compute one valid seq? Or up to a certain number
        if end == T:
            valid_seqs.append(cur_seq)
            valid_dists.append(tot_dist)
            break
        if end > longest_seq_t:
            longest_seqs = [cur_seq]
            longest_seq_t = end
        elif end == longest_seq_t:
            longest_seqs.append(cur_seq)
        next_intervals = tree[end]
        lane_id = interval.data[0]
        found_continuation = False
        for possible_neighbor in next_intervals:
            found_continuation = True
            if lane_id == np.inf or possible_neighbor.data[0] not in lane_graph[lane_id]:
                current_pos = traj[end]
                current_vel = traj_vel[end]
                current_heading = traj_heading[end]
                angle_diff, possible_trans = get_angle(current_pos, current_heading, 
                                                       current_vel, int(possible_neighbor.data[0]))
            else:
                possible_trans = True

            if possible_trans:
                # TODO: enqueue_skip only if no possible transitions? i.e. put found_continuation = True here?
                new_data = (possible_neighbor.data[0], possible_neighbor.data[1][end - possible_neighbor.begin:])
                enqueue_item(Interval(end, possible_neighbor.end, new_data), cur_seq.copy())
        if not found_continuation:
            enqueue_skip(end, cur_seq.copy())
    
    # only possible fail is timeout?
    # if len(valid_seqs):
    #     status = f'VALID_{dist_threshold}'
    # elif closest_lanes_per_timestep[:, 0, 0].max() > dist_threshold/2:
    #     status = f'INVALID_{dist_threshold}_mindist'
    # else:
    #     status = f'INVALID_{dist_threshold}_meat'

    # Build return data
    ret = []
    unique_paths = set()
    for valid_seq in valid_seqs:
        seq_dists = []
        lane_path = []
        for interval in valid_seq:
            assert interval.end - interval.begin == len(interval.data[1]), 'Mismatch in interval data'
            seq_dists.extend(interval.data[1][:, 0])
            lane_path.extend([interval.data[0]] * (interval.end - interval.begin))
        if tuple(lane_path) not in unique_paths:
            unique_paths.add(tuple(lane_path))
            ret.append({'sequence': valid_seq, 'dists': np.array(seq_dists), 'path': lane_path})
    n_valid = 0
    n_tot = T
    if len(ret):
        lanes_used = set(ret[0]['path'])
        if np.inf not in lanes_used:
            status = f'VALID_{dist_threshold}_full'
        elif len(lanes_used) != 1:
            status = f'VALID_{dist_threshold}_partial'
        else:
            status = f'INVALID_{dist_threshold}'
        n_valid = (np.array(ret[0]['path']) != np.inf).sum()
    else:
            status = f'INVALID_{dist_threshold}'
    if 'full' not in status:
        status += '_meat' if closest_lanes_per_timestep[:, 0, 0].max() < dist_threshold/2 else '_mindist'

    return ret, status, n_expanded, time.time() - start_time, tree, n_valid, n_tot

def compute_k_closest_lanes(trajectory, mask, lanes, K = 16, resample_level = 1, threshold=10):
    """
    Inputs:
    -------
        trajectory[np.array(T, D=xyz)]: agent's trajectory.
        mask[np.array(T)]: valid data points mask
        lanes[list(np.array(Nl, D=xy))]: list of lanes in scenario.
        K[int]: K-closest lanes to keep 
    Outputs
    ------- 
        D_full[np.array(N, T, 3)]: distance matrix containing closest distance, lane index of 
            closest point, lane index in list.
        D_full[np.array(N, T, 3)]: as above but for history portion.
        
    """
    trajectory = trajectory.T[:, None, :] # (T, 1, dim) 
    T, _, _ = trajectory.shape
    N = len(lanes)

    # [distance, lane_pos_idx, proj_distance, proj_idx, proj_x, proj_y, proj_z, lane_idx]
    D_full = np.inf * np.ones(shape=(N, T, 6), dtype=np.float64) 
    # Closest distance between lane and trajectory
    for n in range(N):
        # lane = lanes[n][::resample_level]
        # hq_lane = lanes[n]
        lane = lanes[n]

        # (T, 1, dim) - (Nl, dim) --> (T, Nl, dim) --> (T, Nl)
        # dists[n] = np.linalg.norm(trajectory - lane, axis=2).min(axis=1)
        dn = np.linalg.norm(trajectory - lane, axis=2).astype(np.float64)
        dn[~mask] = np.inf

        closest_points = dn[mask].argmin(axis=1)
        closest_dists = dn[mask].min(axis=1)
        closest_pos = np.stack([lane[i] for i in closest_points]).astype(np.float64)
        if closest_dists.min() > threshold or len(trajectory[mask]) == 0:
            D_full[n, mask, 0] = closest_dists # dist value
            D_full[n, mask, 1] = closest_points # idx value
            D_full[n, mask, 2] = closest_pos[:, 0] # x val
            D_full[n, mask, 3] = closest_pos[:, 1] # y val
            D_full[n, mask, 4] = closest_pos[:, 2] # z val
            continue

        segment_points = []
        segment_ids = []
        for i, point_idx in enumerate(closest_points):
            if point_idx == 0:
                segment = (0, 1)
            elif point_idx == len(lane) - 1:
                segment = (len(lane) - 2, len(lane) - 1)
            elif dn[mask][i, point_idx - 1] < dn[mask][i, point_idx + 1]:
                segment = (point_idx - 1, point_idx)
            else:
                segment = (point_idx, point_idx + 1)
            if len(lane) == 1:
                segment = (0, 0)
            left_bound = lane[segment[0]]
            right_bound = lane[segment[1]]
            segment_points.append(np.stack([left_bound, right_bound]))
            segment_ids.append(np.array([segment[0], segment[1]]))
        segment_points = np.array(segment_points).astype(np.float64)
        segment_ids = np.array(segment_ids)

        # Following this: https://arxiv.org/pdf/2305.17965.pdf
        # and also this: https://stackoverflow.com/a/61343727/10101616
        l2 = np.sum((segment_points[:, 1]-segment_points[:, 0])**2, axis=-1)
        eps = 1e-8
        eps_mask = l2 > eps
        # if (l2 < eps).any():
        #     import pdb; pdb.set_trace()
        if eps_mask.any():
            traj_points = trajectory[mask][:, 0][eps_mask]
            # t should be between (0, 1) in order to fall within the segment, but it's okay to be outside for now
            t = np.sum((traj_points - segment_points[:, 0]) * (segment_points[:, 1] - segment_points[:, 0]), axis=-1) / l2
            proj_points = segment_points[:, 0] + t[:, np.newaxis] * (segment_points[:, 1] - segment_points[:, 0])
            along_segments = (t >= 0) & (t <= 1)

            new_dists = np.linalg.norm(proj_points - traj_points, axis=-1)
            new_idxs = t + segment_ids[:, 0][eps_mask]
            
            closest_points = closest_points.astype(np.float64)
            if along_segments.any():
                to_idx = np.arange(len(eps_mask))[eps_mask][along_segments]
                closest_dists[to_idx] = new_dists[along_segments]
                closest_points[to_idx] = new_idxs[along_segments]
                closest_pos[to_idx, 0] = proj_points[along_segments][:, 0]
                closest_pos[to_idx, 1] = proj_points[along_segments][:, 1]
                closest_pos[to_idx, 2] = proj_points[along_segments][:, 2]
        
        D_full[n, mask, 0] = closest_dists # dist value
        D_full[n, mask, 1] = closest_points # idx value
        D_full[n, mask, 2] = closest_pos[:, 0] # x val
        D_full[n, mask, 3] = closest_pos[:, 1] # y val
        D_full[n, mask, 4] = closest_pos[:, 2] # z val

    # K closest lanes --> (K, T)
    full_k_lanes = D_full[:, :, 0].argsort(axis=0)[:K]
    D_k = np.inf * np.ones(shape=(T, full_k_lanes.shape[0], D_full.shape[-1]))
    for t in range(T):
        k_lanes = full_k_lanes[:, t]
        D_full[k_lanes, t, -1] = k_lanes
        D_k[t] = D_full[k_lanes, t]
        
    D_k[~mask] = np.inf
    return D_k