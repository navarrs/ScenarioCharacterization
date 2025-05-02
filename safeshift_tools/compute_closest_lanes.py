import numpy as np
import os
import pandas as pd
import pickle 
import time 

import utils.common as C

from tqdm import tqdm

from data_tools.visualization import plot_static_map_infos
from safeshift_tools.closest_lane_utils import (
    build_lane_sequences, compute_k_closest_lanes, build_lane_graph)

def process_file(
    path: str,  
    dist_threshold: int = 10, 
    prob_threshold: float = 0.5, 
    angle_threshold: int = 20, 
    thresh_iters: int = 1, 
    hist_only: bool = False, 
    timeout: bool = -1,
) -> tuple[list, list]:
    """ Process a single scenario file to find the closest lanes to the trajectories of agents in 
    the scenario.
    
    Args:
        path (str): Path to the scenario file.
        dist_threshold (float): Distance threshold for lane sequence building.
        prob_threshold (float): Probability threshold for lane sequence building.
        angle_threshold (float): Angle threshold for lane sequence building.
        thresh_iters (int): Number of iterations for increasing distance thresholds.
        hist_only (bool): If True, only consider historical trajectories.
        timeout (float): Timeout for lane sequence building. If -1, no timeout is applied.
    Returns:   
        meta_info (list): List of dictionaries containing metadata about the lane sequences.
        best_seqs (list): List of dictionaries containing the best lane sequences for each agent.
    """
    # Load the scenario 
    with open(path, 'rb') as f:
        scenario = pickle.load(f)
    
    # Trajectory data:
    #    center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
    track_infos = scenario['track_infos']
    objects_type = track_infos['object_type']
    object_ids = track_infos['object_id']

    # Map infos:
    #   lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, all_polylines
    static_map_infos = scenario['map_infos']
    dynamic_map_infos = scenario['dynamic_map_infos']
    # polylines = static_map_infos['all_polylines'][:, :3][:, None, :]

    # road_graph, lanes = static_map_infos['all_polylines'][:, :dim], []
    # for lane in static_map_infos['lane']:
    #     start_idx, end_idx = lane['polyline_index']
    #     polyline_pos = road_graph[start_idx:end_idx, :dim]
    #     lanes.append(polyline_pos)
    static_map_pos = plot_static_map_infos(static_map_infos, ax=None, dim=3)
    lanes = static_map_pos['lane']
    lane_graph = build_lane_graph(static_map_infos['lane'])
    lane_segments = [np.stack([lane[:-1, :], lane[1:, :]], axis=1) for lane in lanes]
    
    last_t = C.LAST_TIMESTEP if not hist_only else C.HIST_TIMESTEP
    # Trajectories --> (num_agents, time_steps, 9)
    trajectories = track_infos['trajs'][:, :last_t, :-1]
    # Mask         --> (num_agents, time_steps, 1)
    valid_masks = track_infos['trajs'][:, :last_t, -1] > 0

    num_agents, time_steps, dim = trajectories.shape
    
    meta_info, best_seqs = [], []
    for n in range(num_agents):
        mask = valid_masks[n]
        if not np.any(mask):
            continue
        
        trajectory = trajectories[n, :, C.POS_XYZ_IDX]
        avg_speed = np.linalg.norm(trajectories[n, :, C.VEL_XY_IDX].T[mask], axis=-1).mean()
        # Avg speed < 1.0 m/s; where linear interpolation shouldn't matter that much compared to Frenet
        stationary = (avg_speed < C.STATIONARY_SPEED)

        trajectory_closest_lanes = compute_k_closest_lanes(trajectory, mask, lanes) 
        trajectory_closest_lanes = trajectory_closest_lanes[:, :, [0, -1]]
        
        thresholds = [dist_threshold*(2**i) for i in range(thresh_iters)]
        for threshold in thresholds:
            lane_sequences, status, n_expanded, tot_time, tree, n_valid, n_tot = build_lane_sequences(
                trajectory_closest_lanes[mask], 
                lane_graph, 
                lane_segments, 
                threshold, 
                prob_threshold, 
                angle_threshold, 
                trajectories[n, :, C.POS_XYZ_IDX].T[mask], 
                trajectories[n, :, C.VEL_XY_IDX].T[mask], 
                trajectories[n, :, C.HEADING_IDX].T[mask],
                track_infos, 
                n, 
                timeout)
            if len(lane_sequences) and n_valid == n_tot:
                break

        cur_info = {
            'obj_type': objects_type[n],
            'agent_n': n,
            'stationary': stationary,
            'scenario_id': scenario['scenario_id'],
            'object_id': object_ids[n],
            'valid': 'INVALID' not in status,
            'full': 'full' in status,
            'meat_threshold': threshold,
            'min_dist_okay': 'mindist' not in status,
            'avg_speed': avg_speed,
            'n_expanded': n_expanded,
            'tot_time': tot_time,
            'timeout': timeout > 0 and tot_time > timeout,
            'n_valid': n_valid,
            'n_tot': n_tot
        }
        meta_info.append(cur_info)

        if len(lane_sequences):
            seq_dists = [x['dists'].mean() for x in lane_sequences]
            best_seq_idx = np.argmin(seq_dists)
            best_seq = lane_sequences[best_seq_idx]
            best_seq['closest_lanes'] = trajectory_closest_lanes[mask]
            best_seqs.append(best_seq)
        else:
            best_seq = {
                'sequence': None, 
                'dists': None, 
                'path': None, 
                'closest_lanes': trajectory_closest_lanes[mask]
            }
            best_seqs.append(best_seq)
    return meta_info, best_seqs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Find the closest lanes to a given point.")
    parser.add_argument(
        '--split', type=str, default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--dist_threshold', type=float, default=5)
    parser.add_argument('--angle_threshold', type=float, default=45)
    parser.add_argument('--thresh_iters', type=int, default=1)
    parser.add_argument('--hist_only', action='store_true')
    parser.add_argument('--prob_threshold', type=float, default=0.5)
    parser.add_argument('--timeout', type=float, default=5.0)
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    base = C.load_base(
        args.split, args.num_shards, args.shard_idx, args.num_scenarios, args.hist_only)
    metas, inputs = base['metas'], base['inputs']

    VISUAL_OUT_SUBDIR = os.path.join(C.VISUAL_OUT_DIR, args.split, 'closest_lanes')
    os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)
    CACHE_SUBDIR = os.path.join(C.CACHE_DIR, args.split, 'closest_lanes')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)
    
    msg = f'Processing {args.split} split scenarios...'
    start = time.time()
    if args.parallel:
        from joblib import Parallel, delayed    
        all_outs = Parallel(n_jobs=args.nproc, batch_size=args.batch_size)(
            delayed(process_file)(
                path, 
                dist_threshold=args.dist_threshold, 
                prob_threshold=args.prob_threshold,
                angle_threshold=args.angle_threshold, 
                thresh_iters = args.thresh_iters, 
                hist_only=args.hist_only, 
                timeout = args.timeout)
            for i, ((s, path), meta) in tqdm(enumerate(zip(inputs, metas)), desc=msg, total=len(metas)))
    else:
        all_outs = []
        for i, ((s, path), meta) in tqdm(enumerate(zip(inputs, metas)), desc=msg, total=len(metas)):
            out = process_file(
                path, 
                dist_threshold=args.dist_threshold, 
                prob_threshold=args.prob_threshold, 
                angle_threshold=args.angle_threshold,
                thresh_iters = args.thresh_iters, 
                hist_only=args.hist_only, 
                timeout = args.timeout)
            all_outs.append(out)

    best_seqs = [out[-1] for out in all_outs]
    all_dists = [x['dists'].mean() for best_seq in best_seqs for x in best_seq if x['dists'] is not None]
    all_meta = pd.DataFrame([x for out in all_outs for x in out[0]])
    all_meta['valid_rate'] = all_meta['n_valid']/all_meta['n_tot']
    print(f"Processing {args.split} split took {time.time() - start} seconds.")

    # Save lane information 
    print(f'Saving {len(all_outs)} scenarios...')
    shard_suffix = f'_shard{args.shard_idx}_{args.num_shards}' if args.num_shards > 1 else ''
    if not args.hist_only:
        with open(f'{CACHE_SUBDIR}/lanes{shard_suffix}.npz', 'wb') as f:
            np.savez_compressed(f, all_outs)
        all_meta.to_csv(f'{CACHE_SUBDIR}/lanes_meta{shard_suffix}.csv')
    else:
        with open(f'{CACHE_SUBDIR}/lanes_hist{shard_suffix}.npz', 'wb') as f:
            np.savez_compressed(f, all_outs)
        all_meta.to_csv(f'{CACHE_SUBDIR}/lanes_hist_meta{shard_suffix}.csv')
    print(f"All {len(all_outs)} scenarios have been processed and saved successfully.")