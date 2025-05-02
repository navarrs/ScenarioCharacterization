import numpy as np
import os
import pickle 
import time

import utils.common as C

from tqdm import tqdm

from data_tools.visualization import plot_static_map_infos
from safeshift_tools.closest_lane_utils import build_lane_graph
from safeshift_tools.frenet_utils import process_traj

def process_file(
    path: str, 
    meta: str,  
    plot: bool = False, 
    tag: str ='temp', 
    hist_only: bool = False,
    closest_lanes = None, 
    meta_info=None, 
    linear_only: bool = False, 
    frenet_t_lower: int = -1, 
    frenet_t_upper: int = 2, 
    frenet_d_max=2.5
):
    # Load the scenario 
    with open(path, 'rb') as f:
        scenario = pickle.load(f)
    
    # Trajectory data:
    #    center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
    track_infos = scenario['track_infos']
    objects_type = track_infos['object_type']

    # Map infos:
    #   lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, all_polylines
    static_map_infos = scenario['map_infos']
    dynamic_map_infos = scenario['dynamic_map_infos']

    static_map_pos = plot_static_map_infos(static_map_infos, ax=None, dim=3)
    lane_pos = static_map_pos['lane']
    
    lanes = static_map_infos['lane']
    lane_graph = build_lane_graph(lanes)
    
    last_t = C.LAST_TIMESTEP if not hist_only else C.HIST_TIMESTEP
    # Trajectories --> (num_agents, time_steps, 9)
    trajectories = track_infos['trajs'][:, :last_t, :-1]
    # Mask         --> (num_agents, time_steps, 1)
    valid_masks = track_infos['trajs'][:, :last_t, -1] > 0
    
    num_agents, time_steps, dim = trajectories.shape
    assert closest_lanes is not None, 'Non-cached closest lanes unsupported currently'
    assert meta_info is not None, 'Non-cached closest lanes unsupported currently'
    
    closest_lanes_idx = 0
    outs = []
    for n in range(num_agents):
        mask = valid_masks[n]
        # This is vital, since there are a decent amount of hist_only with no points in hist
        if not np.any(mask):
            # Just save entire sequence as np.infs for invalid
            lane_info = None
            closest_meta = None
        else:
            lane_info = closest_lanes[closest_lanes_idx]
            closest_meta = meta_info.iloc[closest_lanes_idx]
            closest_lanes_idx += 1
        
        if not hist_only:
            out = process_traj(
                trajectories[n][:, C.POS_XYZ_IDX], 
                trajectories[n][:, C.VEL_XY_IDX], 
                mask,
                lane_info, 
                lane_pos, 
                lanes, 
                closest_meta, 
                linear_only,
                frenet_t_lower=frenet_t_lower, 
                frenet_t_upper=frenet_t_upper, 
                frenet_d_max=frenet_d_max)
        else:
            # for hist_only also project the future 80 points for easier metric computation
            traj_points = np.zeros((C.LAST_TIMESTEP, 3), dtype=trajectories.dtype) * np.nan
            traj_vel = np.zeros((C.LAST_TIMESTEP, 2), dtype=trajectories.dtype) * np.nan
            traj_mask = np.zeros((C.LAST_TIMESTEP,), dtype=mask.dtype)

            traj_points[:C.HIST_TIMESTEP] = trajectories[n][:, C.POS_XYZ_IDX]
            traj_vel[:C.HIST_TIMESTEP] = trajectories[n][:, C.VEL_XY_IDX]
            traj_mask[:C.HIST_TIMESTEP] = mask
            out = process_traj(
                traj_points, 
                traj_vel, 
                traj_mask,
                lane_info, 
                lane_pos, 
                lanes, 
                closest_meta, 
                linear_only,
                frenet_t_lower=frenet_t_lower, 
                frenet_t_upper=frenet_t_upper, 
                frenet_d_max=frenet_d_max)
        outs.append(out)

    return np.array(outs)
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split', type=str, default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--hist_only', action='store_true')
    parser.add_argument('--linear_only', action='store_true')
    parser.add_argument('--load_cache', action='store_true')
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--frenet_t_lower', default=0, type=float)
    parser.add_argument('--frenet_t_upper', default=1, type=float)
    parser.add_argument('--frenet_d_max', default=2.5, type=float)
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=10)
    args = parser.parse_args()

    base = C.load_base(
        args.split, num_shards=args.num_shards, shard_idx=args.shard_idx, 
        num_scenarios=args.num_scenarios, hist_only=args.hist_only, extrap=False, load_lanes=True)
    metas, inputs = base['metas'], base['inputs']
    closest_lanes, closest_lane_meta = base['closest_lanes'], base['closest_lanes_meta']

    CACHE_SUBDIR = os.path.join(C.CACHE_DIR, args.split, 'frenet')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)

    agents_per_scene = np.array([len(x) for x in closest_lanes])
    n_agents = np.sum(agents_per_scene)
    closest_lane_meta = closest_lane_meta[:n_agents]
    tots1 = np.cumsum(agents_per_scene)
    tots0 = np.array([0] + [*tots1[:-1]])
    closest_lane_metas = [closest_lane_meta[x0:x1] for x0, x1 in zip(tots0, tots1)]

    msg = f'Processing {args.split} split scenarios...'
    start = time.time()
    if args.parallel:
        from joblib import Parallel, delayed    
        all_outs = Parallel(n_jobs=args.nproc, batch_size=4)(
            delayed(process_file)(
                path, 
                meta, 
                args.plot, 
                tag=f"{s.split('.')[0]}",
                hist_only=args.hist_only, 
                closest_lanes=lane_info, 
                meta_info=meta_info, 
                linear_only=args.linear_only,
                frenet_t_lower=args.frenet_t_lower, 
                frenet_t_upper=args.frenet_t_upper, 
                frenet_d_max=args.frenet_d_max)
            for (s, path), meta, lane_info, meta_info in tqdm(
                zip(inputs, metas, closest_lanes, closest_lane_metas), desc=msg, total=len(metas)))
    else:
        all_outs = []
        for (s, path), meta, lane_info, meta_info in tqdm(
            zip(inputs, metas, closest_lanes, closest_lane_metas), desc=msg, total=len(metas)):
            out = process_file(
                path, 
                meta, 
                args.plot, 
                tag=f"{s.split('.')[0]}",
                hist_only=args.hist_only, 
                closest_lanes=lane_info, 
                meta_info=meta_info, 
                linear_only=args.linear_only,
                frenet_t_lower=args.frenet_t_lower, 
                frenet_t_upper=args.frenet_t_upper, 
                frenet_d_max=args.frenet_d_max)
            all_outs.append(out)

    print(f"Process took {time.time() - start} seconds.")

    print(f'Saving {len(all_outs)} scenarios...')
    shard_suffix = f'_shard{args.shard_idx}_{args.num_shards}' if args.num_shards > 1 else ''
    if not args.hist_only:
        with open(f'{CACHE_SUBDIR}/interp{shard_suffix}.npz', 'wb') as f:
            np.savez_compressed(f, all_outs)
    else:
        with open(f'{CACHE_SUBDIR}/interp_hist{shard_suffix}.npz', 'wb') as f:
            np.savez_compressed(f, all_outs)
    print("Done.")
