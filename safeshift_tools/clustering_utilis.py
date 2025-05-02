import numpy as np
import os
import time

import utils.common as C

from tqdm import tqdm
from scipy.spatial.transform import Rotation

def normalize_time(xyz, step=0.01):
    assert len(xyz) > 1, 'At least one data point needed to perform linear interpolation'
    timesteps = np.arange(len(xyz))
    timesteps = timesteps / timesteps[-1]
    new_times = np.arange(0, 1+step, step)
    xyz_norm = np.stack([np.interp(new_times, timesteps, xyz[:, i]) for i in range(xyz.shape[1])], axis=-1)
    return xyz_norm

def center_traj(xyz):
    return xyz - xyz.mean(axis=0)

def optimally_align_singles(a, b):
    """ Computes optimal alignment
        Returns rmsd: float,
                rot: Rotation [to apply to center_traj(b)]
    """
    assert len(set([len(x) for x in [a, b]])) == 1, 'All vectors must have the same length'
    a = center_traj(a)
    b = center_traj(b)
    rot, rssd = Rotation.align_vectors(a, b)
    return rssd / np.sqrt(len(a)), rot

def optimally_align_and_order_pairs(a, b, c, d):
    """ Computes optimal alignment and ordering. 
        center_traj(a cat b) to center_traj(c cat d), as well as 
        center_traj(a cat b) to center_traj(d cat c) are tested.
        Returns rmsd: float,
                rot: Rotation [to apply to center_traj(c cat d) or center_traj(d cat c)],
                swap_cd: bool [whether or not to swap c and d in order]
    """
    assert len(set([len(x) for x in [a, b, c, d]])) == 1, 'All vectors must have the same length'
    ab = center_traj(np.concatenate([a, b], axis=0))
    cd = center_traj(np.concatenate([c, d], axis=0))
    dc = center_traj(np.concatenate([d, c], axis=0))
    rot_cd, rssd_cd = Rotation.align_vectors(ab, cd)
    rot_dc, rssd_dc = Rotation.align_vectors(ab, dc)
    rmsd_cd = rssd_cd / np.sqrt(len(a))
    rmsd_dc = rssd_dc / np.sqrt(len(a))

    if rmsd_cd <= rmsd_dc:
        return rmsd_cd, rot_cd, False
    else:
        return rmsd_dc, rot_dc, True

def rmsd(a, b):
    distances = np.linalg.norm(a - b, axis=-1)
    rmsd = np.sqrt((distances ** 2).sum() / len(distances))
    return rmsd

def norm_encounters(input, interp_val, primitive, max_add, n_added_single, n_added_pair, hist_only=False, min_timesteps=5):
    # First geometric approximation: orient everything to the first element, then do k-means with aligned L2 dist.
    single_out = []
    for single_info in primitive['singles']:
        if n_added_single >= max_add:
            break
        agent_idx, start, end, prim_id = single_info.astype(int)
        if end - start < min_timesteps:
            continue
        n_added_single += 1
        xyz = interp_val[agent_idx, start:end, :3]
        xyz_norm = normalize_time(xyz)
        single_out.append(xyz_norm)
    single_out = np.array(single_out)
    # N_s x 1 x 101 x 3
    single_out = single_out[:, np.newaxis]

    pair_out = []
    for pair_info in primitive['pairs']:
        if n_added_pair >= max_add:
            break
        agent_idx1, agent_idx2, start, end, prim_id = pair_info.astype(int)
        if end - start < min_timesteps:
            continue
        n_added_pair += 1
        xyz1 = interp_val[agent_idx1, start:end, :3]
        xyz_norm1 = normalize_time(xyz1)
        xyz2 = interp_val[agent_idx2, start:end, :3]
        xyz_norm2 = normalize_time(xyz2)
        pair_out.append([xyz_norm1, xyz_norm2])
    # N_p x 2 x 101 x 3
    pair_out = np.array(pair_out)

    return n_added_single, n_added_pair, single_out, pair_out

def get_aligns(all_singles, all_pairs, single_base1=None, pair_base1=None, pair_base2=None, do_tmux=True):
    if single_base1 is None:
        single_base1 = all_singles[0][0]
    if pair_base1 is None or pair_base2 is None:
        pair_base1, pair_base2 = all_pairs[0][0], all_pairs[0][1]

    single_aligned = []
    for single in tqdm(all_singles, 'Aligning singles', disable=(not do_tmux)):
        xyz1 = single[0]
        _, rot = optimally_align_singles(single_base1, xyz1)
        aligned1 = rot.apply(center_traj(xyz1))
        single_aligned.append(aligned1)

    pair_aligned = []
    for pair in tqdm(all_pairs, 'Aligning pairs', disable=(not do_tmux)):
        xyz1, xyz2 = pair[0], pair[1]
        _, rot, swap = optimally_align_and_order_pairs(pair_base1, pair_base2, xyz1, xyz2)
        aligned12 = rot.apply(center_traj(np.concatenate(
            [xyz1 if not swap else xyz2, xyz2 if not swap else xyz1], axis=0)))
        pair_aligned.append(aligned12)
    return single_aligned, pair_aligned

def get_distmats(
    num_scenarios: int =-1, 
    num_shards: int = -1,
    hist_only: bool = False, 
    extrap: bool = False, 
    max_add: int = 100000,
    min_timesteps: int = 5, 
    split: str = 'training',
    load_dists: bool = False
):
    CACHE_SUBDIR = os.path.join(C.CACHE_DIR, split, 'cluster')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)
    
    hist_suffix = '_hist' if hist_only else '_extrap' if extrap else ''
    
    single_path = f'{CACHE_SUBDIR}/single_aligns{hist_suffix}.npz' 
    pair_path = f'{CACHE_SUBDIR}/pair_aligns{hist_suffix}.npz' 

    if load_dists:
        print('Loading dist matrices from disk')
        start = time.time()
        single_dict = np.load(single_path, allow_pickle=True)['arr_0']
        pair_dict = np.load(pair_path, allow_pickle=True)['arr_0']
        print(f'Process took {time.time() - start}')
        single_dict = single_dict.item()
        pair_dict = pair_dict.item()
        return single_dict['entries'], pair_dict['entries'], single_dict['aligns'], pair_dict['aligns']
    
    # Max for performing the actual clustering, since intractable to go much beyond that with N^3 algorithms
    n_added_single, n_added_pair = 0, 0
    base = C.load_base(
        split, num_shards=num_shards, shard_idx=0, num_scenarios=num_scenarios, hist_only=hist_only, 
        extrap=extrap, load_frenet=True, load_primitives=True)
    zipped = zip(base['metas'], base['inputs'], base['interp_vals'], base['primitives'])
    msg = 'Processing scenarios'

    all_singles, all_pairs = [], []
    for _, input, interp_val, primitive in tqdm(zipped, msg, total=len(base['interp_vals'])):
        new_single, new_pair, single_out, pair_out = norm_encounters(
            input, interp_val, primitive, max_add, n_added_single, n_added_pair, hist_only=hist_only, 
            min_timesteps=min_timesteps)
        if len(single_out):
            all_singles.append(single_out)

        if len(pair_out):
            all_pairs.append(pair_out)

        n_added_single = new_single
        n_added_pair = new_pair
        if n_added_single >= max_add and n_added_pair >= max_add:
            break
    all_singles = np.concatenate(all_singles, axis=0)
    all_pairs = np.concatenate(all_pairs, axis=0)

    single_aligns, pair_aligns = get_aligns(all_singles, all_pairs)

    # Model output path
    print('Saving dist matrices to disk')
    with open(single_path, 'wb') as f:
        np.savez_compressed(f, {'aligns': single_aligns, 'entries': all_singles})
    with open(pair_path, 'wb') as f:
        np.savez_compressed(f, {'aligns': pair_aligns, 'entries': all_pairs})

    return all_singles, all_pairs, single_aligns, pair_aligns