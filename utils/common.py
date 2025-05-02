import numpy as np
import os
import pandas as pd
import pickle
import time

BASE_PATH = "/data/driving/waymo/"
CACHE_DIR = f"{BASE_PATH}/safeshift/cache/"
VISUAL_OUT_DIR = f"{BASE_PATH}/safeshift/vis/"
MEASUREMENT_OUT_DIR = f"{BASE_PATH}/safeshift/measurements/"
STATS_OUT_DIR = f"{BASE_PATH}/safeshift/stats/"

# center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y
AGENT_DIMS = [False, False, False, True, True, True, False, False, False]
HEADING_IDX = [False, False, False, False, False, False, True, False, False]
POS_XY_IDX = [True, True, False, False, False, False, False, False, False]
POS_XYZ_IDX = [True, True, True, False, False, False, False, False, False]
VEL_XY_IDX = [False, False, False, False, False, False, False, True, True]

# Interpolated stuff
IPOS_XY_IDX = [True, True, False, False, False, False, False]
IPOS_SDZ_IDX = [False, False, True, True, True, False, False]
IPOS_SD_IDX = [False, False, False, True, True, False, False]
ILANE_IDX = [False, False, False, False, False, True, False]
IVALID_IDX = [False, False, False, False, False, False, True]

AGENT_TYPE_MAP = {'TYPE_VEHICLE': 0, 'TYPE_PEDESTRIAN': 1, 'TYPE_CYCLIST': 2}
AGENT_NUM_TO_TYPE = {0: 'TYPE_VEHICLE', 1: 'TYPE_PEDESTRIAN', 2: 'TYPE_CYCLIST'}

LAST_TIMESTEP = 91
HIST_TIMESTEP = 11
STATIONARY_SPEED = 0.25 # m/s

SEED = 42

def load_base(
    split: str, 
    num_shards: int = 10, 
    shard_idx: int = 0, 
    num_scenarios: int = -1, 
    hist_only: bool = False, 
    extrap: bool = False, 
    load_lanes: bool = False, 
    load_frenet: bool = False,
    load_primitives: bool = False, 
):
    step = 1
    if split == 'training':
        step = 5
        scenarios_base = f'{BASE_PATH}/new_processed_scenarios_training'
        scenarios_meta = f'{BASE_PATH}/new_processed_scenarios_training_infos.pkl'
    elif split == 'validation':
        scenarios_base = f'{BASE_PATH}/new_processed_scenarios_validation'
        scenarios_meta = f'{BASE_PATH}/new_processed_scenarios_val_infos.pkl'
    else:
        scenarios_base = f'{BASE_PATH}/new_processed_scenarios_testing'
        scenarios_meta = f'{BASE_PATH}/new_processed_scenarios_test_infos.pkl'

    start = time.time()
    print(f"Loading {split} Scenario Data...")
    with open(scenarios_meta, 'rb') as f:
        metas = pickle.load(f)[::step]
    inputs = [(f'sample_{x["scenario_id"]}.pkl', 
               f'{scenarios_base}/sample_{x["scenario_id"]}.pkl') for x in metas]
    print(f"Process took {time.time() - start} seconds.")

    closest_lanes = None
    if load_lanes:
        start = time.time()
        # Train takes ~200s for full, ~60s for hist only
        # Test/val takes ~90s for full, ~30s for hist only
        # Check if lane cache has been computed otherwise return value error
        print("Loading lane cache")
        file_name = 'lanes' if not hist_only else 'lanes_hist'
        shard_suffix = f'_shard{shard_idx}_{num_shards}' if num_shards > 1 else ''
        closest_lanes_filepath = os.path.join(
            CACHE_DIR, split, "closest_lanes", f"{file_name}{shard_suffix}.npz")
        closest_lanes_metapath = os.path.join(
            CACHE_DIR, split, "closest_lanes", f"{file_name}_meta{shard_suffix}.csv")
        assert os.path.exists(closest_lanes_filepath) and os.path.exists(closest_lanes_metapath), \
            f"MEAT lanes need to be cached"
        
        closest_lanes = np.load(closest_lanes_filepath, allow_pickle=True)['arr_0']
        all_meta = pd.read_csv(closest_lanes_metapath)
        all_meta = all_meta.drop(columns='Unnamed: 0')
        print(f"Loading closest lanes took {time.time() - start} seconds")

        closest_lanes = [info[-1] for info in closest_lanes]

    interp_vals = None
    if load_frenet:
        start = time.time()
        print(f"Loading frenet cache {shard_idx} of {num_shards}")
        file_name = 'interp' if (not hist_only and not extrap) else 'interp_hist'
        shard_suffix = f'_shard{shard_idx}_{num_shards}' if num_shards > 1 else ''
        interp_vals_filepath = os.path.join(CACHE_DIR, f"{split}/frenet/{file_name}{shard_suffix}.npz")
        interp_vals = np.load(interp_vals_filepath, allow_pickle=True)['arr_0']
        print(f"Loading shards took {time.time() - start} seconds")

    primitives = None
    if load_primitives:
        print(f"Loading primitive cache {shard_idx} of {10}")
        file_name = 'prims_hist' if hist_only else 'prims_extrap' if extrap else 'prims'
        shard_suffix = f'_shard{shard_idx}_{num_shards}' if num_shards > 1 else ''
        primitives_filepath = os.path.join(CACHE_DIR, f"{split}/primitives_spline/{file_name}{shard_suffix}.npz")
        primitives = np.load(primitives_filepath, allow_pickle=True)['arr_0']
        print(f"Loading primitive shards took {time.time() - start} seconds")

    n_per_shard = np.ceil(len(metas)/num_shards)
    shard_start = int(n_per_shard*shard_idx)
    shard_end = int(n_per_shard*(shard_idx + 1))
    metas = metas[shard_start:shard_end]
    inputs = inputs[shard_start:shard_end]

    tot_scenarios = len(metas)
    if num_scenarios != -1:
        tot_scenarios = num_scenarios
        metas = metas[:tot_scenarios]
        inputs = inputs[:tot_scenarios]
        interp_vals = interp_vals[:tot_scenarios] if interp_vals is not None else None
        closest_lanes = closest_lanes[:tot_scenarios] if closest_lanes is not None else None

    return {
        'metas': metas, 
        'inputs': inputs, 
        'interp_vals': interp_vals, 
        'primitives': primitives, 
        'closest_lanes': closest_lanes,
        'closest_lanes_meta': all_meta if load_lanes else None,
    }