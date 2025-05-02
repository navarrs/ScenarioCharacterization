import argparse
import numpy as np
import os
import time

from tqdm import tqdm

from safeshift_tools.primitives_utils import assign_primitives
from utils.common import CACHE_DIR, load_base
from utils.systems import tqdm_joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--num_shards', type=int, default=10)
    parser.add_argument('--hist_only', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--extrap', action='store_true')
    parser.add_argument('--nproc', type=int, default=20)
    parser.add_argument('--min_timesteps', type=int, default=5)
    args = parser.parse_args()

    assert not (args.extrap and args.hist_only), 'Only one of extrap and hist_only permitted'

    # Model output path
    hist_suffix = '_hist' if args.hist_only else '_extrap' if args.extrap else ''

    # Now, we actually output the labeled primitives
    # for split in ['training', 'validation', 'testing']:
    for split in ['testing']:
        CACHE_SUBDIR = os.path.join(CACHE_DIR, split, 'primitives')
        os.makedirs(CACHE_SUBDIR, exist_ok=True)
        for shard_idx in range(args.num_shards):
            base = load_base(split, 
                num_shards=args.num_shards, shard_idx=shard_idx, num_scenarios=args.num_scenarios, 
                hist_only=args.hist_only, extrap=args.extrap, load_frenet=True)
            metas, inputs, interp_vals = base['metas'], base['inputs'], base['interp_vals']

            msg = f'Processing scenarios for {split}, shard {shard_idx}'
            start_time = time.time()

            if args.parallel:
                from joblib import Parallel, delayed    
                with tqdm_joblib(tqdm(desc=msg, leave=True, total=len(metas))) as progress_bar:
                    all_outs = Parallel(n_jobs=args.nproc, batch_size=1)(
                        delayed(assign_primitives)(
                        input, 
                        interp_val, 
                        hist_only=args.hist_only or args.extrap, 
                        # extrap=args.extrap, 
                        min_timesteps=args.min_timesteps)
                        for _, input, interp_val in zip(metas, inputs, interp_vals))
            else:
                all_outs = []
                for _, input, interp_val in tqdm(zip(metas, inputs, interp_vals), msg, total=len(metas)):
                    out = assign_primitives(
                        input, 
                        interp_val, 
                        hist_only=args.hist_only, 
                        extrap=args.extrap, 
                        min_timesteps=args.min_timesteps)
                    all_outs.append(out)
            
            print(f'Done in {time.time() - start_time}')

            shard_suffix = f'_shard{shard_idx}_{10}'
            with open(f'{CACHE_SUBDIR}/prims{hist_suffix}{shard_suffix}.npz', 'wb') as f:
                np.savez_compressed(f, all_outs)
    print("Done.")