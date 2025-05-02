import argparse
import pdb
import datetime
import os
import pickle as pkl
import logging
from pathlib import Path
import numpy as np
import torch
import hashlib
import sys
import json
import io
import time
import contextlib

from tqdm import tqdm
from copy import deepcopy
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import silhouette_score
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import dtw
from natsort import natsorted
import mdtraj as md
from scipy.interpolate import interp1d
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
from kneed import KneeLocator
import pandas as pd
import mdtraj.geometry.alignment as align
from mdtraj.utils import ensure_type
import pyhsmm

import utils.common as C
from safeshift_tools.compute_primitives import get_encounters, get_singles, get_velocities

from safeshift_tools.clustering_utilis import get_distmats, norm_encounters, get_aligns

def do_clustering(
    num_clusters: int = 20,
    num_scenarios: int =-1, 
    num_shards: int = 10, 
    hist_only: bool = False, 
    extrap: bool = False, 
    max_add: int = 100000,
    min_timesteps: int = 5, 
    load_dists: bool = False, 
    split: str = 'training'
):
    all_singles, all_pairs, single_aligns, pair_aligns = get_distmats(
        num_scenarios, num_shards, hist_only, extrap, max_add, min_timesteps, split, load_dists)
    single_data = np.array([x.flatten() for x in single_aligns])
    pair_data = np.array([x.flatten() for x in pair_aligns])

    print('Clustering...')
    start = time.time()
    kmeans_single = KMeans(n_clusters=num_clusters, init='k-means++', random_state=C.SEED)
    kmeans_single = kmeans_single.fit(single_data)
    kmeans_pair = KMeans(n_clusters=num_clusters, init='k-means++', random_state=C.SEED)
    kmeans_pair = kmeans_pair.fit(pair_data)
    print(f'Done in {time.time() - start}')

    CACHE_SUBDIR = os.path.join(C.CACHE_DIR, split, 'cluster')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)
    hist_suffix = '_hist' if hist_only else '_extrap' if extrap else ''
    single_path = f'{CACHE_SUBDIR}/single_kmeans{hist_suffix}.npz' 
    pair_path = f'{CACHE_SUBDIR}/pair_kmeans{hist_suffix}.npz' 
    with open(single_path, 'wb') as f:
        np.savez_compressed(f, {'labels': kmeans_single.labels_, 'centers': kmeans_single.cluster_centers_})
    with open(pair_path, 'wb') as f:
        np.savez_compressed(f, {'labels': kmeans_pair.labels_, 'centers': kmeans_pair.cluster_centers_})

    return kmeans_single, kmeans_pair

def do_labeling(
    num_shards: int = 10, 
    num_scenarios: int = -1, 
    hist_only: bool = False, 
    extrap: bool = False, 
    min_timesteps: int = 5, 
    parallel: bool = False,
    nproc: int = 20
):

    splits = ['training', 'validation', 'testing']
    # Now, we actually output the labeled primitives

    CACHE_SUBDIR = os.path.join(C.CACHE_DIR, 'training', 'cluster')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)
    hist_suffix = '_hist' if hist_only else '_extrap' if extrap else ''
    kmeans_single_path = f'{CACHE_SUBDIR}/single_kmeans{hist_suffix}.npz' 
    kmeans_pair_path = f'{CACHE_SUBDIR}/pair_kmeans{hist_suffix}.npz' 
    kmeans_single = np.load(kmeans_single_path, allow_pickle=True)['arr_0'].item()
    kmeans_pair = np.load(kmeans_pair_path, allow_pickle=True)['arr_0'].item()
    align_single_path = f'{CACHE_SUBDIR}/single_aligns{hist_suffix}.npz' 
    align_pair_path = f'{CACHE_SUBDIR}/pair_aligns{hist_suffix}.npz' 
    align_single = np.load(align_single_path, allow_pickle=True)['arr_0'].item()
    align_pair = np.load(align_pair_path, allow_pickle=True)['arr_0'].item()

    single_base1 = align_single['entries'][0][0]
    pair_base1, pair_base2 = align_pair['entries'][0][0], align_pair['entries'][0][1]

    for split in splits:
        CACHE_SUBDIR = os.path.join(C.CACHE_DIR, split, 'cluster')
        os.makedirs(CACHE_SUBDIR, exist_ok=True)
        for shard_idx in range(num_shards):
            base = C.load_base(
                split, num_shards=num_shards, shard_idx=shard_idx, num_scenarios=num_scenarios, 
                hist_only=hist_only, extrap=extrap, load_frenet=True, load_primitives=True)
            zipped = zip(base['metas'], base['inputs'], base['interp_vals'], base['primitives'])
            msg = f'Processing scenarios for {split}, shard {shard_idx}'

            def process(input, interp_val, primitive):
                _, _, singles, pairs = norm_encounters(
                    input, interp_val, primitive, 1e100, 0, 0, hist_only, min_timesteps)
                single_aligns, pair_aligns = get_aligns(
                    singles, pairs, single_base1, pair_base1, pair_base2, do_tmux=False)
                single_data = np.array([x.flatten() for x in single_aligns])
                pair_data = np.array([x.flatten() for x in pair_aligns])
                if len(single_data):
                    single_dists = np.linalg.norm(
                        kmeans_single['centers'] - single_data[:, np.newaxis, :], axis=-1) 
                    single_assignments = single_dists.argmin(axis=-1)
                    single_min_dists = single_dists.min(axis=-1)
                    single_out = np.stack([single_assignments, single_min_dists], axis=-1)
                else:
                    single_out = np.zeros((0, 2))

                if len(pair_data):
                    pair_dists = np.linalg.norm(
                        kmeans_pair['centers'] - pair_data[:, np.newaxis, :], axis=-1) 
                    pair_assignments = pair_dists.argmin(axis=-1)
                    pair_min_dists = pair_dists.min(axis=-1)
                    pair_out = np.stack([pair_assignments, pair_min_dists], axis=-1)
                else:
                    pair_out = np.zeros((0, 2))

                return single_out, pair_out

            if parallel:
                from joblib import Parallel, delayed    
                all_outs = Parallel(n_jobs=nproc, batch_size=4)(
                    delayed(process)(input, interp_val, primitive)
                    for _, input, interp_val, primitive in tqdm(
                        zip(zipped), msg, total=len(base['metas'])))
            else:
                all_outs = []
                for _, input, interp_val, primitive in tqdm(zip(zipped), msg, total=len(base['metas'])):
                    out = process(input, interp_val, primitive)
                    all_outs.append(out)

            single_outs = [x[0] for x in all_outs]
            pair_outs = [x[1] for x in all_outs]
            shard_suffix = f'_shard{shard_idx}_{10}'
            with open(f'{CACHE_SUBDIR}/single_center_dists{hist_suffix}{shard_suffix}.npz', 'wb') as f:
                np.savez_compressed(f, single_outs)
            with open(f'{CACHE_SUBDIR}/pair_center_dists{hist_suffix}{shard_suffix}.npz', 'wb') as f:
                np.savez_compressed(f, pair_outs)

def visualize_clusters(hist_only=False, extrap=False):
    CACHE_SUBDIR = os.path.join(C.CACHE_DIR, 'training', 'cluster')
    os.makedirs(CACHE_SUBDIR, exist_ok=True)

    VIS_SUBDIR = os.path.join(C.VISUAL_OUT_DIR, 'cluster')
    os.makedirs(VIS_SUBDIR, exist_ok=True)

    hist_suffix = '_hist' if hist_only else '_extrap' if extrap else ''
    kmeans_single_path = f'{CACHE_SUBDIR}/single_kmeans{hist_suffix}.npz' 
    kmeans_pair_path = f'{CACHE_SUBDIR}/pair_kmeans{hist_suffix}.npz' 
    kmeans_single = np.load(kmeans_single_path, allow_pickle=True)['arr_0'].item()
    kmeans_pair = np.load(kmeans_pair_path, allow_pickle=True)['arr_0'].item()

    n_single = np.unique(kmeans_single['labels'], return_counts=True)[-1]
    n_pair = np.unique(kmeans_pair['labels'], return_counts=True)[-1]
    for i, (center, n) in enumerate(zip(kmeans_single['centers'], n_single)):
        folder = os.path.join(VIS_SUBDIR, f'single{hist_suffix}')
        os.makedirs(folder, exist_ok=True)

        plt.clf()
        center = center.reshape(-1, 3)
        plt.plot(center[:, 0], center[:, 1], marker='.')
        plt.title(f'cluster single{hist_suffix} {i}, N={n}')
        plt.savefig(os.path.join(folder, f'cluster_{i}.png'), dpi=300)

    for i, (center, n) in enumerate(zip(kmeans_pair['centers'], n_pair)):
        folder = os.path.join(VIS_SUBDIR, f'pair{hist_suffix}')
        os.makedirs(folder, exist_ok=True)

        plt.clf()
        center = center.reshape(-1, 3)
        center1 = center[:int(len(center)/2)]
        center2 = center[int(len(center)/2):]
        plt.plot(center1[:, 0], center1[:, 1], marker='.')
        plt.plot(center2[:, 0], center2[:, 1], marker='.')
        plt.title(f'cluster pair{hist_suffix} {i}, N={n}')
        plt.savefig(os.path.join(folder, f'cluster_{i}.png'), dpi=300)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Cluster primitives")
    parser.add_argument('--num_clusters', type=int, default=20)
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--split', default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--num_shards', type=int, default=-1)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--hist_only', action='store_true')
    parser.add_argument('--min_timesteps', type=int, default=5, help='Minimum length of a primitive before normalizing')
    parser.add_argument('--max_add', type=int, default=100000)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--extrap', action='store_true')
    parser.add_argument('--nproc', type=int, default=20)
    parser.add_argument('--load_dists', action='store_true', help='Load distances')
    parser.add_argument('--load_labels', action='store_true', help='Load cluster labels')
    args = parser.parse_args()

    assert not (args.extrap and args.hist_only), 'Only one of extrap and hist_only permitted'

    # Takes ~1-2 minutes
    if not args.load_labels:
        do_clustering(
            num_clusters=args.num_clusters, num_scenarios=args.num_scenarios, 
            num_shards=args.num_shards, hist_only=args.hist_only, extrap=args.extrap, 
            max_add=args.max_add, min_timesteps=args.min_timesteps, load_dists=args.load_dists)
        
        do_labeling(
            num_shards=args.num_shards, num_scenarios=args.num_scenarios, 
            hist_only=args.hist_only, extrap=args.extrap, min_timesteps=args.min_timesteps, 
            parallel=args.parallel, nproc=args.nproc
        )
    
    visualize_clusters(args.hist_only, args.extrap)