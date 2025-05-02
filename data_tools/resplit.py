import argparse
import numpy as np
import os
import pickle
import shutil

from natsort import natsorted
from operator import itemgetter
from tqdm import tqdm

def resplit_data(base_path, seed, splits):
    """Resplit the MTR dataset into training, validation, and test sets.
    Args:
        base_path (str): Base path where the MTR data is stored.
        seed (int): Random seed for reproducibility.
        splits (list of float): Ratios for train, validation, and test splits.
    """
    assert len(splits) == 3, "Splits must be a list of three floats."

    # Original MTR data splits 
    train_base = f'{base_path}/processed_scenarios_training'
    val_base = f'{base_path}/processed_scenarios_validation'    
    train_meta = f'{base_path}/processed_scenarios_training_infos.pkl'
    val_meta = f'{base_path}/processed_scenarios_val_infos.pkl'

    # New MTR data splits
    new_train_base = f'{base_path}/new_processed_scenarios_training'
    new_val_base = f'{base_path}/new_processed_scenarios_validation'
    new_test_base = f'{base_path}/new_processed_scenarios_testing'
    new_train_meta = f'{base_path}/new_processed_scenarios_training_infos.pkl'
    new_val_meta = f'{base_path}/new_processed_scenarios_val_infos.pkl'
    new_test_meta = f'{base_path}/new_processed_scenarios_test_infos.pkl'
    
    os.makedirs(new_train_base, exist_ok=True)
    os.makedirs(new_val_base, exist_ok=True)
    os.makedirs(new_test_base, exist_ok=True)

    train_inputs = [(x, f'{train_base}/{x}') for x in os.listdir(train_base)]
    with open(train_meta, 'rb') as f:
        train_metas = pickle.load(f)

    val_inputs = [(x, f'{val_base}/{x}') for x in os.listdir(val_base)]
    with open(val_meta, 'rb') as f:
        val_metas = pickle.load(f)

    input_scenarios = natsorted(train_inputs) + natsorted(val_inputs)
    input_metas = natsorted(train_metas, key=itemgetter('scenario_id')) + \
                  natsorted(val_metas, key=itemgetter('scenario_id'))
    zipped = zip(input_scenarios, input_metas)
    num_scenarios = len(input_scenarios)

    random_state = np.random.RandomState(seed)
    train_metas, val_metas, test_metas = [], [], []
    for (scenario, path), input_meta in tqdm(zipped, 'Processing scenarios...', total=num_scenarios):
        decision = random_state.rand()
        if decision < splits[0]:
            shutil.copy(path, f'{new_train_base}/{scenario}')
            train_metas.append(input_meta)
        elif decision < splits[0] + splits[1]:
            shutil.copy(path, f'{new_val_base}/{scenario}')
            val_metas.append(input_meta)
        else:
            assert decision < splits[0] + splits[1] + splits[2], 'Invalid random split'
            shutil.copy(path, f'{new_test_base}/{scenario}')
            test_metas.append(input_meta)
    
    with open(new_train_meta, 'wb') as f:
        pickle.dump(train_metas, f)
    with open(new_val_meta, 'wb') as f:
        pickle.dump(val_metas, f)
    with open(new_test_meta, 'wb') as f:
        pickle.dump(test_metas, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resplit data into training, validation, and test sets.")
    parser.add_argument(
        '--base_path', type=str, default='/data/driving/waymo', help='Path to the input data file.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random splitting seed.')
    parser.add_argument(
        '--splits', type=float, nargs=3, default=[0.85,0.075,0.075], 
        help='Split ratios for train, val, and test sets.')
    args = parser.parse_args()
    
    resplit_data(args.base_path, args.seed, args.splits)