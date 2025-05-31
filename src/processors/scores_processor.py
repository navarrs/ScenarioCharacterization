import numpy as np
import os
import pickle

from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.datasets.dataset import BaseDataset
from src.scores.base_score import BaseScore
from utils.logger import get_logger

from typing import Dict, List, AnyStr
logger = get_logger(__name__)


class ScoresProcessor:
    def __init__(self, config: DictConfig, dataset: BaseDataset, scorer: BaseScore) -> None:
        self.parallel = config.parallel if "parallel" in config else True
        self.batch_size = config.batch_size if "batch_size" in config else 4
        self.num_processes = config.num_processes if "num_processes" in config else 10
        self.save = config.save if "save" in config else True
        self.scenario_type = config.scenario_type if "scenario_type" in config else 'gt'
        
        self.feature_paths = config.feature_paths if "feature_paths" in config else None
        if self.feature_paths is None:
            raise ValueError("Feature paths must be specified in the configuration.")
        self.features_to_score = list(self.feature_paths.keys())

        self.output_path = config.output_path if "output_path" in config else None
        if self.output_path is None:
            raise ValueError("Output path must be specified in the configuration.")

        self.dataset = dataset
        self.scorer = scorer

    def name(self):
        """
        Identify the feature and dataset being processed.
        This method can be overridden by subclasses to provide specific identification.
        """
        return f"{self.__class__.__name__}"

    def run(self):
        logger.info(f"Processing {self.features_to_score} {self.scorer.name} scores for {self.dataset.name()}.")

        # TODO: generalize this unpacking 
        features = {}
        for key, path in self.feature_paths.items():
            features[key] = [feature for feature in np.load(path, allow_pickle=True)['features']]
        zipped = zip(*[features[key] for key in self.features_to_score])

        if self.parallel:
            from joblib import Parallel, delayed
            scores = Parallel(n_jobs=self.num_processes, batch_size=self.batch_size)(
                delayed(self.scorer.compute)(scenario_features=feature)
                for feature in tqdm(zipped, total=len(self.dataset))
            )
        else:
            scores = []
            for feature in tqdm(zipped, total=len(self.dataset)):
                out = self.scorer.compute(scenario_features=feature)
                scores.append(out)

        if self.save:
            cache_filepath = os.path.join(self.output_path, f"{self.scorer.name}.npz")
            logger.info(f"Saving processed scores to {cache_filepath}")
            with open(cache_filepath, 'wb') as f:
                np.savez_compressed(f, scores=scores)
        
        return scores