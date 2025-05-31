import numpy as np
import os
import pickle

from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.datasets.dataset import BaseDataset
from src.features.base_feature import BaseFeature
from utils.logger import get_logger

from typing import Dict, List, AnyStr
logger = get_logger(__name__)


class FeatureProcessor:
    def __init__(
        self, config: DictConfig, dataset: BaseDataset, feature: BaseFeature 
    ) -> None:
        self.parallel = config.parallel if "parallel" in config else True
        self.batch_size = config.batch_size if "batch_size" in config else 4
        self.num_processes = config.num_processes if "num_processes" in config else 10
        self.save = config.save if "save" in config else True
        self.scenario_type = config.scenario_type if "scenario_type" in config else 'gt'

        self.output_path = config.output_path if "output_path" in config else None
        if self.output_path is None:
            raise ValueError("Output path must be specified in the configuration.")

        self.dataset = dataset
        self.feature = feature

    def name(self):
        """
        Identify the feature and dataset being processed.
        This method can be overridden by subclasses to provide specific identification.
        """
        return f"{self.__class__.__name__}"

    def run(self):
        logger.info(f"Processing {self.feature.name} features for {self.dataset.name()}.")
        zipped = self.dataset.get_zipped()
        if self.parallel:
            from joblib import Parallel, delayed

            features = Parallel(n_jobs=self.num_processes, batch_size=self.batch_size)(
                delayed(self.process_scenario)(
                    scenario_id=scenario_id, 
                    scenario_path=scenario_path
                )
                for scenario_id, scenario_path, scenario_meta in tqdm(zipped, total=len(self.dataset))
            )
        else:
            features = []
            for scenario_id, scenario_path, scenario_meta in tqdm(zipped, total=len(self.dataset)):
                out = self.process_scenario(scenario_id=scenario_id, scenario_path=scenario_path)
                features.append(out)

        if self.save:
            cache_filepath = os.path.join(self.output_path, f"{self.feature.name}.npz")
            logger.info(f"Saving processed features to {cache_filepath}")
            with open(cache_filepath, 'wb') as f:
                np.savez_compressed(f, features=features)
        
        return features

    def process_scenario(self, scenario_id: AnyStr, scenario_path: AnyStr):
        """
        Base method to process a file.
        Should be overridden by subclasses.

        :param file_path: Path to the file to process
        """
        # TODO: Should not pass a path and load anything here in principle. 
        # TODO: Remove any pickle loading
        with open(scenario_path, 'rb') as f:
            scenario = pickle.load(f)

        # -------------------------------
        # TODO: handle scenario type here
        # -------------------------------
        return self.feature.compute(scenario, scenario_id)