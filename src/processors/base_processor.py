import os
import pickle

from abc import ABC, abstractmethod
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import Dict, AnyStr

from src.features.base_feature import BaseFeature
from src.scorer.base_scorer import BaseScorer
from src.utils.common import get_logger

logger = get_logger(__name__)

class BaseProcessor(ABC):
    def __init__(
        self, config: DictConfig, dataset: Dataset, characterizer: BaseFeature | BaseScorer
    ) -> None:
        """ Initialize the FeatureProcessor with a configuration, dataset, and feature.
        :param config: Configuration for the feature processor, including parameters like
                       batch size, number of workers, and whether to save the output.
        :param dataset: The dataset to process, which should be a subclass of torch.utils.data.Dataset.
        :param processor: An instance of BaseFeature or its subclass that defines the feature to compute.
        """
        super(BaseProcessor, self).__init__()

        self.scenario_type = config.scenario_type if "scenario_type" in config else 'gt'
        self.dataset = dataset
        self.characterizer = characterizer

        # DataLoader parameters
        self.batch_size = config.get("batch_size", 4)
        self.num_workers = config.get("num_worker", 4)
        self.shuffle = config.get("shuffle", False)

        self.save = config.get("save", True)
        self.output_path = config.get("output_path", None)
        if self.save:
            if self.output_path is None:
                logger.error("Output path must be specified in the configuration.")
                raise ValueError
            else: 
                logger.info(f"Features {self.characterizer.name} will be saved to {self.output_path}")

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_batch
        )

    @property
    def name(self):
        """ Identify the feature and dataset being processed. """
        return f"{self.__class__.__name__}"

    @abstractmethod
    def run(self):
        """ """
        raise NotImplementedError("The run method must be implemented in the subclass.")

    # TODO: remove I/O methods from class?
    def to_pickle(self, input_data: Dict, tag: AnyStr):
        """ Save data to a pickle file. """
        data = {}
        data_file = os.path.join(self.output_path, f"{tag}.pkl")
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)

        for key, value in input_data.items():
            data[key] = value
        
        with open(data_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def from_pickle(self, data_file: AnyStr) -> Dict:
        """ Load data from a pickle file. """
        if not os.path.exists(data_file):
            logger.error(f"Data file {data_file} does not exist.")
            raise FileNotFoundError(f"Data file {data_file} does not exist.")
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        return data