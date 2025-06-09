import os
import pickle
from abc import ABC, abstractmethod

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from features.base_feature import BaseFeature
from scorer.base_scorer import BaseScorer
from utils.common import get_logger

logger = get_logger(__name__)


class BaseProcessor(ABC):
    def __init__(
        self,
        config: DictConfig,
        dataset: Dataset,
        characterizer: BaseFeature | BaseScorer,
    ) -> None:
        """Initializes the BaseProcessor with configuration, dataset, and characterizer.

        Args:
            config (DictConfig): Configuration for the processor, including parameters like batch size, number of workers, and output path.
            dataset (Dataset): The dataset to process, which should be a subclass of torch.utils.data.Dataset.
            characterizer (BaseFeature | BaseScorer): An instance that defines the feature or score to compute.

        Raises:
            ValueError: If saving is enabled but no output path is specified.
        """
        super(BaseProcessor, self).__init__()

        self.scenario_type = config.scenario_type if "scenario_type" in config else "gt"
        self.dataset = dataset
        self.characterizer = characterizer

        # DataLoader parameters
        self.batch_size = config.get("batch_size", 4)
        self.num_workers = config.get("num_workers", 4)
        self.shuffle = config.get("shuffle", False)

        self.save = config.get("save", True)
        self.output_path = config.get("output_path", None)
        if self.save:
            if self.output_path is None:
                logger.error("Output path must be specified in the configuration.")
                raise ValueError
            else:
                logger.info(
                    f"Features {self.characterizer.name} will be saved to {self.output_path}"
                )

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_batch,
        )

    @property
    def name(self):
        """Identifies the feature and dataset being processed.

        Returns:
            str: The name of the processor class.
        """
        return f"{self.__class__.__name__}"

    @abstractmethod
    def run(self):
        """Runs the processor.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("The run method must be implemented in the subclass.")

    def to_pickle(self, input_data: dict, tag: str) -> None:
        """Saves data to a pickle file.

        Args:
            input_data (Dict): The data to save.
            tag (AnyStr): The tag to use for the output file name.
        """
        data = {}
        data_file = os.path.join(self.output_path, f"{tag}.pkl")
        if os.path.exists(data_file):
            with open(data_file, "rb") as f:
                data = pickle.load(f)

        for key, value in input_data.items():
            data[key] = value

        with open(data_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def from_pickle(self, data_file: str) -> dict:
        """Loads data from a pickle file.

        Args:
            data_file (AnyStr): The path to the pickle file.

        Returns:
            Dict: The loaded data.

        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        if not os.path.exists(data_file):
            logger.error(f"Data file {data_file} does not exist.")
            raise FileNotFoundError(f"Data file {data_file} does not exist.")

        with open(data_file, "rb") as f:
            data = pickle.load(f)

        return data
