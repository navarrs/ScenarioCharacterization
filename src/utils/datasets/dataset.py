import math
from abc import abstractmethod

from easydict import EasyDict
from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils.common import SUPPORTED_SCENARIO_TYPES, get_logger

logger = get_logger(__name__)


class BaseDataset(Dataset):
    """Base class for datasets that handle scenarios."""

    def __init__(self, config: DictConfig):
        """Initializes the BaseDataset with configuration.

        Args:
            config (DictConfig): Configuration for the dataset, including paths and parameters.

        Raises:
            AssertionError: If the scenario type is not supported.
            Exception: If loading scenario information fails.
        """
        super(BaseDataset, self).__init__()

        self.scenario_type = config.scenario_type
        assert (
            self.scenario_type in SUPPORTED_SCENARIO_TYPES
        ), f"Scenario type {self.scenario_type} not supported: {SUPPORTED_SCENARIO_TYPES}"

        self.scenario_base_path = config.scenario_base_path
        self.scenario_meta_path = config.scenario_meta_path

        self.conflict_points_path = config.conflict_points_path
        self.conflict_points_cfg = config.get("conflict_points", None)

        self.parallel = config.get("parallel", True)
        self.batch_size = config.get("batch_size", 4)
        self.step = config.get("step", 1)
        self.num_scenarios = config.get("num_scenarios", -1)
        self.num_workers = config.get("num_workers", 0)
        self.num_shards = config.get("num_shards", 1)
        self.shard_index = config.get("shard_index", 0)

        self.data = EasyDict()
        self.data.scenarios = []
        self.data.scenarios_ids = []
        self.data.conflict_points = []
        self.data.metas = []

        self.load = config.get("load", True)
        if self.load:
            try:
                logger.info("Loading scenario infos...")
                self.load_data()
            except AssertionError as e:
                logger.error("Error loading scenario infos: %s", e)
                raise e

    @property
    def name(self) -> str:
        """Identifies the dataset.

        Returns:
            str: The name of the dataset class and its base path.
        """
        return f"{self.__class__.__name__}\n\t(from: {self.scenario_base_path})"

    def shard(self) -> None:
        """Shards the dataset into smaller parts.

        This is useful for distributed processing or handling large datasets.
        """
        if self.num_shards > 1:
            n_per_shard = math.ceil(len(self.data.metas) / self.num_shards)
            shard_start = int(n_per_shard * self.shard_index)
            shard_end = int(n_per_shard * (self.shard_index + 1))

            self.data.metas = self.data.metas[shard_start:shard_end]
            self.data.scenarios = self.data.scenarios[shard_start:shard_end]
            self.data.scenarios_ids = self.data.scenarios_ids[shard_start:shard_end]

        if self.num_scenarios != -1:
            self.data.metas = self.data.metas[: self.num_scenarios]
            self.data.scenarios = self.data.scenarios[: self.num_scenarios]
            self.data.scenarios_ids = self.data.scenarios_ids[: self.num_scenarios]

    def __len__(self):
        """Returns the number of scenarios in the dataset.

        Returns:
            int: Number of scenarios.
        """
        return len(self.data.scenarios)

    @abstractmethod
    def load_data(self):
        """Loads the dataset.

        This method should load the dataset and populate the data attribute.
        """
        logger.error(
            "Method load_data is not implemented yet."
            "This method should load the dataset and populate the data attribute."
        )

    @abstractmethod
    def collate_batch(self, batch_data) -> dict:
        """Collates a batch of data into a single EasyDict.

        Args:
            batch_data: The batch data to collate.

        Returns:
            Dict: The collated batch.
        """
        logger.error(
            "Method collate_batch is not implemented yet. "
            "This method should collate a batch of data into a single EasyDict."
        )

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Returns the data for the given index.

        Args:
            index (int): The index of the data to retrieve.

        Returns:
            Dict: The data for the given index.
        """
        logger.error(
            "Method __getitem__ is not implemented yet. "
            "This method should return the data for the given index."
        )
