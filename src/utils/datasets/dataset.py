import numpy as np

from abc import abstractmethod
from easydict import EasyDict
from omegaconf import DictConfig

from utils.common import SUPPORTED_SCENARIO_TYPES

# TODO: Implement this as a dataloader class
class BaseDataset:
    def __init__(self, config: DictConfig):
        super().__init__()

        self.scenario_type = config.scenario_type
        assert (
            self.scenario_type in SUPPORTED_SCENARIO_TYPES
        ), f"Scenario type {self.scenario_type} not supported"

        self.scenario_base_path = config.scenario_base_path
        self.scenario_meta_path = config.scenario_meta_path
        self.step = config.step if "step" in config else 1
        self.num_scenarios = config.num_scenarios if "num_scenarios" in config else -1
        self.num_processes = config.num_processes if "num_processes" in config else 10
        self.num_shards = config.num_shards if "num_shards" in config else 10
        self.shard_idx = config.shard_idx if "shard_idx" in config else 0

        self.data = EasyDict()
        self.data.scenarios = []
        self.data.scenarios_ids = []
        self.data.metas = []

    def name(self) -> str:
        """
        Identify the dataset.
        """
        return f"{self.__class__.__name__}\n\t(from: {self.scenario_base_path})"

    def shard(self) -> None:
        """
        Shard the dataset into smaller parts.
        This is useful for distributed processing or handling large datasets.
        """
        if self.num_shards > 1:
            n_per_shard = np.ceil(len(self.data.metas) / self.num_shards)
            shard_start = int(n_per_shard * self.shard_idx)
            shard_end = int(n_per_shard * (self.shard_idx + 1))

            self.data.metas = self.data.metas[shard_start:shard_end]
            self.data.scenarios = self.data.scenarios[shard_start:shard_end]
            self.data.scenarios_ids = self.data.scenarios_ids[shard_start:shard_end]

        if self.num_scenarios != -1:
            self.data.metas = self.data.metas[: self.num_scenarios]
            self.data.scenarios = self.data.scenarios[: self.num_scenarios]
            self.data.scenarios_ids = self.data.scenarios_ids[: self.num_scenarios]

    def __len__(self):
        return len(self.data.scenarios)

    @abstractmethod
    def load_data(self):
        """Load the dataset."""
        raise NotImplementedError("Method load_data is not implemented yet.")
