from abc import ABC, abstractmethod

from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils.common import SUPPORTED_SCENARIO_TYPES, get_logger

logger = get_logger(__name__)


class BaseVisualizer(ABC):
    def __init__(self, config: DictConfig, dataset: Dataset):
        """Initializes the BaseVisualizer with configuration.

        Args:
            config (DictConfig): Configuration for the visualizer, including paths and parameters.

        Raises:
            AssertionError: If the scenario type is not supported.
        """
        self.config = config
        self.scenario_type = config.scenario_type
        assert (
            self.scenario_type in SUPPORTED_SCENARIO_TYPES
        ), f"Scenario type {self.scenario_type} not supported: {SUPPORTED_SCENARIO_TYPES}"

        self.static_map_keys = config.get("static_map_keys", None)
        assert self.static_map_keys is not None, "static_map_keys must be provided in the configuration."

        self.dynamic_map_keys = config.get("dynamic_map_keys", None)
        assert self.dynamic_map_keys is not None, "dynamic_map_keys must be provided in the configuration."

        self.map_colors = config.get("map_colors", None)
        assert self.map_colors is not None, "map_colors must be provided in the configuration."

        self.map_alphas = config.get("map_alphas", None)
        assert self.map_alphas is not None, "map_alphas must be provided in the configuration."

        self.agent_colors = config.get("agent_colors", None)
        assert self.agent_colors is not None, "agent_colors must be provided in the configuration."

        self.dataset = dataset

    @abstractmethod
    def visualize_scenario(self, scenario: dict) -> None:
        """Visualizes a single scenario.

        Args:
            scenario (Dict): The scenario data to visualize.

        Returns:
            None: This method should handle visualization and save the output.
        """
        raise NotImplementedError("Subclasses must implement this method.")