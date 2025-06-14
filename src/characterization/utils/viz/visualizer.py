from abc import ABC, abstractmethod

from omegaconf import DictConfig
from torch.utils.data import Dataset

from characterization.utils.common import SUPPORTED_SCENARIO_TYPES, get_logger

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
        if self.scenario_type not in SUPPORTED_SCENARIO_TYPES:
            raise AssertionError(
                f"Scenario type {self.scenario_type} not supported. " f"Supported types are: {SUPPORTED_SCENARIO_TYPES}"
            )

        self.static_map_keys = config.get("static_map_keys", None)
        if self.static_map_keys is None:
            raise AssertionError("static_map_keys must be provided in the configuration.")

        self.dynamic_map_keys = config.get("dynamic_map_keys", None)
        if self.dynamic_map_keys is None:
            raise AssertionError("dynamic_map_keys must be provided in the configuration.")

        self.map_colors = config.get("map_colors", None)
        if self.map_colors is None:
            raise AssertionError("map_colors must be provided in the configuration.")

        self.map_alphas = config.get("map_alphas", None)
        if self.map_alphas is None:
            raise AssertionError("map_alphas must be provided in the configuration.")

        self.agent_colors = config.get("agent_colors", None)
        if self.agent_colors is None:
            raise AssertionError("agent_colors must be provided in the configuration.")

        self.dataset = dataset

    @abstractmethod
    def visualize_scenario(self, scenario: dict, title: str = "Scenario", output_filepath: str = "temp.png") -> None:
        """Visualizes a single scenario.

        Args:
            scenario (Dict): The scenario data to visualize.

        Returns:
            None: This method should handle visualization and save the output.
        """
        raise NotImplementedError("Subclasses must implement this method.")
