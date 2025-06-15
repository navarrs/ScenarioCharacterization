from abc import ABC, abstractmethod

from omegaconf import DictConfig

from characterization.utils.common import SUPPORTED_SCENARIO_TYPES, get_logger
from characterization.utils.schemas import Scenario

logger = get_logger(__name__)


class BaseVisualizer(ABC):
    def __init__(self, config: DictConfig):
        """
        Initializes the BaseVisualizer with visualization configuration and validates required keys.

        This base class provides a flexible interface for scenario visualizers, supporting custom map and agent color
        schemes, transparency, and scenario type validation. Subclasses should implement scenario-specific visualization
        logic.

        Args:
            config (DictConfig): Configuration for the visualizer, including scenario type, map/agent keys, colors, and
                alpha values.

        Raises:
            AssertionError: If the scenario type or any required configuration key is missing or unsupported.
        """
        self.config = config
        self.scenario_type = config.scenario_type
        if self.scenario_type not in SUPPORTED_SCENARIO_TYPES:
            raise AssertionError(
                f"Scenario type {self.scenario_type} not supported. Supported types are: {SUPPORTED_SCENARIO_TYPES}"
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

    @abstractmethod
    def visualize_scenario(
        self, scenario: Scenario, scores: dict = {}, title: str = "Scenario", output_filepath: str = "temp.png"
    ) -> None:
        """
        Visualizes a single scenario and saves the output to a file.

        This method should be implemented by subclasses to provide scenario-specific visualization, supporting flexible
        titles and output paths. It is designed to handle both static and dynamic map features, as well as agent
        trajectories and attributes.

        Args:
            scenario (dict): The scenario data to visualize.
            title (str, optional): Title for the visualization. Defaults to "Scenario".
            output_filepath (str, optional): Path to save the visualization output. Defaults to "temp.png".

        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement this method.")
