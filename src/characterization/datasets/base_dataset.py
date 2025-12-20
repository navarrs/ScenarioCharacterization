import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from characterization.schemas import Scenario
from characterization.utils.common import SUPPORTED_SCENARIO_TYPES
from characterization.utils.geometric_utils import find_closest_lanes, find_conflict_points
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


class BaseDataset(Dataset, ABC):  # pyright: ignore[reportMissingTypeArgument, reportUntypedBaseClass]
    """Base class for datasets that handle scenario data."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseDataset with configuration.

        Args:
            config (DictConfig): Configuration for the dataset, including paths, scenario type,
                sharding, batching, and other parameters.

        Raises:
            ValueError: If the scenario type is not supported.
            Exception: If loading scenario information fails.
        """
        super().__init__()

        self.scenario_type = config.scenario_type
        if self.scenario_type not in SUPPORTED_SCENARIO_TYPES:
            error_message = (
                f"Scenario type {self.scenario_type} not supported. Supported types are: {SUPPORTED_SCENARIO_TYPES}"
            )
            raise ValueError(error_message)

        self.scenario_base_path = Path(config.scenario_base_path)
        assert self.scenario_base_path.exists(), f"Scenario base path {self.scenario_base_path} does not exist."

        self.create_metadata = config.get("create_metadata", True)
        self.conflict_points_path = Path(config.conflict_points_path)
        self.conflict_points_cfg = config.get("conflict_points", None)
        self.closest_lanes_path = Path(config.closest_lanes_path)
        self.closest_lanes_cfg = config.get("closest_lanes", None)

        self.ego_only = config.get("ego_only", False)
        self.parallel = config.get("parallel", True)
        self.batch_size = config.get("batch_size", 4)
        self.step = config.get("step", 1)
        self.num_scenarios = config.get("num_scenarios", -1)
        self.num_workers = config.get("num_workers", 0)
        self.num_shards = config.get("num_shards", 1)
        self.shard_index = config.get("shard_index", 0)
        self.config = config
        self.total_steps = config.get("total_steps", 91)

        self.scenarios: list[str] = []

    @property
    def name(self) -> str:
        """Returns the name and base path of the dataset.

        Returns:
            str: The name of the dataset class and its base path.
        """
        return f"{self.__class__.__name__} (loaded from: {self.scenario_base_path})"

    def compute_metadata(self) -> None:
        """Computes and validates metadata for each scenario in the dataset."""
        if self.num_workers > 1:
            self._compute_metadata_parallel()
        else:
            self._compute_metadata_sequential()

    def _compute_metadata_parallel(self) -> None:
        """Computes and validates metadata for each scenario in the dataset."""
        total = len(self.scenarios)
        logger.info("Checking metadata for %d scenarios using %d workers...", total, self.num_workers)

        def _process(index: int) -> None:
            """Processes a single scenario to validate its metadata.

            Args:
                index (int): Index of the scenario to process.

            Returns:
                None if successful, or an Exception if an error occurred.
            """
            scenario = self.load_scenario_information(index)
            if scenario is None:
                error_message = f"Scenario file for index {index} could not be loaded."
                logger.warning(error_message)
                return

            scenario = self.transform_scenario_data(scenario)
            if scenario.static_map_data is None:
                return

            # Add conflict point and closest lane information
            self.compute_conflict_point_information(scenario)
            self.compute_closest_lanes_information(scenario)
            return

        Parallel(n_jobs=self.num_workers, batch_size=self.batch_size)(
            delayed(_process)(index) for index in tqdm(range(total), desc="Computing metadata", total=total)
        )

    def _compute_metadata_sequential(self) -> None:
        """Computes metadata sequentially for each scenario in the dataset."""
        total = len(self.scenarios)
        for i in tqdm(range(total), desc="Computing metadata"):
            # Load scenario
            scenario = self.load_scenario_information(i)
            if scenario is None:
                error_message = f"Scenario file for index {i} could not be loaded."
                raise ValueError(error_message)

            scenario = self.transform_scenario_data(scenario)
            if scenario.static_map_data is None:
                continue

            # Add conflict point and closest lane information
            self.compute_conflict_point_information(scenario)
            self.compute_closest_lanes_information(scenario)

    def compute_conflict_point_information(self, scenario: Scenario) -> None:
        """Computes and stores conflict points for a given scenario.

        Args:
            scenario (Scenario): The scenario to retrieve conflict points for.
        """
        conflict_points_filepath = self.conflict_points_path / f"{scenario.metadata.scenario_id}.pkl"
        if conflict_points_filepath.exists() and conflict_points_filepath.stat().st_size > 0:
            return

        # Safely read cfg values (cfg may be None)
        conflict_point_info = find_conflict_points(
            scenario,
            resample_factor=self.conflict_points_cfg.get("resample_factor", 1),
            intersection_threshold=self.conflict_points_cfg.get("intersection_threshold", 0.5),
            return_static_conflict_points=self.conflict_points_cfg.get("return_static_conflict_points", False),
            return_lane_conflict_points=self.conflict_points_cfg.get("return_lane_conflict_points", False),
            return_dynamic_conflict_points=self.conflict_points_cfg.get("return_dynamic_conflict_points", False),
        )
        if conflict_point_info is not None:
            # ensure directory exists before writing
            conflict_points_filepath.parent.mkdir(parents=True, exist_ok=True)
            with conflict_points_filepath.open("wb") as f:
                pickle.dump(conflict_point_info, f)  # nosec B301

    def load_conflict_point_information(self, scenario_id: str) -> dict[str, Any] | None:
        """Retrieves conflict points for a given scenario.

        Args:
            scenario_id (str): The scenario ID to retrieve conflict points for.

        Returns:
            conflict_point_info (dict[str, Any] | None): The conflict points associated with the scenario or 'None' if
            not found.
        """
        conflict_points_filepath = self.conflict_points_path / f"{scenario_id}.pkl"
        if conflict_points_filepath.exists() and conflict_points_filepath.stat().st_size > 0:
            try:
                with conflict_points_filepath.open("rb") as f:
                    return pickle.load(f)  # nosec B301
            except (EOFError, pickle.UnpicklingError) as e:
                logger.warning(
                    "Failed to load conflict points from %s (will recompute): %s",
                    conflict_points_filepath,
                    e,
                )
        return None

    def compute_closest_lanes_information(self, scenario: Scenario) -> dict[str, Any] | None:
        """Computes and stores closest lanes for a given scenario.

        Args:
            scenario (Scenario): The scenario to compute closest lanes for.
        """
        closest_lanes_filepath = self.closest_lanes_path / f"{scenario.metadata.scenario_id}.pkl"
        if closest_lanes_filepath.exists() and closest_lanes_filepath.stat().st_size > 0:
            return

        closest_lanes_info = find_closest_lanes(
            scenario,
            k_closest=self.closest_lanes_cfg.get("num_lanes", 16),
            threshold_distance=self.closest_lanes_cfg.get("threshold_distance", 10.0),
            subsample_factor=self.closest_lanes_cfg.get("subsample_factor", 2),
        )
        if closest_lanes_info is not None:
            with closest_lanes_filepath.open("wb") as f:
                pickle.dump(closest_lanes_info, f)  # nosec B301

    def load_closest_lanes_information(self, scenario: Scenario) -> dict[str, Any] | None:
        """Retrieves closest lanes for a given scenario.

        Args:
            scenario (Scenario): The scenario to retrieve closest lanes for.

        Returns:
            closest_lanes_info (dict[str, Any] | None): The closest lanes associated with the scenario or 'None' if
            not found.
        """
        closest_lanes_filepath = self.closest_lanes_path / f"{scenario.metadata.scenario_id}.pkl"
        if closest_lanes_filepath.exists() and closest_lanes_filepath.stat().st_size > 0:
            try:
                with closest_lanes_filepath.open("rb") as f:
                    return pickle.load(f)  # nosec B301
            except (EOFError, pickle.UnpicklingError) as e:
                logger.warning(
                    "Failed to load closest lanes from %s (will recompute): %s",
                    closest_lanes_filepath,
                    e,
                )
        return None

    def __len__(self) -> int:
        """Returns the number of scenarios in the dataset.

        Returns:
            int: The number of scenarios in the dataset.
        """
        return len(self.scenarios)

    def __getitem__(self, index: int) -> Scenario:
        """Retrieves a single scenario by index.

        Args:
            index (int): Index of the scenario to retrieve.

        Returns:
            Scenario: A Scenario object constructed from the scenario data.

        Raises:
            ValidationError: If the scenario data does not pass schema validation.
        """
        # Load scenario
        scenario = self.load_scenario_information(index)
        if scenario is None:
            error_message = f"Scenario information for index {index} is missing or invalid."
            raise ValueError(error_message)

        scenario = self.transform_scenario_data(scenario)
        if scenario.static_map_data is None:
            return scenario

        # Add conflict point information to the scenario
        conflict_points_info = self.load_conflict_point_information(scenario.metadata.scenario_id)
        agent_distances_to_conflict_points, conflict_points = None, None
        if conflict_points_info is not None:
            agent_distances_to_conflict_points = (
                None
                if conflict_points_info["agent_distances_to_conflict_points"] is None
                else conflict_points_info["agent_distances_to_conflict_points"][:, : self.total_steps, :]
            )
            conflict_points = (
                None
                if conflict_points_info["all_conflict_points"] is None
                else conflict_points_info["all_conflict_points"]
            )
            scenario.static_map_data.map_conflict_points = conflict_points
            scenario.static_map_data.agent_distances_to_conflict_points = agent_distances_to_conflict_points

        # Add closest lane information to the scenario
        closest_lanes_info = self.load_closest_lanes_information(scenario)
        if closest_lanes_info is not None:
            agent_closest_lanes = closest_lanes_info["agent_closest_lanes"][:, : self.total_steps, :, :]
            scenario.static_map_data.agent_closest_lanes = agent_closest_lanes

        return scenario

    @abstractmethod
    def transform_scenario_data(self, scenario_data: dict[str, Any]) -> Scenario:
        """Transforms scenario data and conflict points into a model-ready format.

        Args:
            scenario_data (dict): The scenario data to transform.
            conflict_points_info (dict): Conflict points associated with the scenario.

        Returns:
            dict: Transformed scenario data.
        """

    @abstractmethod
    def load_data(self) -> None:
        """Loads the dataset and populates the data attribute.

        This method should be implemented by subclasses to load all required data.
        """

    @abstractmethod
    def collate_batch(self, batch_data: dict[str, Any]) -> dict[str, dict[str, Any]]:  # pyright: ignore[reportMissingParameterType]
        """Collates a batch of data into a single dictionary.

        Args:
            batch_data: The batch data to collate.

        Returns:
            dict: The collated batch.
        """

    @abstractmethod
    def load_scenario_information(self, index: int) -> dict[str, dict[str, Any]] | None:
        """Loads scenario information for a given index.

        Args:
            index (int): The index of the scenario to load.

        Returns:
            dict: The loaded scenario information.
        """
