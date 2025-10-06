import itertools
import math
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from scipy.signal import resample
from torch.utils.data import Dataset

from characterization.schemas import Scenario
from characterization.utils.common import SUPPORTED_SCENARIO_TYPES, AgentTrajectoryMasker
from characterization.utils.geometric_utils import compute_dists_to_conflict_points
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
        self.config = config
        self.total_steps = config.get("total_steps", 91)

        self.data = DictConfig(
            {
                "scenarios": [],
                "scenarios_ids": [],
                "metas": [],
            },
        )

    @property
    def name(self) -> str:
        """Returns the name and base path of the dataset.

        Returns:
            str: The name of the dataset class and its base path.
        """
        return f"{self.__class__.__name__} (loaded from: {self.scenario_base_path})"

    def shard(self) -> None:
        """Shards the dataset into smaller parts for distributed or parallel processing.

        This method updates the internal data attributes to only include the shard assigned
        to this instance, based on the number of shards and the shard index.
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

    def find_conflict_points(self, scenario: Scenario, ndim: int = 3) -> dict[str, Any] | None:
        """Finds the conflict points in the map for a scenario.

        Args:
            scenario (Scenario): The scenario for which to find conflict points.
            ndim (int): Number of dimensions to consider (2 or 3). Defaults to 3.

        Returns:
            dict: The conflict points in the map, including:
                - 'static': Static conflict points (e.g., crosswalks, speed bumps, stop signs).
                - 'dynamic': Dynamic conflict points (e.g., traffic lights).
                - 'lane_intersections': Lane intersection points.
                - 'all_conflict_points': All conflict points concatenated.
                - 'agent_distances_to_conflict_points': Distances from each agent to each conflict point.
        """
        if scenario.static_map_data is None:
            return None

        # polylines = static_map_info["all_polylines"]
        polylines = scenario.static_map_data.map_polylines
        if polylines is None or polylines.shape[0] == 0:
            return None

        agent_trajectories = AgentTrajectoryMasker(scenario.agent_data.agent_trajectories)
        agent_positions = agent_trajectories.agent_xyz_pos

        # Static Conflict Points: Crosswalks, Speed Bumps and Stop Signs
        static_conflict_points_list = []
        crosswalks_idxs = scenario.static_map_data.crosswalk_polyline_idxs
        speed_bumps_idxs = scenario.static_map_data.speed_bump_polyline_idxs
        stop_signs_idxs = scenario.static_map_data.stop_sign_polyline_idxs
        conflict_idxs = np.concatenate(
            [
                crosswalks_idxs if crosswalks_idxs is not None else np.empty((0, 2), dtype=int),
                speed_bumps_idxs if speed_bumps_idxs is not None else np.empty((0, 2), dtype=int),
                stop_signs_idxs if stop_signs_idxs is not None else np.empty((0, 2), dtype=int),
            ],
            axis=0,
        )
        for start, end in conflict_idxs:
            points = polylines[start:end][:, :ndim]
            points = resample(points, points.shape[0] * self.conflict_points_cfg.resample_factor)
            static_conflict_points_list.append(points)
        static_conflict_points = (
            np.concatenate(static_conflict_points_list) if len(static_conflict_points_list) > 0 else np.empty((0, ndim))
        )

        # Lane Intersections
        lane_intersections_list = []
        lane_idxs = scenario.static_map_data.lane_polyline_idxs
        if lane_idxs is not None:
            num_lanes = len(lane_idxs)

            lane_combinations = list(itertools.combinations(range(num_lanes), 2))
            for i, j in lane_combinations:
                lane_i_idxs, lane_j_idxs = lane_idxs[i], lane_idxs[j]
                lane_i = polylines[lane_i_idxs[0] : lane_i_idxs[1]][:, :ndim]
                lane_j = polylines[lane_j_idxs[0] : lane_j_idxs[1]][:, :ndim]

                dists_ij = np.linalg.norm(lane_i[:, None] - lane_j, axis=-1)

                i_idx, j_idx = np.where(dists_ij < self.conflict_points_cfg.intersection_threshold)
                i_idx, j_idx = np.unique(i_idx), np.unique(j_idx)

                # TODO: determine if two lanes are consecutive, but not entry/exit lanes. If this is the
                # case there'll be an intersection that is not a conflict point.
                # start_i, end_i = i_idx[:min_timesteps], i_idx[-min_timesteps:]
                # start_j, end_j = j_idx[:min_timesteps], j_idx[-min_timesteps:]
                # if (np.any(start_i < min_timesteps) and np.any(end_j > lane_j.shape[0] - min_timesteps)) or (
                #     np.any(start_j < min_timesteps) and np.any(end_i > lane_i.shape[0] - min_timesteps)
                # ):
                #     lanes_i_ee = lane_infos[i]["entry_lanes"] + lane_infos[i]["exit_lanes"]
                #     lanes_j_ee = lane_infos[j]["entry_lanes"] + lane_infos[j]["exit_lanes"]
                #     if j not in lanes_i_ee and i not in lanes_j_ee:
                #         continue

                if i_idx.shape[0] > 0:
                    lane_intersections_list.append(lane_i[i_idx])

                if j_idx.shape[0] > 0:
                    lane_intersections_list.append(lane_j[j_idx])

        lane_intersections = (
            np.concatenate(lane_intersections_list) if len(lane_intersections_list) > 0 else np.empty((0, 3))
        )

        # Dynamic Conflict Points: Traffic Lights
        dynamic_conflict_points = np.empty((0, ndim))
        if scenario.dynamic_map_data is not None:
            stops = (
                scenario.dynamic_map_data.stop_points
                if scenario.dynamic_map_data.stop_points is not None
                else np.empty((0, ndim))
            )
            if len(stops) > 0 and len(stops[0]) > 0 and stops[0].shape[1] == ndim:
                dynamic_conflict_points = np.concatenate(stops[0])

        # Concatenate all conflict points into a single array if they are not empty
        conflict_point_list = []
        if static_conflict_points.shape[0] > 0:
            conflict_point_list.append(static_conflict_points)
        if dynamic_conflict_points.shape[0] > 0:
            conflict_point_list.append(dynamic_conflict_points)
        if lane_intersections.shape[0] > 0:
            conflict_point_list.append(lane_intersections)

        conflict_points = np.concatenate(conflict_point_list, dtype=np.float32) if conflict_point_list else None

        dists_to_conflict_points = (
            compute_dists_to_conflict_points(conflict_points, agent_positions) if conflict_points is not None else None
        )

        return {
            "static": static_conflict_points,
            "dynamic": dynamic_conflict_points,
            "lane_intersections": lane_intersections,
            "all_conflict_points": conflict_points,
            "agent_distances_to_conflict_points": dists_to_conflict_points,
        }

    def get_conflict_point_info(self, scenario: Scenario) -> dict[str, Any] | None:
        """Retrieves conflict points for a given scenario.

        Args:
            scenario (Scenario): The scenario to retrieve conflict points for.

        Returns:
            Any: The conflict points associated with the scenario.
        """
        conflict_points_filepath = Path(self.conflict_points_path) / f"{scenario.metadata.scenario_id}.pkl"
        if conflict_points_filepath.exists():
            with conflict_points_filepath.open("rb") as f:
                return pickle.load(f)  # nosec B301

        conflict_point_info = self.find_conflict_points(scenario)
        if conflict_point_info is not None:
            with conflict_points_filepath.open("wb") as f:
                pickle.dump(conflict_point_info, f)  # nosec B301
        return conflict_point_info

    def __len__(self) -> int:
        """Returns the number of scenarios in the dataset.

        Returns:
            int: The number of scenarios in the dataset.
        """
        return len(self.data.scenarios)

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

        # Add conflict point information to the scenario
        conflict_points_data = self.get_conflict_point_info(scenario)

        agent_distances_to_conflict_points, conflict_points = None, None
        if conflict_points_data is not None:
            agent_distances_to_conflict_points = (
                None
                if conflict_points_data["agent_distances_to_conflict_points"] is None
                else conflict_points_data["agent_distances_to_conflict_points"][:, : self.total_steps, :]
            )
            conflict_points = (
                None
                if conflict_points_data["all_conflict_points"] is None
                else conflict_points_data["all_conflict_points"]
            )
        if scenario.static_map_data is not None:
            scenario.static_map_data.map_conflict_points = conflict_points
            scenario.static_map_data.agent_distances_to_conflict_points = agent_distances_to_conflict_points

        return scenario

    @abstractmethod
    def transform_scenario_data(self, scenario_data: dict[str, Any]) -> Scenario:
        """Transforms scenario data and conflict points into a model-ready format.

        Args:
            scenario_data (dict): The scenario data to transform.
            conflict_points_data (dict): Conflict points associated with the scenario.

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
