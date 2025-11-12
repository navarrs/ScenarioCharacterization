import pickle  # nosec B403
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray
from omegaconf import DictConfig

from characterization.datasets.dataset import BaseDataset
from characterization.schemas.scenario import AgentData, Scenario, ScenarioMetadata
from characterization.utils import common
from characterization.utils.io_utils import get_logger
from characterization.utils.scenario_types import AMELIA_VALUE_TO_AGENT_TYPE, AgentType

logger = get_logger(__name__)


class AmeliaDataset(BaseDataset):
    """Class to handle the Amelia dataset."""

    # Speed, Heading, Lat, Lon, Range, Bearing, x, y, z
    _TRAJECTORY_SPEED: ClassVar[list[bool]] = [True, False, False, False, False, False, False, False, False]
    _TRAJECTORY_HEADING: ClassVar[list[bool]] = [False, True, False, False, False, False, False, False, False]
    _TRAJECTORY_XY: ClassVar[list[bool]] = [False, False, False, False, False, False, True, True, False]
    _TRAJECTORY_Z: ClassVar[list[bool]] = [False, False, False, False, False, False, False, False, True]

    def __init__(self, config: DictConfig) -> None:
        """Initializes the Amelia dataset handler."""
        super().__init__(config=config)
        # Three challengingness levels: 0 (easy), 1 (medium), 2 (hard) obtained from Amelia
        self.DIFFICULTY_WEIGHTS = {0: 0.8, 1: 0.9, 2: 1.0}

        self.LAST_TIMESTEP = 60
        self.HIST_TIMESTEP = 10

        self.LAST_TIMESTEP_TO_CONSIDER = {
            "gt": self.LAST_TIMESTEP,
            "ho": self.HIST_TIMESTEP,
        }

        self.airports = config.airports
        self.data = DictConfig(
            {
                "scenarios": [],
                "airports": [],
            },
        )

        self.frequency_hz = 1.0  # 1 Hz frequency for Amelia dataset, specified here: https://arxiv.org/pdf/2407.21185
        self.timestamps = np.arange(0, self.LAST_TIMESTEP, 1 / self.frequency_hz)

        self.base_conflict_points_path = Path(config.conflict_points_path)
        self.base_closest_lanes_path = Path(config.closest_lanes_path)

        self.num_relevant_agents = config.get("num_relevant_agents", 5)
        self.ego_selection_strategy = config.get("ego_selection_strategy", "critical")

        self.generator = default_rng(config.get("seed", 42))

        self.load = config.get("load", True)
        if self.load:
            try:
                logger.info("Loading scenario infos...")
                self.load_data()
            except AssertionError:
                logger.exception("Error loading scenario infos")
                raise

    def load_data(self) -> None:
        """Loads the Amelia dataset and scenario metadata.

        Loads scenario metadata and scenario file paths, applies sharding if enabled,
        and checks that the number of scenarios matches the number of conflict points.

        Raises:
            AssertionError: If the number of scenarios and conflict points do not match.
        """
        start = time.time()
        logger.info("Loading Amelia scenario base data from %s", self.scenario_base_path)
        for airport in self.airports:
            scenarios_path = self.scenario_base_path / airport
            scenario_files = list(scenarios_path.glob("**/*.pkl"))
            if not scenario_files:
                logger.debug("No scenarios found under %s", scenarios_path)
                continue

            # Randomize order of files
            self.generator.shuffle(scenario_files)
            if self.num_scenarios is not None and self.num_scenarios > 0:
                scenario_files = scenario_files[: self.num_scenarios]

            # Store file paths and airport info
            self.data.scenarios.extend(scenario_files)
            self.data.airports.extend([airport] * len(scenario_files))

        logger.info("Loading data took %2f seconds.", time.time() - start)

    def repack_agent_data(
        self,
        agent_sequences: NDArray[np.float32],
        agent_ids: list[int],
        agent_types: list[int],
        agent_masks: NDArray[np.bool_],
    ) -> AgentData:
        """Packs agent information from Waymo format to AgentData format.

        Args:
            agent_sequences (NDArray[np.float32]): array containing agent sequences.
            agent_ids (list[int]): list of agent IDs.
            agent_types (list[int]): list of agent types.
            agent_masks (NDArray[np.bool_]): array of agent masks.

        Returns:
            AgentData: pydantic validator encapsulating agent information.
        """
        num_agents, num_timesteps, _ = agent_sequences.shape
        trajectories = np.full((num_agents, num_timesteps, 10), np.nan, dtype=np.float32)

        last_timestep = self.LAST_TIMESTEP_TO_CONSIDER[self.scenario_type]
        if num_timesteps < last_timestep:
            error_message = (
                f"Scenario has only {num_timesteps} timesteps, but expected at least {last_timestep} timesteps."
            )
            raise AssertionError(error_message)

        self.total_steps = last_timestep

        # Fill in the trajectory information
        # IDXs 0 to 2: x, y, z (need to be converted to meters)
        trajectories[..., :2] = common.km_to_m(agent_sequences[..., self._TRAJECTORY_XY])
        trajectories[..., 2] = common.feet_to_m(agent_sequences[..., self._TRAJECTORY_Z]).squeeze(-1)
        # IDX 3 to 5: length, width, height are not available in Amelia, we leave them as NaN
        # IDX 6: heading (need to be converted to radians)
        trajectories[..., 6] = common.deg_to_rad(agent_sequences[..., self._TRAJECTORY_HEADING]).squeeze(-1)
        # IDX 7 to 8: are the velocity components in m/s
        speed_ms = common.knots_to_ms(agent_sequences[..., self._TRAJECTORY_SPEED]).squeeze(-1)
        trajectories[..., 7] = speed_ms * np.cos(trajectories[..., 6])
        trajectories[..., 8] = speed_ms * np.sin(trajectories[..., 6])
        # IDX 9: flag indicating if the information is valid at that timestep
        trajectories[..., 9] = agent_masks.astype(np.float32)

        trajectories = trajectories[:, :last_timestep, :]  # shape: [num_agents, last_timestep, dim]
        object_types = [AgentType[AMELIA_VALUE_TO_AGENT_TYPE[n]] for n in agent_types]
        return AgentData(agent_ids=agent_ids, agent_types=object_types, agent_trajectories=trajectories)

    # @staticmethod
    # def get_polyline_ids(polyline: dict[str, Any], key: str) -> np.ndarray:
    #     """Extracts polyline indices from the polyline dictionary."""
    #     return np.array([value["id"] for value in polyline[key]], dtype=np.int32)

    # @staticmethod
    # def get_speed_limit_mph(polyline: dict[str, Any], key: str) -> np.ndarray:
    #     """Extracts speed limit in mph from the polyline dictionary."""
    #     return np.array([value["speed_limit_mph"] for value in polyline[key]], dtype=np.float32)

    # @staticmethod
    # def get_polyline_idxs(polyline: dict[str, Any], key: str) -> np.ndarray | None:
    #     """Extracts polyline start and end indices from the polyline dictionary."""
    #     polyline_idxs = np.array(
    #         [[value["polyline_index"][0], value["polyline_index"][1]] for value in polyline[key]],
    #         dtype=np.int32,
    #     )

    #     if polyline_idxs.shape[0] == 0:
    #         return None
    #     return polyline_idxs

    # def repack_static_map_data(self, static_map_data: dict[str, Any] | None) -> StaticMapData | None:
    #     """Packs static map information from Waymo format to StaticMapData format.

    #     Args:
    #         static_map_data (dict): dictionary containing Waymo static scenario data:
    #             'all_polylines': all road data in the form of polyline mapped by type to specific road types.

    #     Returns:
    #         StaticMapData: pydantic validator encapsulating static map information.
    #     """
    #     if static_map_data is None:
    #         return None

    #     map_polylines = static_map_data["all_polylines"].astype(np.float32)  # shape: [N, 3] or [N, 3, 2]

    #     return StaticMapData(
    #         map_polylines=map_polylines,
    #         lane_ids=WaymoDataset.get_polyline_ids(static_map_data, "lane") if "lane" in static_map_data else None,
    #         lane_speed_limits_mph=WaymoDataset.get_speed_limit_mph(static_map_data, "lane")
    #         if "lane" in static_map_data
    #         else None,
    #         lane_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "lane")
    #         if "lane" in static_map_data
    #         else None,
    #         road_line_ids=WaymoDataset.get_polyline_ids(static_map_data, "road_line")
    #         if "road_line" in static_map_data
    #         else None,
    #         road_line_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "road_line")
    #         if "road_line" in static_map_data
    #         else None,
    #         road_edge_ids=WaymoDataset.get_polyline_ids(static_map_data, "road_edge")
    #         if "road_edge" in static_map_data
    #         else None,
    #         road_edge_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "road_edge")
    #         if "road_edge" in static_map_data
    #         else None,
    #         crosswalk_ids=WaymoDataset.get_polyline_ids(static_map_data, "crosswalk")
    #         if "crosswalk" in static_map_data
    #         else None,
    #         crosswalk_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "crosswalk")
    #         if "crosswalk" in static_map_data
    #         else None,
    #         speed_bump_ids=WaymoDataset.get_polyline_ids(static_map_data, "speed_bump")
    #         if "speed_bump" in static_map_data
    #         else None,
    #         speed_bump_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "speed_bump")
    #         if "speed_bump" in static_map_data
    #         else None,
    #         stop_sign_ids=WaymoDataset.get_polyline_ids(static_map_data, "stop_sign")
    #         if "stop_sign" in static_map_data
    #         else None,
    #         stop_sign_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "stop_sign")
    #         if "stop_sign" in static_map_data
    #         else None,
    #         stop_sign_lane_ids=[
    #             stop_sign["lane_ids"] for stop_sign in static_map_data.get("stop_sign", {"lane_ids": []})
    #         ],
    #     )

    # def repack_dynamic_map_data(self, dynamic_map_data: dict[str, Any]) -> DynamicMapData:
    #     """Packs dynamic map information from Waymo format to DynamicMapData format.

    #     Args:
    #         dynamic_map_data (dict): dictionary containing Waymo dynamic scenario data:
    #             'stop_points': traffic light stopping points.
    #             'lane_id': IDs of the lanes where the traffic light is.
    #             'state': state of the traffic light (e.g., red, etc).

    #     Returns:
    #         DynamicMapData: pydantic validator encapsulating static map information.
    #     """
    #     stop_points = dynamic_map_data["stop_point"][: self.total_steps]
    #     lane_id = [lid.astype(np.int64) for lid in dynamic_map_data["lane_id"][: self.total_steps]]
    #     states = dynamic_map_data["state"][: self.total_steps]
    #     num_dynamic_stop_points = len(stop_points)

    #     if num_dynamic_stop_points == 0:
    #         stop_points = None
    #         lane_id = None
    #         states = None

    #     return DynamicMapData(stop_points=stop_points, lane_ids=lane_id, states=states)

    def transform_scenario_data(self, scenario_data: dict[str, Any]) -> Scenario:
        """Transforms raw scenario data into the standardized Scenario format.

        Args:
            scenario_data (dict): Raw scenario data containing:
                - 'track_infos': Agent trajectories and metadata.
                - 'map_infos': Static map information.
                - 'dynamic_map_infos': Dynamic map information.
                - 'timestamps_seconds': Timestamps for each timestep.
                - 'sdc_track_index': Index of the ego vehicle.
                - 'tracks_to_predict': List of tracks to predict with their difficulty levels.
                - 'scenario_id': Unique identifier for the scenario.
                - 'current_time_index': Current time index in the scenario.
                - 'objects_of_interest': List of object IDs that are of interest in the scenario.
            conflict_points_data (dict, optional): Precomputed conflict point data containing:
                - 'agent_distances_to_conflict_points': Distances from each agent to each conflict point.
                - 'all_conflict_points': All conflict points in the scenario.
        """
        # Repack agent information from input scenario
        agent_data = self.repack_agent_data(
            scenario_data["agent_sequences"],
            scenario_data["agent_ids"],
            scenario_data["agent_types"],
            scenario_data["agent_masks"],
        )

        # TODO: Incorporate map data
        # static_map_data = self.repack_static_map_data(scenario_data["map_infos"])

        agent_order = scenario_data["meta"]["agent_order"][self.ego_selection_strategy]
        agent_relevance = np.zeros(agent_data.num_agents, dtype=np.float32)
        relevant_idxs = agent_order[: self.num_relevant_agents]
        agent_relevance[relevant_idxs] = 1.0
        ego_agent_index = self.generator.choice(relevant_idxs)

        # Repack meta information
        metadata = ScenarioMetadata(
            scenario_id=scenario_data["scenario_id"],
            timestamps_seconds=self.timestamps[: self.total_steps].tolist(),
            frequency_hz=self.frequency_hz,
            current_time_index=self.HIST_TIMESTEP,
            ego_vehicle_id=agent_data.agent_ids[ego_agent_index],
            ego_vehicle_index=ego_agent_index,
            track_length=self.total_steps,
            objects_of_interest=scenario_data["objects_of_interest"],
            dataset=f"amelia-{scenario_data['airport_id']}",
            # Thresholds adapted from: AmeliaScenes/blob/main/amelia_scenes/scoring/interactive.py
            agent_to_agent_max_distance=4000.0,  # m, runway extent (10,000-13,000ft)
            agent_to_conflict_point_max_distance=50.0,  # m, distance to a hold-line
            agent_to_agent_distance_breach=300.0,  # m, separation standards: airservicesaustralia.com
        )

        return Scenario(metadata=metadata, agent_data=agent_data, static_map_data=None)

    def load_scenario_information(self, index: int) -> dict[str, dict[str, Any]] | None:
        """Loads scenario and conflict point information by index.

        Args:
            index (int): Index of the scenario to load.

        Returns:
            dict: A dictionary containing the scenario and conflict points.

        Raises:
            ValidationError: If the scenario data does not pass schema validation.
        """
        scenario_filepath = Path(self.data.scenarios[index])
        if not scenario_filepath.exists():
            return None

        # Update the cache paths
        airport = self.data.airports[index]
        self.conflict_points_path = self.base_conflict_points_path / airport
        self.closest_lanes_path = self.base_closest_lanes_path / airport

        with scenario_filepath.open("rb") as f:
            return pickle.load(f)  # nosec B301

    def collate_batch(self, batch_data: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportMissingParameterType]
        """Collates a batch of scenario data for processing.

        Args:
            batch_data (list): List of scenario data dictionaries.

        Returns:
            dict: A dictionary containing the batch size and the batch of scenarios.
        """
        batch_size = len(batch_data)
        return {"batch_size": batch_size, "scenario": batch_data}
