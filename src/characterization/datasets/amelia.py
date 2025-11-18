import pickle  # nosec B403
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray
from omegaconf import DictConfig

from characterization.datasets.dataset import BaseDataset
from characterization.schemas.scenario import AgentData, Scenario, ScenarioMetadata, StaticMapData
from characterization.utils import common, geometric_utils
from characterization.utils.io_utils import get_logger
from characterization.utils.scenario_types import (
    AMELIA_VALUE_TO_AGENT_TYPE,
    AMELIA_VALUE_TO_MAP_TYPE,
    AgentType,
    PolylineType,
)

logger = get_logger(__name__)


class AmeliaDataset(BaseDataset):
    """Class to handle the Amelia dataset."""

    # Speed, Heading, Lat, Lon, Range, Bearing, x, y, z
    _TRAJECTORY_SPEED: ClassVar[list[bool]] = [True, False, False, False, False, False, False, False, False]
    _TRAJECTORY_HEADING: ClassVar[list[bool]] = [False, True, False, False, False, False, False, False, False]
    _TRAJECTORY_XY: ClassVar[list[bool]] = [False, False, False, False, False, False, True, True, False]
    _TRAJECTORY_Z: ClassVar[list[bool]] = [False, False, False, False, False, False, False, False, True]

    # Lat Start, Lon Start, X start, Y start, Lat End, Lon End, X end, Y end, Class, ID
    _MAP_XY_START: ClassVar[list[bool]] = [False, False, True, True, False, False, False, False, False, False]
    _MAP_XY_END: ClassVar[list[bool]] = [False, False, False, False, False, False, True, True, False, False]
    _MAP_CLASS: ClassVar[list[bool]] = [False, False, False, False, False, False, False, False, True, False]

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
        self.airport_maps = {}

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

            # Load airport map
            map_path = self.scenario_meta_path / airport / "semantic_graph.pkl"
            with map_path.open("rb") as f:
                airport_map = pickle.load(f)  # nosec B301
            self.airport_maps[airport] = self.repack_static_map_data(airport_map)

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

    @staticmethod
    def get_polyline_ids(polyline: dict[str, Any], key: str) -> NDArray[np.int32]:
        """Extracts polyline indices from the polyline dictionary."""
        return np.array([value["id"] for value in polyline[key]], dtype=np.int32)

    @staticmethod
    def get_polyline_idxs(polyline: dict[str, Any], key: str) -> NDArray[np.int32] | None:
        """Extracts polyline start and end indices from the polyline dictionary."""
        polyline_idxs = np.array(
            [[value["polyline_index"], value["polyline_index"] + 1] for value in polyline[key]],
            dtype=np.int32,
        )

        if polyline_idxs.shape[0] == 0:
            return None
        return polyline_idxs

    def repack_static_map_data(self, static_map_data: dict[str, Any] | None) -> StaticMapData | None:
        """Packs static map information from Waymo format to StaticMapData format.

        Args:
            static_map_data (dict): dictionary containing Waymo static scenario data:
                'all_polylines': all road data in the form of polyline mapped by type to specific road types.

        Returns:
            StaticMapData: pydantic validator encapsulating static map information.
        """
        if static_map_data is None:
            return None
        map_data = static_map_data.get("map_infos", None)
        if map_data is None:
            return None

        all_polylines = map_data["all_polylines"].astype(np.float32)

        # Fill in the map information
        map_polylines = np.full((all_polylines.shape[0], 7), np.nan, dtype=np.float32)
        # IDXs 0 to 2: x_start, y_start, z_start (need to be converted to meters). z_start is not available.
        xy_start = common.km_to_m(all_polylines[:, self._MAP_XY_START])
        map_polylines[:, :2] = xy_start

        # IDXs 3 to 5: direction vector, dir_x, dir_y. dir_z is not available
        xy_end = common.km_to_m(all_polylines[:, self._MAP_XY_END])
        direction = geometric_utils.compute_direction_vector(xy_start, xy_end)
        map_polylines[:, 3:5] = direction

        # IDX 6: semantic attribute
        map_polylines[:, 6] = np.array(
            [
                PolylineType[AMELIA_VALUE_TO_MAP_TYPE[v]].value
                for v in all_polylines[:, self._MAP_CLASS].squeeze(1).astype(np.int32)
            ]
        )

        return StaticMapData(
            map_polylines=map_polylines,
            exit_ids=AmeliaDataset.get_polyline_ids(map_data, "exit") if "exit" in map_data else None,
            exit_polyline_idxs=AmeliaDataset.get_polyline_idxs(map_data, "exit") if "exit" in map_data else None,
            runway_ids=AmeliaDataset.get_polyline_ids(map_data, "thr_id") if "thr_id" in map_data else None,
            runway_polyline_idxs=AmeliaDataset.get_polyline_idxs(map_data, "thr_id") if "thr_id" in map_data else None,
            taxiway_ids=AmeliaDataset.get_polyline_ids(map_data, "taxiway") if "taxiway" in map_data else None,
            taxiway_polyline_idxs=AmeliaDataset.get_polyline_idxs(map_data, "taxiway")
            if "taxiway" in map_data
            else None,
            ramp_ids=AmeliaDataset.get_polyline_ids(map_data, "ramp") if "ramp" in map_data else None,
            ramp_polyline_idxs=AmeliaDataset.get_polyline_idxs(map_data, "ramp") if "ramp" in map_data else None,
            stop_sign_ids=AmeliaDataset.get_polyline_ids(map_data, "hold_line") if "hold_line" in map_data else None,
            stop_sign_polyline_idxs=AmeliaDataset.get_polyline_idxs(map_data, "hold_line")
            if "hold_line" in map_data
            else None,
        )

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

        # Select the relevant agents in the scene
        agent_order = scenario_data["meta"]["agent_order"][self.ego_selection_strategy]
        agent_relevance = np.zeros(agent_data.num_agents, dtype=np.float32)
        relevant_idxs = agent_order[: self.num_relevant_agents]
        agent_relevance[relevant_idxs] = 1.0
        # Select the an ego-agent from the relevant agents
        ego_agent_index = self.generator.choice(relevant_idxs)

        # Repack meta information
        airport_id = scenario_data["airport_id"]
        metadata = ScenarioMetadata(
            scenario_id=f"{scenario_data['scenario_subdir']}_{scenario_data['scenario_id']}",
            timestamps_seconds=self.timestamps[: self.total_steps].tolist(),
            frequency_hz=self.frequency_hz,
            current_time_index=self.HIST_TIMESTEP,
            ego_vehicle_id=agent_data.agent_ids[ego_agent_index],
            ego_vehicle_index=ego_agent_index,
            track_length=self.total_steps,
            objects_of_interest=relevant_idxs.tolist(),
            dataset="amelia",
            # Thresholds adapted from: AmeliaScenes/blob/main/amelia_scenes/scoring/interactive.py
            agent_to_agent_max_distance=4000.0,  # m, runway extent (10,000-13,000ft)
            agent_to_conflict_point_max_distance=50.0,  # m, distance to a hold-line
            agent_to_agent_distance_breach=300.0,  # m, separation standards: airservicesaustralia.com
        )
        return Scenario(metadata=metadata, agent_data=agent_data, static_map_data=self.airport_maps[airport_id])

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

        airport_subdir = scenario_filepath.parent.name
        airport = self.data.airports[index]

        # Update the cache paths
        self.conflict_points_path = self.base_conflict_points_path / airport / airport_subdir
        self.closest_lanes_path = self.base_closest_lanes_path / airport / airport_subdir

        with scenario_filepath.open("rb") as f:
            scenario = pickle.load(f)  # nosec B301
            scenario["scenario_subdir"] = airport_subdir
            return scenario

    def collate_batch(self, batch_data: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportMissingParameterType]
        """Collates a batch of scenario data for processing.

        Args:
            batch_data (list): List of scenario data dictionaries.

        Returns:
            dict: A dictionary containing the batch size and the batch of scenarios.
        """
        batch_size = len(batch_data)
        return {"batch_size": batch_size, "scenario": batch_data}
