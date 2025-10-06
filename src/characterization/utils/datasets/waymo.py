import pickle  # nosec B403
import time
from pathlib import Path
from typing import Any

import numpy as np
from natsort import natsorted
from omegaconf import DictConfig

from characterization.schemas.scenario import (
    AgentData,
    AgentType,
    DynamicMapData,
    Scenario,
    ScenarioMetadata,
    StaticMapData,
)
from characterization.utils.datasets.dataset import BaseDataset
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


class WaymoData(BaseDataset):
    """Class to handle the Waymo Open Motion Dataset (WOMD)."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the Waymo Open Motion Dataset (WOMD) handler."""
        super().__init__(config=config)
        # Three challengingness levels: 0 (easy), 1 (medium), 2 (hard) obtained from WOMD
        self.DIFFICULTY_WEIGHTS = {0: 0.8, 1: 0.9, 2: 1.0}

        self.LAST_TIMESTEP = 91
        self.HIST_TIMESTEP = 11

        self.LAST_TIMESTEP_TO_CONSIDER = {
            "gt": self.LAST_TIMESTEP,
            "ho": self.HIST_TIMESTEP,
        }

        self.load = config.get("load", True)
        if self.load:
            try:
                logger.info("Loading scenario infos...")
                self.load_data()
            except AssertionError:
                logger.exception("Error loading scenario infos")
                raise

    def load_data(self) -> None:
        """Loads the Waymo dataset and scenario metadata.

        Loads scenario metadata and scenario file paths, applies sharding if enabled,
        and checks that the number of scenarios matches the number of conflict points.

        Raises:
            AssertionError: If the number of scenarios and conflict points do not match.
        """
        start = time.time()
        logger.info("Loading WOMD scenario base data from %s", self.scenario_base_path)
        with open(self.scenario_meta_path, "rb") as f:
            self.data.metas = pickle.load(f)[:: self.step]  # nosec B301
        self.data.scenarios_ids = natsorted([f"sample_{x['scenario_id']}.pkl" for x in self.data.metas])
        self.data.scenarios = natsorted(
            [f"{self.scenario_base_path}/sample_{x['scenario_id']}.pkl" for x in self.data.metas],
        )
        logger.info("Loading data took %2f seconds.", time.time() - start)

        # TODO: remove this
        self.shard()

    def repack_agent_data(self, agent_data: dict[str, Any]) -> AgentData:
        """Packs agent information from Waymo format to AgentData format.

        Args:
            agent_data (dict): dictionary containing Waymo actor data:
                'object_id': indicating each agent IDs
                'object_type': indicating each agent type
                'trajs': tensor(num_agents, num_timesteps, num_features) containing each agent's kinematic information.

        Returns:
            AgentData: pydantic validator encapsulating agent information.
        """
        trajectories = agent_data["trajs"]  # shape: [num_agents, num_timesteps, num_features]
        _, num_timesteps, _ = trajectories.shape

        last_timestep = self.LAST_TIMESTEP_TO_CONSIDER[self.scenario_type]
        if num_timesteps < last_timestep:
            error_message = (
                f"Scenario has only {num_timesteps} timesteps, but expected at least {last_timestep} timesteps."
            )
            raise AssertionError(error_message)

        trajectories = trajectories[:, :last_timestep, :]  # shape: [num_agents, last_timestep, dim]
        self.total_steps = last_timestep
        object_types = [AgentType[n] for n in agent_data["object_type"]]
        return AgentData(agent_ids=agent_data["object_id"], agent_types=object_types, agent_trajectories=trajectories)

    @staticmethod
    def get_polyline_ids(polyline: dict[str, Any], key: str) -> np.ndarray:
        """Extracts polyline indices from the polyline dictionary."""
        return np.array([value["id"] for value in polyline[key]], dtype=np.int32)

    @staticmethod
    def get_speed_limit_mph(polyline: dict[str, Any], key: str) -> np.ndarray:
        """Extracts speed limit in mph from the polyline dictionary."""
        return np.array([value["speed_limit_mph"] for value in polyline[key]], dtype=np.float32)

    @staticmethod
    def get_polyline_idxs(polyline: dict[str, Any], key: str) -> np.ndarray | None:
        """Extracts polyline start and end indices from the polyline dictionary."""
        polyline_idxs = np.array(
            [[value["polyline_index"][0], value["polyline_index"][1]] for value in polyline[key]],
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

        map_polylines = static_map_data["all_polylines"].astype(np.float32)  # shape: [N, 3] or [N, 3, 2]

        return StaticMapData(
            map_polylines=map_polylines,
            lane_ids=WaymoData.get_polyline_ids(static_map_data, "lane") if "lane" in static_map_data else None,
            lane_speed_limits_mph=WaymoData.get_speed_limit_mph(static_map_data, "lane")
            if "lane" in static_map_data
            else None,
            lane_polyline_idxs=WaymoData.get_polyline_idxs(static_map_data, "lane")
            if "lane" in static_map_data
            else None,
            road_line_ids=WaymoData.get_polyline_ids(static_map_data, "road_line")
            if "road_line" in static_map_data
            else None,
            road_line_polyline_idxs=WaymoData.get_polyline_idxs(static_map_data, "road_line")
            if "road_line" in static_map_data
            else None,
            road_edge_ids=WaymoData.get_polyline_ids(static_map_data, "road_edge")
            if "road_edge" in static_map_data
            else None,
            road_edge_polyline_idxs=WaymoData.get_polyline_idxs(static_map_data, "road_edge")
            if "road_edge" in static_map_data
            else None,
            crosswalk_ids=WaymoData.get_polyline_ids(static_map_data, "crosswalk")
            if "crosswalk" in static_map_data
            else None,
            crosswalk_polyline_idxs=WaymoData.get_polyline_idxs(static_map_data, "crosswalk")
            if "crosswalk" in static_map_data
            else None,
            speed_bump_ids=WaymoData.get_polyline_ids(static_map_data, "speed_bump")
            if "speed_bump" in static_map_data
            else None,
            speed_bump_polyline_idxs=WaymoData.get_polyline_idxs(static_map_data, "speed_bump")
            if "speed_bump" in static_map_data
            else None,
            stop_sign_ids=WaymoData.get_polyline_ids(static_map_data, "stop_sign")
            if "stop_sign" in static_map_data
            else None,
            stop_sign_polyline_idxs=WaymoData.get_polyline_idxs(static_map_data, "stop_sign")
            if "stop_sign" in static_map_data
            else None,
            stop_sign_lane_ids=[
                stop_sign["lane_ids"] for stop_sign in static_map_data.get("stop_sign", {"lane_ids": []})
            ],
        )

    def repack_dynamic_map_data(self, dynamic_map_data: dict[str, Any]) -> DynamicMapData:
        """Packs dynamic map information from Waymo format to DynamicMapData format.

        Args:
            dynamic_map_data (dict): dictionary containing Waymo dynamic scenario data:
                'stop_points': traffic light stopping points.
                'lane_id': IDs of the lanes where the traffic light is.
                'state': state of the traffic light (e.g., red, etc).

        Returns:
            DynamicMapData: pydantic validator encapsulating static map information.
        """
        stop_points = dynamic_map_data["stop_point"][: self.total_steps]
        lane_id = [lid.astype(np.int64) for lid in dynamic_map_data["lane_id"][: self.total_steps]]
        states = dynamic_map_data["state"][: self.total_steps]
        num_dynamic_stop_points = len(stop_points)

        if num_dynamic_stop_points == 0:
            stop_points = None
            lane_id = None
            states = None

        return DynamicMapData(stop_points=stop_points, lane_ids=lane_id, states=states)

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
        agent_data = self.repack_agent_data(scenario_data["track_infos"])
        # Repack static map information from input scenario
        static_map_data = self.repack_static_map_data(scenario_data["map_infos"])

        # Add conflict point information

        # TODO: refactor dynamic map data schema.
        # Repack dynamic map information
        dynamic_map_data = self.repack_dynamic_map_data(scenario_data["dynamic_map_infos"])

        timestamps = scenario_data["timestamps_seconds"][: self.total_steps]

        # Select tracks to predict
        ego_vehicle_index = scenario_data["sdc_track_index"]

        agent_relevance = np.zeros(agent_data.num_agents, dtype=np.float32)
        tracks_to_predict = scenario_data["tracks_to_predict"]
        tracks_to_predict_index = np.asarray(tracks_to_predict["track_index"] + [ego_vehicle_index])
        tracks_to_predict_difficulty = np.asarray(tracks_to_predict["difficulty"] + [2.0])

        # Set agent_relevance for tracks_to_predict_index based on tracks_to_predict_difficulty
        for idx, difficulty in zip(tracks_to_predict_index, tracks_to_predict_difficulty, strict=False):
            agent_relevance[idx] = self.DIFFICULTY_WEIGHTS.get(difficulty, 0.0)
        agent_data.agent_relevance = agent_relevance

        # Repack meta information
        freq = np.round(1 / np.mean(np.diff(timestamps))).item()
        metadata = ScenarioMetadata(
            scenario_id=scenario_data["scenario_id"],
            timestamps_seconds=timestamps,
            frequency_hz=min(freq, 10.0),
            current_time_index=scenario_data["current_time_index"],
            ego_vehicle_id=agent_data.agent_ids[ego_vehicle_index],
            ego_vehicle_index=ego_vehicle_index,
            track_length=self.total_steps,
            objects_of_interest=scenario_data["objects_of_interest"],
            dataset="waymo",
        )

        return Scenario(
            metadata=metadata,
            agent_data=agent_data,
            static_map_data=static_map_data,
            # NOTE: the model is not currently using dynamic map data.
            dynamic_map_data=dynamic_map_data,
        )

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
