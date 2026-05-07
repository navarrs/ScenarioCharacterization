import pickle  # nosec B403
from pathlib import Path
from typing import Any

import numpy as np
from natsort import natsorted
from numpy.typing import NDArray
from omegaconf import DictConfig

from characterization.datasets.base_dataset import BaseDataset
from characterization.schemas.scenario import (
    AgentData,
    AgentType,
    DynamicMapData,
    Scenario,
    ScenarioMetadata,
    StaticMapData,
)
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)

_MIN_TIMESTAMPS_FOR_DT = 2


class NuScenesData(BaseDataset):
    """Dataset adapter for the nuScenes dataset.

    Loads pickle files produced by nuscenes_preprocess.py and transforms them into
    Scenario objects using the same schema as WaymoData. Trajectories in the pickles
    are pre-interpolated to 10Hz.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes the nuScenes dataset handler."""
        super().__init__(config=config)
        self.last_timestep_to_consider = {
            "gt": config.last_timestep,
            "ho": config.hist_timestep,
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
        """Loads the nuScenes dataset and scenario metadata."""
        logger.info("Loading nuScenes scenario base data from %s", self.scenario_base_path)
        self.scenarios = natsorted(list(map(str, self.scenario_base_path.rglob("*.pkl"))))

        if self.num_scenarios != -1:
            self.scenarios = self.scenarios[: self.num_scenarios]

        logger.info("Total number of scenarios found: %d", len(self.scenarios))
        if self.create_metadata:
            self.compute_metadata()

    def repack_agent_data(self, agent_data: dict[str, Any], ego_index: int) -> AgentData:
        """Packs agent information from the nuScenes pickle format into AgentData.

        Args:
            agent_data: dict with keys "object_id", "object_type", and "trajs"
                (shape: num_agents x num_timesteps x 10).
            ego_index: Index of the ego vehicle in the agent arrays.

        Returns:
            AgentData pydantic model.
        """
        trajectories = agent_data["trajs"]
        _, num_timesteps, _ = trajectories.shape

        last_timestep = self.last_timestep_to_consider[self.scenario_type]
        if num_timesteps < last_timestep:
            pad_length = last_timestep - num_timesteps
            pad = np.zeros((trajectories.shape[0], pad_length, trajectories.shape[2]), dtype=trajectories.dtype)
            trajectories = np.concatenate([trajectories, pad], axis=1)

        self.total_steps = last_timestep
        trajectories = trajectories[:, :last_timestep, :]
        object_types = [AgentType[n] for n in agent_data["object_type"]]
        object_types[ego_index] = AgentType.TYPE_EGO_AGENT
        return AgentData(
            agent_ids=agent_data["object_id"],
            agent_types=object_types,
            agent_trajectories=trajectories.astype(np.float32),
        )

    @staticmethod
    def get_polyline_ids(polyline: dict[str, Any], key: str) -> NDArray[np.int32]:
        """Extracts polyline IDs from the polyline dictionary."""
        return np.array([value["id"] for value in polyline[key]], dtype=np.int32)

    @staticmethod
    def get_speed_limit_mph(polyline: dict[str, Any], key: str) -> NDArray[np.float32]:
        """Extracts speed limits in mph from the polyline dictionary (zeros for nuScenes)."""
        return np.array([value["speed_limit_mph"] for value in polyline[key]], dtype=np.float32)

    @staticmethod
    def get_polyline_idxs(polyline: dict[str, Any], key: str) -> NDArray[np.int32] | None:
        """Extracts polyline start and end indices from the polyline dictionary."""
        polyline_idxs = np.array(
            [[value["polyline_index"][0], value["polyline_index"][1]] for value in polyline[key]],
            dtype=np.int32,
        )
        if polyline_idxs.shape[0] == 0:
            return None
        return polyline_idxs

    def repack_static_map_data(self, static_map_data: dict[str, Any] | None) -> StaticMapData | None:
        """Packs static map information from the nuScenes pickle format into StaticMapData.

        The pickle format produced by nuscenes_preprocess.py matches the Waymo format key-for-key,
        except that "speed_bump" is absent (handled by the `in static_map_data` guards below).

        Args:
            static_map_data: Map info dict from the nuScenes pickle, or None.

        Returns:
            StaticMapData pydantic model, or None if static_map_data is None.
        """
        if static_map_data is None:
            return None

        map_polylines = static_map_data["all_polylines"].astype(np.float32)

        return StaticMapData(
            map_polylines=map_polylines,
            lane_ids=NuScenesData.get_polyline_ids(static_map_data, "lane") if "lane" in static_map_data else None,
            lane_speed_limits_mph=(
                NuScenesData.get_speed_limit_mph(static_map_data, "lane") if "lane" in static_map_data else None
            ),
            lane_polyline_idxs=(
                NuScenesData.get_polyline_idxs(static_map_data, "lane") if "lane" in static_map_data else None
            ),
            road_line_ids=(
                NuScenesData.get_polyline_ids(static_map_data, "road_line") if "road_line" in static_map_data else None
            ),
            road_line_polyline_idxs=(
                NuScenesData.get_polyline_idxs(static_map_data, "road_line") if "road_line" in static_map_data else None
            ),
            road_edge_ids=(
                NuScenesData.get_polyline_ids(static_map_data, "road_edge") if "road_edge" in static_map_data else None
            ),
            road_edge_polyline_idxs=(
                NuScenesData.get_polyline_idxs(static_map_data, "road_edge") if "road_edge" in static_map_data else None
            ),
            crosswalk_ids=(
                NuScenesData.get_polyline_ids(static_map_data, "crosswalk") if "crosswalk" in static_map_data else None
            ),
            crosswalk_polyline_idxs=(
                NuScenesData.get_polyline_idxs(static_map_data, "crosswalk") if "crosswalk" in static_map_data else None
            ),
            speed_bump_ids=None,
            speed_bump_polyline_idxs=None,
            stop_sign_ids=(
                NuScenesData.get_polyline_ids(static_map_data, "stop_sign") if "stop_sign" in static_map_data else None
            ),
            stop_sign_polyline_idxs=(
                NuScenesData.get_polyline_idxs(static_map_data, "stop_sign") if "stop_sign" in static_map_data else None
            ),
            stop_sign_lane_ids=[stop_sign["lane_ids"] for stop_sign in static_map_data.get("stop_sign", [])],
        )

    def repack_dynamic_map_data(self, dynamic_map_data: dict[str, Any]) -> DynamicMapData:
        """Packs dynamic map data into DynamicMapData.

        nuScenes does not have per-timestep traffic signal states, so this always returns
        a DynamicMapData with all fields set to None.
        """
        stop_points = dynamic_map_data["stop_point"][: self.total_steps]
        lane_id = [lid.astype(np.int64) for lid in dynamic_map_data["lane_id"][: self.total_steps]]
        states = dynamic_map_data["state"][: self.total_steps]

        if len(stop_points) == 0:
            stop_points = None
            lane_id = None
            states = None

        return DynamicMapData(stop_points=stop_points, lane_ids=lane_id, states=states)

    def transform_scenario_data(self, scenario_data: dict[str, Any]) -> Scenario:
        """Transforms a nuScenes pickle dict into a Scenario object.

        Args:
            scenario_data: Dict loaded from a nuScenes scenario pickle file.

        Returns:
            Scenario object ready for feature extraction.
        """
        agent_data = self.repack_agent_data(scenario_data["track_infos"], scenario_data["sdc_track_index"])
        static_map_data = self.repack_static_map_data(scenario_data["map_infos"])
        dynamic_map_data = self.repack_dynamic_map_data(scenario_data["dynamic_map_infos"])

        timestamps = scenario_data["timestamps_seconds"]
        if len(timestamps) < self.total_steps:
            dt = float(timestamps[-1] - timestamps[-2]) if len(timestamps) >= _MIN_TIMESTAMPS_FOR_DT else 1.0 / 10.0
            extra = timestamps[-1] + dt * np.arange(1, self.total_steps - len(timestamps) + 1)
            timestamps = np.concatenate([timestamps, extra])
        timestamps = timestamps[: self.total_steps]

        agent_relevance = np.ones(agent_data.num_agents, dtype=np.float32)
        agent_data.agent_relevance = agent_relevance

        freq = np.round(1.0 / np.mean(np.diff(timestamps))).item()
        metadata = ScenarioMetadata(
            scenario_id=scenario_data["scenario_id"],
            timestamps_seconds=list(timestamps),
            frequency_hz=min(freq, 10.0),
            current_time_index=scenario_data["current_time_index"],
            ego_vehicle_id=agent_data.agent_ids[scenario_data["sdc_track_index"]],
            ego_vehicle_index=scenario_data["sdc_track_index"],
            track_length=self.total_steps,
            objects_of_interest=scenario_data["objects_of_interest"],
            dataset="nuscenes",
        )

        return Scenario(
            metadata=metadata,
            agent_data=agent_data,
            static_map_data=static_map_data,
            dynamic_map_data=dynamic_map_data,
        )

    def load_scenario_information(self, index: int) -> dict[str, dict[str, Any]] | None:
        """Loads a scenario pickle file by index.

        Args:
            index: Index of the scenario to load.

        Returns:
            Deserialized scenario dict, or None if the file is missing or corrupt.
        """
        scenario_filepath = Path(self.scenarios[index])
        if not scenario_filepath.exists():
            return None

        with scenario_filepath.open("rb") as f:
            try:
                scenario = pickle.load(f)  # nosec B301
            except (EOFError, pickle.UnpicklingError) as e:
                logger.warning("Failed to load scenario from %s: %s", scenario_filepath, e)
                return None
        return scenario

    def collate_batch(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        """Collates a batch of scenario data for processing."""
        return {"batch_size": len(batch_data), "scenario": batch_data}
