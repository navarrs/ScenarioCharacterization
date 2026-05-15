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


class Argoverse2Data(BaseDataset):
    """Dataset adapter for the Argoverse 2 Motion Forecasting dataset.

    Loads pickle files produced by av2_preprocess.py and transforms them into Scenario objects
    using the same schema as WaymoData and NuScenesData. AV2 scenarios run for 11 seconds at 10
    Hz (110 timesteps): 50 observed timesteps followed by 60 future timesteps.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes the AV2 Motion Forecasting dataset adapter."""
        super().__init__(config=config)
        self.last_timestep_to_consider = {
            "gt": config.last_timestep,  # 110 for full 11-second scenario
            "ho": config.hist_timestep,  # 50 for history-only (5 seconds observed)
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
        """Loads the AV2 dataset scenario paths from disk."""
        logger.info("Loading AV2 scenario base data from %s", self.scenario_base_path)
        self.scenarios = natsorted(list(map(str, self.scenario_base_path.rglob("*.pkl"))))

        if self.num_scenarios != -1:
            self.scenarios = self.scenarios[: self.num_scenarios]

        logger.info("Total number of scenarios found: %d", len(self.scenarios))
        if self.create_metadata:
            self.compute_metadata()

    def repack_agent_data(self, agent_data: dict[str, Any], ego_index: int) -> AgentData:
        """Packs agent information from the AV2 pickle format into AgentData.

        Pads trajectories with zeros if shorter than last_timestep (should not normally
        occur for AV2, but matches the nuScenes safety pattern for robustness).

        Args:
            agent_data: dict with keys "object_id", "object_type", and "trajs"
                (shape: num_agents x num_timesteps x 10).
            ego_index: Index of the focal (ego) vehicle in the agent arrays.

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
        """Extracts speed limits in mph from the polyline dictionary (always 0.0 for AV2)."""
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
        """Packs static map information from the AV2 pickle format into StaticMapData.

        AV2 does not have road edges, speed bumps, or stop signs; those fields are None. Speed limits are always 0.0
        (not available in AV2). Lane boundaries are stored as "road_line" entries in the pickle, with type IDs
        reflecting mark type (solid/dashed).

        Args:
            static_map_data: Map info dict from the AV2 pickle, or None.

        Returns:
            StaticMapData pydantic model, or None if static_map_data is None.
        """
        if static_map_data is None:
            return None

        map_polylines = static_map_data["all_polylines"].astype(np.float32)

        return StaticMapData(
            map_polylines=map_polylines,
            lane_ids=(Argoverse2Data.get_polyline_ids(static_map_data, "lane") if "lane" in static_map_data else None),
            lane_speed_limits_mph=(
                Argoverse2Data.get_speed_limit_mph(static_map_data, "lane") if "lane" in static_map_data else None
            ),
            lane_polyline_idxs=(
                Argoverse2Data.get_polyline_idxs(static_map_data, "lane") if "lane" in static_map_data else None
            ),
            road_line_ids=(
                Argoverse2Data.get_polyline_ids(static_map_data, "road_line")
                if "road_line" in static_map_data
                else None
            ),
            road_line_polyline_idxs=(
                Argoverse2Data.get_polyline_idxs(static_map_data, "road_line")
                if "road_line" in static_map_data
                else None
            ),
            road_edge_ids=None,
            road_edge_polyline_idxs=None,
            crosswalk_ids=(
                Argoverse2Data.get_polyline_ids(static_map_data, "crosswalk")
                if "crosswalk" in static_map_data
                else None
            ),
            crosswalk_polyline_idxs=(
                Argoverse2Data.get_polyline_idxs(static_map_data, "crosswalk")
                if "crosswalk" in static_map_data
                else None
            ),
            speed_bump_ids=None,
            speed_bump_polyline_idxs=None,
            stop_sign_ids=None,
            stop_sign_polyline_idxs=None,
            stop_sign_lane_ids=[],
        )

    def repack_dynamic_map_data(self, dynamic_map_data: dict[str, Any]) -> DynamicMapData:
        """Packs dynamic map data into DynamicMapData.

        AV2 Motion Forecasting does not include per-timestep traffic signal states, so this
        always returns a DynamicMapData with all fields set to None.
        """
        stop_points = dynamic_map_data["stop_point"][: self.total_steps]
        lane_id = list(dynamic_map_data["lane_id"][: self.total_steps])
        states = dynamic_map_data["state"][: self.total_steps]

        if len(stop_points) == 0:
            stop_points = None
            lane_id = None
            states = None

        return DynamicMapData(stop_points=stop_points, lane_ids=lane_id, states=states)

    def transform_scenario_data(self, scenario_data: dict[str, Any]) -> Scenario:
        """Transforms an AV2 pickle dict into a Scenario object.

        Differs from NuScenesData in two ways:
        1. dataset field is "argoverse2".
        2. agent_relevance is 1.0 for FOCAL and SCORED tracks (tracks_to_predict with
           difficulty >= 1), and 0.0 for all others. This focuses scoring on the
           evaluated subset rather than treating all agents uniformly.

        Args:
            scenario_data: Dict loaded from an AV2 scenario pickle file.

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

        # Scored tracks (FOCAL + SCORED categories) get relevance 1.0; others get 0.0.
        agent_relevance = np.zeros(agent_data.num_agents, dtype=np.float32)
        for idx in scenario_data["tracks_to_predict"]["track_index"]:
            agent_relevance[idx] = 1.0
        agent_relevance[scenario_data["sdc_track_index"]] = 1.0
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
            dataset="argoverse2",
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
