import itertools
import os
import pickle  # nosec B403
import time
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from natsort import natsorted
from omegaconf import DictConfig
from scipy.signal import resample
from tqdm import tqdm

from characterization.schemas.scenario import (
    AgentData,
    AgentType,
    DynamicMapData,
    Scenario,
    ScenarioMetadata,
    StaticMapData,
)
from characterization.utils.common import AgentTrajectoryMasker
from characterization.utils.datasets.dataset import BaseDataset
from characterization.utils.geometric_utils import compute_dists_to_conflict_points
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


class WaymoData(BaseDataset):
    """Class to handle the Waymo Open Motion Dataset (WOMD)."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the Waymo Open Motion Dataset (WOMD) handler."""
        super().__init__(config=config)

        self.AGENT_TYPE_MAP = {
            "TYPE_VEHICLE": 0,
            "TYPE_PEDESTRIAN": 1,
            "TYPE_CYCLIST": 2,
        }
        self.AGENT_NUM_TO_TYPE = {
            0: "TYPE_VEHICLE",
            1: "TYPE_PEDESTRIAN",
            2: "TYPE_CYCLIST",
        }

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

        num_scenarios = len(self.data.scenarios_ids)

        # Pre-checks: conflict points
        self.check_conflict_points()
        num_conflict_points = len(self.data.conflict_points)
        if num_scenarios != num_conflict_points:
            error_message = (
                f"Number of scenarios ({num_scenarios}) != number of conflict points ({num_conflict_points})."
            )
            raise AssertionError(error_message)

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

    def transform_scenario_data(
        self, scenario_data: dict[str, Any], conflict_points_data: dict[str, Any] | None = None
    ) -> Scenario:
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
        agent_distances_to_conflict_points = None
        conflict_points = None
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
        if static_map_data is not None:
            static_map_data.map_conflict_points = conflict_points
            static_map_data.agent_distances_to_conflict_points = agent_distances_to_conflict_points

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
        metadata = ScenarioMetadata(
            scenario_id=scenario_data["scenario_id"],
            timestamps_seconds=timestamps,
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

    def check_conflict_points(self) -> None:
        """Checks if conflict points are already computed for each scenario.

        If not, computes conflict points for each scenario and saves them to disk.
        Updates the dataset's conflict points list.

        Returns:
            None
        """
        logger.info("Checking if conflict points have been computed for each scenario.")
        start = time.time()
        zipped = zip(self.data.scenarios_ids, self.data.scenarios, strict=False)

        def process_file(scenario_id: str, scenario_path: str) -> str:
            conflict_points_filepath = os.path.join(self.conflict_points_path, scenario_id)
            if os.path.exists(conflict_points_filepath):
                return conflict_points_filepath

            # Otherwise compute conflict points
            with open(scenario_path, "rb") as f:
                scenario = pickle.load(f)  # nosec B301

            static_map_infos = scenario["map_infos"]
            dynamic_map_infos = scenario["dynamic_map_infos"]
            agent_trajectories = AgentTrajectoryMasker(scenario["track_infos"]["trajs"])
            agent_positions = agent_trajectories.agent_xyz_pos
            conflict_points = self.find_conflict_points(static_map_infos, dynamic_map_infos, agent_positions)

            with open(conflict_points_filepath, "wb") as f:
                pickle.dump(conflict_points, f, protocol=pickle.HIGHEST_PROTOCOL)

            return conflict_points_filepath

        if self.parallel:
            outs = Parallel(n_jobs=self.num_workers, batch_size=self.batch_size)(
                delayed(process_file)(scenario_id=scenario_id, scenario_path=scenario_path)
                for scenario_id, scenario_path in tqdm(zipped, total=len(self.data.scenarios_ids))
            )
            self.data.conflict_points = natsorted(outs)
        else:
            for scenario_id, scenario_path in tqdm(zipped, total=len(self.data.scenarios_ids)):
                out = process_file(scenario_id=scenario_id, scenario_path=scenario_path)
                self.data.conflict_points.append(out)

        self.data.conflict_points = natsorted(self.data.conflict_points)

        logger.info("Conflict points check completed in %.2f seconds.", time.time() - start)

    def find_conflict_points(
        self,
        static_map_info: dict[str, Any],
        dynamic_map_info: dict[str, Any],
        agent_positions: np.ndarray,
        ndim: int = 3,
        min_timesteps: int = 5,
    ) -> dict[str, Any]:
        """Finds the conflict points in the map for a scenario.

        Args:
            static_map_info (dict): The static map information.
            dynamic_map_info (dict): The dynamic map information.
            agent_positions (np.ndarray): Array of agent positions (shape: [N_agents, T, 3]).
            ndim (int): Number of dimensions to consider (2 or 3). Defaults to 3.
            min_timesteps (int): Minimum number of timesteps required to compute distances. Defaults to

        Returns:
            dict: The conflict points in the map, including:
                - 'static': Static conflict points (e.g., crosswalks, speed bumps, stop signs).
                - 'dynamic': Dynamic conflict points (e.g., traffic lights).
                - 'lane_intersections': Lane intersection points.
                - 'all_conflict_points': All conflict points concatenated.
                - 'agent_distances_to_conflict_points': Distances from each agent to each conflict point.
        """
        polylines = static_map_info["all_polylines"]

        # Static Conflict Points: Crosswalks, Speed Bumps and Stop Signs
        static_conflict_points_list = []
        for conflict_point in static_map_info["crosswalk"] + static_map_info["speed_bump"]:
            start, end = conflict_point["polyline_index"]
            points = polylines[start:end][:, :ndim]
            points = resample(points, points.shape[0] * self.conflict_points_cfg.resample_factor)
            static_conflict_points_list.append(points)

        for conflict_point in static_map_info["stop_sign"]:
            start, end = conflict_point["polyline_index"]
            points = polylines[start:end][:, :ndim]
            static_conflict_points_list.append(points)

        static_conflict_points = (
            np.concatenate(static_conflict_points_list) if len(static_conflict_points_list) > 0 else np.empty((0, ndim))
        )

        # Lane Intersections
        lane_infos = static_map_info["lane"]
        lanes = [polylines[li["polyline_index"][0] : li["polyline_index"][1]][:, :ndim] for li in lane_infos]
        # lanes = []
        # for lane_info in static_map_info['lane']:
        #     start, end = lane_info['polyline_index']
        #     lane = P[start:end]
        #     lane = signal.resample(lane, lane.shape[0] * resample_factor)
        #     lanes.append(lane)
        num_lanes = len(lanes)

        lane_combinations = list(itertools.combinations(range(num_lanes), 2))
        lane_intersections_list = []
        for i, j in lane_combinations:
            lane_i, lane_j = lanes[i], lanes[j]

            dists_ij = np.linalg.norm(lane_i[:, None] - lane_j, axis=-1)
            i_idx, j_idx = np.where(self.conflict_points_cfg.intersection_threshold > dists_ij)

            # TODO: determine if two lanes are consecutive, but not entry/exit lanes. If this is the
            # case there'll be an intersection that is not a conflict point.
            start_i, end_i = i_idx[:min_timesteps], i_idx[-min_timesteps:]
            start_j, end_j = j_idx[:min_timesteps], j_idx[-min_timesteps:]
            if (np.any(start_i < min_timesteps) and np.any(end_j > lane_j.shape[0] - min_timesteps)) or (
                np.any(start_j < min_timesteps) and np.any(end_i > lane_i.shape[0] - min_timesteps)
            ):
                lanes_i_ee = lane_infos[i]["entry_lanes"] + lane_infos[i]["exit_lanes"]
                lanes_j_ee = lane_infos[j]["entry_lanes"] + lane_infos[j]["exit_lanes"]
                if j not in lanes_i_ee and i not in lanes_j_ee:
                    continue

            if i_idx.shape[0] > 0:
                lane_intersections_list.append(lane_i[i_idx])

            if j_idx.shape[0] > 0:
                lane_intersections_list.append(lane_j[j_idx])

        lane_intersections = (
            np.concatenate(lane_intersections_list) if len(lane_intersections_list) > 0 else np.empty((0, 3))
        )

        # Dynamic Conflict Points: Traffic Lights
        stops = dynamic_map_info["stop_point"]
        dynamic_conflict_points = np.empty((0, ndim))
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

    def load_scenario_information(self, index: int) -> dict[str, dict[str, Any]]:
        """Loads scenario and conflict point information by index.

        Args:
            index (int): Index of the scenario to load.

        Returns:
            dict: A dictionary containing the scenario and conflict points.

        Raises:
            ValidationError: If the scenario data does not pass schema validation.
        """
        with open(self.data.scenarios[index], "rb") as f:
            scenario = pickle.load(f)  # nosec B301

        with open(self.data.conflict_points[index], "rb") as f:
            conflict_points = pickle.load(f)  # nosec B301

        return {
            "scenario": scenario,
            "conflict_points": conflict_points,
        }

    def collate_batch(self, batch_data: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportMissingParameterType]
        """Collates a batch of scenario data for processing.

        Args:
            batch_data (list): List of scenario data dictionaries.

        Returns:
            dict: A dictionary containing the batch size and the batch of scenarios.
        """
        batch_size = len(batch_data)
        # key_to_list = {}
        # for key in batch_data[0].keys():
        #     key_to_list[key] = [batch_data[idx][key] for idx in range(batch_size)]

        # input_dict = {}
        # for key, val_list in key_to_list.items():
        #     if key in ['scenario_id', 'num_agents', 'ego_index', 'ego_id', 'current_time_index']:
        #         input_dict[key] = np.asarray(val_list)

        return {
            "batch_size": batch_size,
            "scenario": batch_data,
        }
