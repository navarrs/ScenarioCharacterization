import itertools
import os
import pickle  # nosec B403
import time
from typing import Any

import numpy as np
from natsort import natsorted
from omegaconf import DictConfig
from pydantic import ValidationError
from rich.progress import track
from scipy.signal import resample

from characterization.utils.common import compute_dists_to_conflict_points, get_logger
from characterization.utils.datasets.dataset import BaseDataset
from characterization.utils.schemas import Scenario

logger = get_logger(__name__)


class WaymoData(BaseDataset):
    def __init__(self, config: DictConfig) -> None:
        super(WaymoData, self).__init__(config=config)

        # Waymo dataset masks
        # center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
        # center_x, center_y, center_z -> coordinates fo the object's BBox center
        # length, width, height -> dimensions of the object's BBox in meters
        # heading -> yaw angle in radians of the forward direction of the the BBox
        # velocity_x, velocity_y -> x and y components of the object's velocity in m/s
        self.AGENT_DIMS = [False, False, False, True, True, True, False, False, False, False]
        self.AGENT_LENGTHS = [False, False, False, True, False, False, False, False, False, False]
        self.AGENT_WIDTHS = [False, False, False, False, True, False, False, False, False, False]
        self.AGENT_HEIGHTS = [False, False, False, False, False, True, False, False, False, False]
        self.HEADING_IDX = [False, False, False, False, False, False, True, False, False, False]
        self.POS_XY_IDX = [True, True, False, False, False, False, False, False, False, False]
        self.POS_XYZ_IDX = [True, True, True, False, False, False, False, False, False, False]
        self.VEL_XY_IDX = [False, False, False, False, False, False, False, True, True, False]
        self.AGENT_VALID = [False, False, False, False, False, False, False, False, False, True]

        # Interpolated stuff
        self.IPOS_XY_IDX = [True, True, False, False, False, False, False]
        self.IPOS_SDZ_IDX = [False, False, True, True, True, False, False]
        self.IPOS_SD_IDX = [False, False, False, True, True, False, False]
        self.ILANE_IDX = [False, False, False, False, False, True, False]
        self.IVALID_IDX = [False, False, False, False, False, False, True]

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

        self.STATIONARY_SPEED = 0.25  # m/s
        self.AGENT_TO_AGENT_MAX_DISTANCE = 50.0  # meters
        self.AGENT_TO_CONFLICT_POINT_MAX_DISTANCE = 2.0  # meters
        self.AGENT_TO_AGENT_DISTANCE_BREACH = 1.0  # meters
        self.HEADING_THRESHOLD = 45  # degrees

        self.AGENT_TO_AGENT_MAX_HEADING = 45.0  # degrees

        self.load = config.get("load", True)
        if self.load:
            try:
                logger.info("Loading scenario infos...")
                self.load_data()
            except AssertionError as e:
                logger.error("Error loading scenario infos: %s", e)
                raise e

    def load_data(self) -> None:
        """Loads the Waymo dataset and scenario metadata.

        Loads scenario metadata and scenario file paths, applies sharding if enabled,
        and checks that the number of scenarios matches the number of conflict points.

        Raises:
            AssertionError: If the number of scenarios and conflict points do not match.
        """
        start = time.time()
        logger.info(f"Loading WOMD scenario base data from {self.scenario_base_path}")
        with open(self.scenario_meta_path, "rb") as f:
            self.data.metas = pickle.load(f)[:: self.step]  # nosec B301
        self.data.scenarios_ids = natsorted([f"sample_{x['scenario_id']}.pkl" for x in self.data.metas])
        self.data.scenarios = natsorted(
            [f"{self.scenario_base_path}/sample_{x['scenario_id']}.pkl" for x in self.data.metas],
        )
        logger.info(f"Loading data took {time.time() - start} seconds.")

        # TODO: remove this
        self.shard()

        num_scenarios = len(self.data.scenarios_ids)

        # Pre-checks: conflict points
        self.check_conflict_points()
        num_conflict_points = len(self.data.conflict_points)
        if not num_scenarios == num_conflict_points:
            raise AssertionError(
                f"Number of scenarios ({num_scenarios}) != number of conflict points ({num_conflict_points}).",
            )

    def transform_scenario_data(
        self,
        scenario_data: dict[str, Any],
        conflict_points_data: dict[str, Any] | None = None,
    ) -> Scenario:
        """Transforms the scene data into a format suitable for processing.

        Args:
            scenario_data (dict): The raw scenario data.
            conflict_points (dict or None): The conflict points for the scenario.

        Returns:
            dict: The transformed scenario data, including agent and map information.
        """

        def get_polyline_ids(polyline: dict[str, Any], key: str) -> np.ndarray:
            """Extracts polyline indices from the polyline dictionary."""
            return np.array([value["id"] for value in polyline[key]], dtype=np.int32)

        def get_speed_limit_mph(polyline: dict[str, Any], key: str) -> np.ndarray:
            """Extracts speed limit in mph from the polyline dictionary."""
            speed_limit_mph = np.array([value["speed_limit_mph"] for value in polyline[key]], dtype=np.float32)
            # if speed_limit_mph.shape[0] == 0:
            #     return np.empty((0,), dtype=np.float32)
            return speed_limit_mph

        def get_polyline_idxs(polyline: dict[str, Any], key: str) -> np.ndarray | None:
            polyline_idxs = np.array(
                [[value["polyline_index"][0], value["polyline_index"][1]] for value in polyline[key]],
                dtype=np.int32,
            )
            if polyline_idxs.shape[0] == 0:
                return None
            return polyline_idxs

        sdc_index = scenario_data["sdc_track_index"]
        trajs = scenario_data["track_infos"]["trajs"]
        num_agents, num_timesteps, _ = trajs.shape

        T_last = self.LAST_TIMESTEP_TO_CONSIDER[self.scenario_type]
        if num_timesteps < T_last:
            raise AssertionError(
                f"Scenario {scenario_data['scenario_id']} has only {num_timesteps} timesteps, "
                f"but expected at least {T_last} timesteps.",
            )
        trajs = trajs[:, :T_last, :]  # shape: [num_agents, T_last, dim]

        agent_distances_to_conflict_points = None
        conflict_points = None
        if conflict_points_data is not None:
            agent_distances_to_conflict_points = (
                None
                if conflict_points_data["agent_distances_to_conflict_points"] is None
                else conflict_points_data["agent_distances_to_conflict_points"][:, :T_last, :]
            )
            conflict_points = (
                None
                if conflict_points_data["all_conflict_points"] is None
                else conflict_points_data["all_conflict_points"]
            )
        timestamps = np.asarray(scenario_data["timestamps_seconds"], dtype=np.float32)[:T_last]
        num_conflict_points = 0 if conflict_points is None else conflict_points.shape[0]

        # TODO: improve this relevance criteria
        agent_relevance = np.zeros(num_agents, dtype=np.float32)
        tracks_to_predict = scenario_data["tracks_to_predict"]
        tracks_to_predict_index = np.asarray(tracks_to_predict["track_index"] + [sdc_index])
        tracks_to_predict_difficulty = np.asarray(tracks_to_predict["difficulty"] + [2.0])

        # Set agent_relevance for tracks_to_predict_index based on tracks_to_predict_difficulty
        for idx, difficulty in zip(tracks_to_predict_index, tracks_to_predict_difficulty, strict=False):
            agent_relevance[idx] = self.DIFFICULTY_WEIGHTS.get(difficulty, 0.0)

        # Extract static map information
        map_infos = scenario_data.get("map_infos")
        num_polylines, map_polylines = 0, None
        if map_infos is not None:
            map_polylines = map_infos["all_polylines"].astype(np.float32)  # shape: [N, 3] or [N, 3, 2]
            num_polylines = map_polylines.shape[0]
            lane_ids = get_polyline_ids(map_infos, "lane") if "lane" in map_infos else None
            lane_speed_limits_mph = get_speed_limit_mph(map_infos, "lane") if "lane" in map_infos else None
            lane_polyline_idxs = get_polyline_idxs(map_infos, "lane") if "lane" in map_infos else None
            road_line_ids = get_polyline_ids(map_infos, "road_line") if "road_line" in map_infos else None
            road_line_polyline_idxs = get_polyline_idxs(map_infos, "road_line") if "road_line" in map_infos else None
            road_edge_ids = get_polyline_ids(map_infos, "road_edge") if "road_edge" in map_infos else None
            road_edge_polyline_idxs = get_polyline_idxs(map_infos, "road_edge") if "road_edge" in map_infos else None
            crosswalk_ids = get_polyline_ids(map_infos, "crosswalk") if "crosswalk" in map_infos else None
            crosswalk_polyline_idxs = get_polyline_idxs(map_infos, "crosswalk") if "crosswalk" in map_infos else None
            speed_bump_ids = get_polyline_ids(map_infos, "speed_bump") if "speed_bump" in map_infos else None
            speed_bump_polyline_idxs = get_polyline_idxs(map_infos, "speed_bump") if "speed_bump" in map_infos else None
            stop_sign_ids = get_polyline_ids(map_infos, "stop_sign") if "stop_sign" in map_infos else None
            stop_sign_polyline_idxs = get_polyline_idxs(map_infos, "stop_sign") if "stop_sign" in map_infos else None
            stop_sign_lane_ids = [stop_sign["lane_ids"] for stop_sign in map_infos.get("stop_sign", {"lane_ids": []})]
        else:
            lane_ids = None
            lane_speed_limits_mph = None
            lane_polyline_idxs = None
            road_line_ids = None
            road_line_polyline_idxs = None
            road_edge_ids = None
            road_edge_polyline_idxs = None
            crosswalk_ids = None
            crosswalk_polyline_idxs = None
            speed_bump_ids = None
            speed_bump_polyline_idxs = None
            stop_sign_ids = None
            stop_sign_polyline_idxs = None
            stop_sign_lane_ids = []

        # Extract static and dynamic map information
        dynamic_map_infos = scenario_data.get("dynamic_map_infos")
        num_dynamic_stop_points = 0
        dynamic_stop_points = None
        dynamic_stop_points_lane_ids = None
        if dynamic_map_infos is not None:
            # For dynamic map information, we only need stop points for conflict points
            if "stop_point" in dynamic_map_infos and len(dynamic_map_infos["stop_point"]) > 0:
                dynamic_stop_points = dynamic_map_infos["stop_point"]  # shape: [N, 3] or [N, 3, 2]
                num_dynamic_stop_points = len(dynamic_stop_points)
                if num_dynamic_stop_points > 0 and len(dynamic_stop_points[0]) > 0:
                    dynamic_stop_points = dynamic_stop_points[0].astype(np.float32).squeeze(axis=0)  # shape: [N, 3]
                    dynamic_stop_points_lane_ids = dynamic_map_infos["lane_id"][0].astype(np.int32).squeeze(axis=0)

        try:
            # TODO: add type of lane, road, etc
            scenario = Scenario(
                num_agents=num_agents,
                scenario_id=scenario_data["scenario_id"],
                scenario_type=self.scenario_type,
                ego_index=sdc_index,
                ego_id=scenario_data["track_infos"]["object_id"][sdc_index],
                agent_ids=scenario_data["track_infos"]["object_id"],
                agent_types=scenario_data["track_infos"]["object_type"],
                agent_valid=trajs[..., self.AGENT_VALID].astype(np.bool_).squeeze(axis=-1),
                agent_positions=trajs[..., self.POS_XYZ_IDX],
                agent_velocities=trajs[..., self.VEL_XY_IDX],
                agent_lengths=trajs[..., self.AGENT_LENGTHS].squeeze(axis=-1),
                agent_widths=trajs[..., self.AGENT_WIDTHS].squeeze(axis=-1),
                agent_heights=trajs[..., self.AGENT_HEIGHTS].squeeze(axis=-1),
                agent_headings=trajs[..., self.HEADING_IDX].squeeze(axis=-1),
                agent_relevance=agent_relevance,
                last_observed_timestep=scenario_data["current_time_index"],
                total_timesteps=self.LAST_TIMESTEP,
                last_timestep_to_consider=T_last,
                stationary_speed=self.STATIONARY_SPEED,
                agent_to_agent_max_distance=self.AGENT_TO_AGENT_MAX_DISTANCE,
                agent_to_conflict_point_max_distance=self.AGENT_TO_CONFLICT_POINT_MAX_DISTANCE,
                agent_to_agent_distance_breach=self.AGENT_TO_AGENT_DISTANCE_BREACH,
                heading_threshold=self.HEADING_THRESHOLD,
                timestamps=timestamps,
                num_conflict_points=num_conflict_points,
                map_conflict_points=conflict_points,
                agent_distances_to_conflict_points=agent_distances_to_conflict_points,
                num_polylines=num_polylines,
                map_polylines=map_polylines,
                lane_ids=lane_ids,
                lane_speed_limits_mph=lane_speed_limits_mph,
                lane_polyline_idxs=lane_polyline_idxs,
                road_line_ids=road_line_ids,
                road_line_polyline_idxs=road_line_polyline_idxs,
                road_edge_ids=road_edge_ids,
                road_edge_polyline_idxs=road_edge_polyline_idxs,
                crosswalk_ids=crosswalk_ids,
                crosswalk_polyline_idxs=crosswalk_polyline_idxs,
                speed_bump_ids=speed_bump_ids,
                speed_bump_polyline_idxs=speed_bump_polyline_idxs,
                stop_sign_ids=stop_sign_ids,
                stop_sign_polyline_idxs=stop_sign_polyline_idxs,
                stop_sign_lane_ids=stop_sign_lane_ids,
                num_dynamic_stop_points=num_dynamic_stop_points,
                dynamic_stop_points=dynamic_stop_points,
                dynamic_stop_points_lane_ids=dynamic_stop_points_lane_ids,
            )
        except (ValidationError, TypeError) as e:
            raise e
        return scenario

    def check_conflict_points(self):
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
            agent_positions = scenario["track_infos"]["trajs"][:, :, self.POS_XYZ_IDX]
            conflict_points = self.find_conflict_points(static_map_infos, dynamic_map_infos, agent_positions)

            with open(conflict_points_filepath, "wb") as f:
                pickle.dump(conflict_points, f, protocol=pickle.HIGHEST_PROTOCOL)

            return conflict_points_filepath

        if self.parallel:
            from joblib import Parallel, delayed

            outs = Parallel(n_jobs=self.num_workers, batch_size=self.batch_size)(
                delayed(process_file)(scenario_id=scenario_id, scenario_path=scenario_path)
                for scenario_id, scenario_path in track(zipped, total=len(self.data.scenarios_ids))
            )
            self.data.conflict_points = natsorted(outs)
        else:
            for scenario_id, scenario_path in track(zipped, total=len(self.data.scenarios_ids)):
                out = process_file(scenario_id=scenario_id, scenario_path=scenario_path)
                self.data.conflict_points.append(out)

        self.data.conflict_points = natsorted(self.data.conflict_points)

        logger.info(f"Conflict points check completed in {time.time() - start:.2f} seconds.")

    def find_conflict_points(
        self,
        static_map_info: dict[str, Any],
        dynamic_map_info: dict[str, Any],
        agent_positions: np.ndarray,
    ) -> dict[str, Any]:
        """Finds the conflict points in the map for a scenario.

        Args:
            static_map_info (dict): The static map information.
            dynamic_map_info (dict): The dynamic map information.
            agent_positions (np.ndarray): Array of agent positions (shape: [N_agents, T, 3]).

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
            points = polylines[start:end][:, :3]
            points = resample(points, points.shape[0] * self.conflict_points_cfg.resample_factor)
            static_conflict_points_list.append(points)

        for conflict_point in static_map_info["stop_sign"]:
            start, end = conflict_point["polyline_index"]
            points = polylines[start:end][:, :3]
            static_conflict_points_list.append(points)

        static_conflict_points = (
            np.concatenate(static_conflict_points_list) if len(static_conflict_points_list) > 0 else np.empty((0, 3))
        )

        # Lane Intersections
        lane_infos = static_map_info["lane"]
        lanes = [polylines[li["polyline_index"][0] : li["polyline_index"][1]][:, :3] for li in lane_infos]  # noqa: E203
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

            D = np.linalg.norm(lane_i[:, None] - lane_j, axis=-1)
            i_idx, j_idx = np.where(self.conflict_points_cfg.intersection_threshold > D)

            # TODO: determine if two lanes are consecutive, but not entry/exit lanes. If this is the
            # case there'll be an intersection that is not a conflict point.
            start_i, end_i = i_idx[:5], i_idx[-5:]
            start_j, end_j = j_idx[:5], j_idx[-5:]
            if (np.any(start_i < 5) and np.any(end_j > lane_j.shape[0] - 5)) or (
                np.any(start_j < 5) and np.any(end_i > lane_i.shape[0] - 5)
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
        dynamic_conflict_points = np.empty((0, 3))
        if len(stops) > 0 and len(stops[0]) > 0:
            if stops[0].shape[1] == 3:
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

    def collate_batch(self, batch_data) -> dict[str, Any]:  # pyright: ignore[reportMissingParameterType]
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
