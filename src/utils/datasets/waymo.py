import itertools
import os
import pickle
import time
from typing import Dict

import numpy as np
from easydict import EasyDict
from natsort import natsorted
from omegaconf import DictConfig
from pydantic import ValidationError
from scipy.signal import resample
from tqdm import tqdm

from src.utils.common import get_logger
from src.utils.datasets.dataset import BaseDataset
from src.utils.schemas import Scenario

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
        self.AGENT_DIMS = [False,False,False,True,True,True,False,False,False,False]
        self.HEADING_IDX = [False,False,False,False,False,False,True,False,False,False]
        self.POS_XY_IDX = [True,True,False,False,False,False,False,False,False,False]
        self.POS_XYZ_IDX = [True,True,True,False,False,False,False,False,False,False]
        self.VEL_XY_IDX = [False,False,False,False,False,False,False,True,True,False]
        self.AGENT_VALID = [False,False,False,False,False,False,False,False,False,True]

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
        self.STATIONARY_SPEED = 0.25  # m/s

    def load_data(self) -> None:
        """Loads the dataset.

        Raises:
            AssertionError: If the number of scenarios and conflict points do not match.
        """
        start = time.time()
        logger.info(f"Loading WOMD scenario base data from {self.scenario_base_path}")
        with open(self.scenario_meta_path, "rb") as f:
            self.data.metas = pickle.load(f)[:: self.step]
        self.data.scenarios_ids = natsorted(
            [f'sample_{x["scenario_id"]}.pkl' for x in self.data.metas]
        )
        self.data.scenarios = natsorted(
            [
                f'{self.scenario_base_path}/sample_{x["scenario_id"]}.pkl'
                for x in self.data.metas
            ]
        )
        logger.info(f"Loading data took {time.time() - start} seconds.")

        # TODO: remove this
        self.shard()

        num_scenarios = len(self.data.scenarios_ids)

        # Pre-checks: conflict points
        self.check_conflict_points()
        num_conflict_points = len(self.data.conflict_points)
        assert (
            num_scenarios == num_conflict_points
        ), f"Mismatch in number of scenarios and conflict points: {num_scenarios} vs {num_conflict_points}"

    def transform_scenario_data(
        self, scenario_data: Dict, conflict_points: Dict
    ) -> Dict:
        """Transforms the scene data into a format suitable for processing.

        Args:
            scenario_data (Dict): The raw scenario data.
            conflict_points (Dict): The conflict points for the scenario.

        Returns:
            Dict: The transformed scenario data.
        """
        sdc_index = scenario_data["sdc_track_index"]
        trajs = scenario_data["track_infos"]["trajs"]
        num_agents = trajs.shape[0]

        # TODO: improve this relevance criteria
        agent_relevance = np.zeros(num_agents, dtype=np.float32)
        tracks_to_predict = scenario_data["tracks_to_predict"]
        tracks_to_predict_index = np.asarray(
            tracks_to_predict["track_index"] + [sdc_index]
        )
        tracks_to_predict_difficulty = np.asarray(
            tracks_to_predict["difficulty"] + [2.0]
        )

        # Set agent_relevance for tracks_to_predict_index based on tracks_to_predict_difficulty
        for idx, difficulty in zip(
            tracks_to_predict_index, tracks_to_predict_difficulty
        ):
            agent_relevance[idx] = self.DIFFICULTY_WEIGHTS.get(difficulty, 0.0)

        return {
            "num_agents": num_agents,
            "scenario_id": scenario_data["scenario_id"],
            "ego_index": sdc_index,
            "ego_id": scenario_data["track_infos"]["object_id"][sdc_index],
            "agent_ids": scenario_data["track_infos"]["object_id"],
            "agent_types": scenario_data["track_infos"]["object_type"],
            "agent_valid": trajs[:, :, self.AGENT_VALID].astype(np.bool_),
            "agent_positions": trajs[:, :, self.POS_XYZ_IDX],
            "agent_velocities": trajs[:, :, self.VEL_XY_IDX],
            "agent_headings": trajs[:, :, self.HEADING_IDX],
            "agent_relevance": agent_relevance,
            "last_observed_timestep": scenario_data["current_time_index"],
            "total_timesteps": self.LAST_TIMESTEP,
            "stationary_speed": self.STATIONARY_SPEED,
            "timestamps": np.asarray(
                scenario_data["timestamps_seconds"], dtype=np.float32
            ),
            "map_conflict_points": conflict_points["all_conflict_points"],
        }

    def check_conflict_points(self):
        """Checks if conflict points are already computed for each scenario.

        If not, computes them and saves to disk.
        """
        logger.info("Checking if conflict points have been computed for each scenario.")
        start = time.time()
        zipped = zip(self.data.scenarios_ids, self.data.scenarios)

        def process_file(scenario_id, scenario_path):
            conflict_points_filepath = os.path.join(
                self.conflict_points_path, scenario_id
            )
            if os.path.exists(conflict_points_filepath):
                return conflict_points_filepath

            # Otherwise compute conflict points
            with open(scenario_path, "rb") as f:
                scenario = pickle.load(f)

            static_map_infos = scenario["map_infos"]
            dynamic_map_infos = scenario["dynamic_map_infos"]
            conflict_points = self.find_conflict_points(
                static_map_infos, dynamic_map_infos
            )

            with open(conflict_points_filepath, "wb") as f:
                pickle.dump(conflict_points, f, protocol=pickle.HIGHEST_PROTOCOL)

            return conflict_points_filepath

        if self.parallel:
            from joblib import Parallel, delayed

            outs = Parallel(n_jobs=self.num_workers, batch_size=self.batch_size)(
                delayed(process_file)(
                    scenario_id=scenario_id, scenario_path=scenario_path
                )
                for scenario_id, scenario_path in tqdm(
                    zipped, total=len(self.data.scenarios_ids)
                )
            )
            self.data.conflict_points = natsorted(outs)
        else:
            for scenario_id, scenario_path in tqdm(
                zipped, total=len(self.data.scenarios_ids)
            ):
                out = process_file(scenario_id=scenario_id, scenario_path=scenario_path)
                self.data.conflict_points.append(out)

        self.data.conflict_points = natsorted(self.data.conflict_points)

        logger.info(
            f"Conflict points check completed in {time.time() - start:.2f} seconds."
        )

    def find_conflict_points(
        self, static_map_info: Dict, dynamic_map_info: Dict
    ) -> Dict:
        """Finds the conflict points in the map.

        Args:
            static_map_info (Dict): The static map information.
            dynamic_map_info (Dict): The dynamic map information.

        Returns:
            Dict: The conflict points in the map divided into:
                - 'static': (Ns, 3) static points (e.g., crosswalks, speed bumps, stop signs)
                - 'dynamic': (Nd, 3) dynamic points (e.g., traffic lights)
                - 'lane_intersections': (Nl, 3) intersections between lanes
                - 'all_conflict_points': (N, 3) all conflict points concatenated
        """
        polylines = static_map_info["all_polylines"]

        # Static Conflict Points: Crosswalks, Speed Bumps and Stop Signs
        static_conflict_points = []
        for conflict_point in (
            static_map_info["crosswalk"] + static_map_info["speed_bump"]
        ):
            start, end = conflict_point["polyline_index"]
            points = polylines[start:end][:, :3]
            points = resample(
                points, points.shape[0] * self.conflict_points_cfg.resample_factor
            )
            static_conflict_points.append(points)

        for conflict_point in static_map_info["stop_sign"]:
            start, end = conflict_point["polyline_index"]
            points = polylines[start:end][:, :3]
            static_conflict_points.append(points)

        static_conflict_points = (
            np.concatenate(static_conflict_points)
            if len(static_conflict_points) > 0
            else np.empty((0, 3))
        )

        # Lane Intersections
        lane_infos = static_map_info["lane"]
        lanes = [
            polylines[li["polyline_index"][0] : li["polyline_index"][1]][:, :3]
            for li in lane_infos
        ]
        # lanes = []
        # for lane_info in static_map_info['lane']:
        #     start, end = lane_info['polyline_index']
        #     lane = P[start:end]
        #     lane = signal.resample(lane, lane.shape[0] * resample_factor)
        #     lanes.append(lane)
        num_lanes = len(lanes)

        lane_combinations = list(itertools.combinations(range(num_lanes), 2))
        lane_intersections = []
        for i, j in lane_combinations:
            lane_i, lane_j = lanes[i], lanes[j]

            D = np.linalg.norm(lane_i[:, None] - lane_j, axis=-1)
            i_idx, j_idx = np.where(D < self.conflict_points_cfg.intersection_threshold)

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
                lane_intersections.append(lane_i[i_idx])

            if j_idx.shape[0] > 0:
                lane_intersections.append(lane_j[j_idx])

        lane_intersections = (
            np.concatenate(lane_intersections)
            if len(lane_intersections) > 0
            else np.empty((0, 3))
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

        conflict_points = (
            np.concatenate(conflict_point_list, dtype=np.float32)
            if len(conflict_point_list)
            else None
        )

        return {
            "static": static_conflict_points,
            "dynamic": dynamic_conflict_points,
            "lane_intersections": lane_intersections,
            "all_conflict_points": conflict_points,
        }

    def __getitem__(self, index: int) -> Dict:
        """Gets a single scenario by index.

        Args:
            index (int): Index of the scenario to retrieve.

        Returns:
            Dict: A dictionary containing the scenario ID, metadata, and scenario data.

        Raises:
            ValidationError: If the scenario data does not pass schema validation.
        """
        with open(self.data.scenarios[index], "rb") as f:
            scenario = pickle.load(f)

        with open(self.data.conflict_points[index], "rb") as f:
            conflict_points = pickle.load(f)

        # ------------------------------------
        # TODO: Figure out if needed
        scenario_meta = self.data.metas[index]
        # ------------------------------------
        scenario_data = self.transform_scenario_data(scenario, conflict_points)
        try:
            Scenario(**scenario_data)  # Validate scenario data
        except ValidationError as e:
            logger.error(f"Validation error for scenario {index}: {e}")
            raise e
        return scenario_data

    def collate_batch(self, batch_data) -> EasyDict:
        """Collates a batch of scenario data.

        Args:
            batch_data (list): List of scenario data dictionaries.

        Returns:
            EasyDict: A dictionary containing the batch size and the batch of scenarios.
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
