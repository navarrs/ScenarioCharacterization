import numpy as np
import pickle
import time

from easydict import EasyDict
from omegaconf import DictConfig
from typing import Dict

from utils.datasets.dataset import BaseDataset
from utils.common import get_logger

logger = get_logger(__name__)

class WaymoData(BaseDataset):
    def __init__(self, config: DictConfig) -> None:
        super(WaymoData, self).__init__(config=config)

        # Waymo dataset masks
        # center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid
        self.AGENT_DIMS  = [False, False, False, True, True, True, False, False, False, False]
        self.HEADING_IDX = [False, False, False, False, False, False, True, False, False, False]
        self.POS_XY_IDX  = [True, True, False, False, False, False, False, False, False, False]
        self.POS_XYZ_IDX = [True, True, True, False, False, False, False, False, False, False]
        self.VEL_XY_IDX  = [False, False, False, False, False, False, False, True, True, False]
        self.AGENT_VALID = [False, False, False, False, False, False, False, False, False, True]

        # Interpolated stuff
        self.IPOS_XY_IDX  = [True, True, False, False, False, False, False]
        self.IPOS_SDZ_IDX = [False, False, True, True, True, False, False]
        self.IPOS_SD_IDX  = [False, False, False, True, True, False, False]
        self.ILANE_IDX    = [False, False, False, False, False, True, False]
        self.IVALID_IDX   = [False, False, False, False, False, False, True]

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

        self.LAST_TIMESTEP = 91
        self.HIST_TIMESTEP = 11
        self.STATIONARY_SPEED = 0.25  # m/s

    def load_data(self) -> None:
        """
        Load the dataset.
        """
        start = time.time()
        logger.info(f"Loading WOMD scenario base data from {self.scenario_base_path}")
        with open(self.scenario_meta_path, "rb") as f:
            self.data.metas = pickle.load(f)[:: self.step]
        self.data.scenarios_ids = [f'sample_{x["scenario_id"]}.pkl' for x in self.data.metas] 
        self.data.scenarios = [
            f'{self.scenario_base_path}/sample_{x["scenario_id"]}.pkl' for x in self.data.metas
        ] 
        logger.info(f"Loading took {time.time() - start} seconds.")

        # TODO: add support for sharding?
        self.shard() 
    
    def collate_batch(self, batch_data) -> EasyDict:
        batch_size = len(batch_data)
        # key_to_list = {}
        # for key in batch_data[0].keys():
        #     key_to_list[key] = [batch_data[idx][key] for idx in range(batch_size)]

        # input_dict = {}
        # for key, val_list in key_to_list.items():
        #     if key in ['scenario_id', 'num_agents', 'ego_index', 'ego_id', 'current_time_index']:
        #         input_dict[key] = np.asarray(val_list)
            
        return {
            'batch_size': batch_size,
            'scenario': batch_data,
        }
    
    def transform_scenario_data(self, scenario_data: Dict) -> Dict:
        """
        Transform the scene data into a format suitable for processing.

        Actors keys:
            'scenario_id'        -> size(1) 
            'current_time_index' -> size(1) 
            'timestamps_seconds' -> size(LAST_TIMESTEP)
            'sdc_track_index'    -> size(1)
            'track_infos'
                'object_id'      -> size(NUM_AGENTS)
                'object_type'    -> size(NUM_AGENTS)
                'trajs'          -> size(NUM_AGENTS, self.LAST_TIMESTEP, len(self.AGENT_DIMS)),
        
        TODO: Add map data keys:
            'map_infos', 
            'dynamic_map_infos'
                'lane_id'        
                'state'          
                'stop_point'     
            'objects_of_interest', 
            'tracks_to_predict'
        """
        sdc_index = scenario_data['sdc_track_index']
        trajs = scenario_data['track_infos']['trajs']
        return {
            'num_agents': trajs.shape[0],
            'scenario_id': scenario_data['scenario_id'],
            'ego_index': sdc_index,
            'ego_id': scenario_data['track_infos']['object_id'][sdc_index],
            'agent_ids': scenario_data['track_infos']['object_id'],
            'agent_types': scenario_data['track_infos']['object_type'],
            'agent_valid': trajs[:, :, self.AGENT_VALID],
            'agent_positions': trajs[:, :, self.POS_XYZ_IDX],
            'agent_velocities': trajs[:, :, self.VEL_XY_IDX],
            'agent_headings': trajs[:, :, self.HEADING_IDX],
            'current_time_index': scenario_data['current_time_index'],
            'timestamps': scenario_data['timestamps_seconds'],
        }
  
    def __getitem__(self, index: int) -> Dict:
        """ Gets a single scenario by index. 
        Args:
            index (int): Index of the scenario to retrieve.
        Returns:
            Dict: A dictionary containing the scenario ID, metadata, and scenario data.
        """
        with open(self.data.scenarios[index], 'rb') as f:
            scenario = pickle.load(f)
        
        # ------------------------------------
        # TODO: Figure out if needed
        scenario_meta = self.data.metas[index]
        # ------------------------------------
        return self.transform_scenario_data(scenario)