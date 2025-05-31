import pickle
import time

from omegaconf import DictConfig


from utils.datasets.dataset import BaseDataset
from utils.logger import get_logger

logger = get_logger(__name__)

class WaymoData(BaseDataset):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        # TODO: Generalize dataset specific attributes
        # center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y
        self.AGENT_DIMS  = [False, False, False, True, True, True, False, False, False]
        self.HEADING_IDX = [False, False, False, False, False, False, True, False, False]
        self.POS_XY_IDX  = [True, True, False, False, False, False, False, False, False]
        self.POS_XYZ_IDX = [True, True, True, False, False, False, False, False, False]
        self.VEL_XY_IDX  = [False, False, False, False, False, False, False, True, True]

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

        self.shard() 

    def get_zipped(self):
        """
        Get the zipped scenarios.

        Returns:
            list: List of tuples containing scenario name and path.
        """
        return zip(self.data.scenarios_ids, self.data.scenarios, self.data.metas)