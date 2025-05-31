import os
import pickle

from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from src.features.base_feature import BaseFeature
from utils.common import get_logger

logger = get_logger(__name__)

class FeatureProcessor:
    def __init__(
        self, config: DictConfig, dataset: Dataset, feature: BaseFeature 
    ) -> None:
        """ Initialize the FeatureProcessor with a configuration, dataset, and feature.
        :param config: Configuration for the feature processor, including parameters like
                       batch size, number of workers, and whether to save the output.
        :param dataset: The dataset to process, which should be a subclass of torch.utils.data.Dataset.
        :param feature: An instance of BaseFeature or its subclass that defines the feature to compute.
        """
        self.scenario_type = config.scenario_type if "scenario_type" in config else 'gt'

        # DataLoader parameters
        self.batch_size = config.get("batch_size", 4)
        self.num_workers = config.get("num_worker", 4)
        self.shuffle = config.get("shuffle", False)

        self.dataset = dataset
        self.feature = feature

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_batch
        )

        self.save = config.get("save", True)
        self.output_path = config.get("output_path", None)
        if self.save:
            if self.output_path is None:
                logger.error("Output path must be specified in the configuration.")
                raise ValueError
            else: 
                logger.info(f"Features {self.feature.name} will be saved to {self.output_path}")

    @property
    def name(self):
        """ Identify the feature and dataset being processed. """
        return f"{self.__class__.__name__}"

    def run(self):
        """ Run the feature processing on the dataset.

        A single scenario is a dictionary containing the following keys:
            'num_agents'         -> int, number of agents in the scenario
            'scenario_id'        -> str, unique identifier for the scenario
            'ego_index'          -> int, index of the ego vehicle in the scenario  
            'ego_id'             -> int, unique identifier for the ego vehicle
            'agent_ids'          -> np.ndarray(num_agents,), unique identifiers for each agent
            'agent_types'        -> np.ndarray(num_agents,), types of each agent
            'agent_valid'        -> np.ndarray(num_agents, timesteps), if an agent's timestep is valid
            'agent_positions'    -> np.ndarray(num_agents, timesteps, 3), agent [x, y, z] positions
            'agent_velocities'   -> np.ndarray(num_agents, timesteps, 2), agent [vx, vy] velocities  
            'agent_headings'     -> np.ndarray(num_agents, timesteps, 1), agent headings in radians
            'current_time_index' -> int, index of the last observed time step in the scenario
            'timestamps'         ->  np.ndarray(timesteps,), scenario timestamps for each time step
        """
        logger.info(f"Processing {self.feature.name} features for {self.dataset.name}.")
        
        # TODO: Need more elegant iteration over the dataset to avoid the two-level for loop. 
        for scenario_batch in tqdm(self.dataloader, desc="Processing scenarios"):
            for scenario in scenario_batch['scenario']:
                
                # At this point, the scenario dictionary should be standarized regardless of the 
                # dataset type. See docstring for the expected keys.  
                feature = self.feature.compute(scenario)
            
                # TODO: make the saver a separate class to support arbitrary saving formats?
                if self.save:
                    scenario_id = scenario['scenario_id']
                    feature_data = {}
                    scenario_feature_file = os.path.join(self.output_path, f"{scenario_id}.pkl")
                    if os.path.exists(scenario_feature_file):
                        with open(scenario_feature_file, 'rb') as f:
                            feature_data = pickle.load(f)

                    for key, value in feature.items():
                        feature_data[key] = value
                    
                    with open(scenario_feature_file, 'wb') as f:
                        pickle.dump(feature_data, f, protocol=pickle.HIGHEST_PROTOCOL)