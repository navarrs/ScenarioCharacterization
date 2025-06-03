from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import Dataset

from src.features.base_feature import BaseFeature
from src.processors.base_processor import BaseProcessor
from src.scorer.base_scorer import BaseScorer
from src.utils.common import get_logger

logger = get_logger(__name__)

class FeatureProcessor(BaseProcessor):
    def __init__(
        self, config: DictConfig, dataset: Dataset, characterizer: BaseFeature | BaseScorer
    ) -> None:
        
        """ Initialize the FeatureProcessor with a configuration, dataset, and feature.
        :param config: Configuration for the feature processor, including parameters like
                       batch size, number of workers, and whether to save the output.
        :param dataset: The dataset to process, which should be a subclass of torch.utils.data.Dataset.
        :param feature: An instance of BaseFeature or its subclass that defines the feature to compute.
        """
        super(FeatureProcessor, self).__init__(config, dataset, characterizer)
        assert self.characterizer.characterizer_type == 'feature', \
            f"Expected characterizer of type 'feature', got {self.characterizer.characterizer_type}."

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
        logger.info(f"Processing {self.characterizer.name} features for {self.dataset.name}.")
        
        # TODO: Need more elegant iteration over the dataset to avoid the two-level for loop. 
        for scenario_batch in tqdm(self.dataloader, desc="Processing scenarios"):
            for scenario in scenario_batch['scenario']:
                # At this point, the scenario dictionary should be standarized regardless of the 
                # dataset type. See docstring for the expected keys.  
                feature = self.characterizer.compute(scenario)
                
                if self.save:
                    self.to_pickle(feature, scenario['scenario_id'])
                    
        logger.info(f"Finished processing {self.characterizer.name} features for {self.dataset.name}.")