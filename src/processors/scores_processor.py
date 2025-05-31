import os
import pickle

from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from features import SUPPORTED_FEATURES
from scorer.base_scorer import BaseScorer
from utils.common import get_logger

logger = get_logger(__name__)

class ScoresProcessor:
    def __init__(self, config: DictConfig, dataset: Dataset, scorer: BaseScorer) -> None:
        """
        Initialize the ScoresProcessor with a configuration, dataset, and scorer.
        :param config: Configuration for the scores processor, including parameters like
                       batch size, number of workers, and whether to save the output.
        :param dataset: The dataset to process, which should be a subclass of torch.utils.data.Dataset.
        :param scorer: An instance of BaseScorer or its subclass that defines the scoring method.
        """
        self.scenario_type = config.scenario_type if "scenario_type" in config else 'gt'

        # DataLoader parameters
        self.batch_size = config.get("batch_size", 4)
        self.num_workers = config.get("num_worker", 4)
        self.shuffle = config.get("shuffle", False)

        self.features = config.get("features", None)
        if self.features is None:
            logger.error("Features must be specified in the configuration.")
            raise ValueError

        unsupported = [f for f in self.features if f not in SUPPORTED_FEATURES]
        if unsupported:
            logger.error(f"Features {unsupported} not in supported list {SUPPORTED_FEATURES}")
            raise ValueError

        self.dataset = dataset
        self.scorer = scorer

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
                logger.info(f"Scores {self.scorer.name} will be saved to {self.output_path}")
        
        self.feature_path = config.get("feature_path", None)
        if not self.feature_path:
            logger.error("Feature paths must be specified in the configuration.")
            raise ValueError
        else:
            logger.info(f"Features will be loaded from {self.feature_path}")
        
    def name(self):
        """
        Identify the feature and dataset being processed.
        This method can be overridden by subclasses to provide specific identification.
        """
        return f"{self.__class__.__name__}"

    def run(self):
        logger.info(f"Processing {self.features} {self.scorer.name} scores for {self.dataset.name}.")
        
        # TODO: Need more elegant iteration over the dataset to avoid the two-level for loop. 
        for scenario_batch in tqdm(self.dataloader, desc="Processing scenarios"):
            for scenario in scenario_batch['scenario']:
                
                scenario_id = scenario['scenario_id']
                # Get corresponding features for the current scenario 
                scenario_feature_file = os.path.join(self.feature_path, f"{scenario_id}.pkl")
                with open(scenario_feature_file, 'rb') as f:
                    scenario_features = pickle.load(f)
                
                # TODO: this might add too much overhead, probably need to optimize.
                # Maybe by saving all features for all scenarios in a single large file. This would
                # still require to check that the scenario features for a given scenario are in the 
                # file. 
                missing_features = [f for f in self.features if f not in scenario_features]
                if missing_features:
                    logger.error(f"Scenario {scenario_id} is missing features: {missing_features}")
                    raise ValueError

                scores = self.scorer.compute(scenario_features=scenario_features)
               
                score_data = {}
                scenario_score_file = os.path.join(self.output_path, f"{scenario_id}.pkl")
                if os.path.exists(scenario_score_file):
                    with open(scenario_score_file, 'rb') as f:
                        score_data = pickle.load(f)

                for key, value in scores.items():
                    score_data[key] = value
                
                with open(scenario_score_file, 'wb') as f:
                    pickle.dump(score_data, f, protocol=pickle.HIGHEST_PROTOCOL)