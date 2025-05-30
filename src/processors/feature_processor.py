import hydra

from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.datasets.dataset import AbstractDataset
from src.features.base_feature import BaseFeature
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureProcessor:
    def __init__(
        self, config: DictConfig, dataset: AbstractDataset, feature: BaseFeature 
    ) -> None:
        self.parallel = config.parallel if "parallel" in config else True
        self.batch_size = config.batch_size if "batch_size" in config else 4
        self.num_processes = config.num_processes if "num_processes" in config else 10
        self.save = config.save if "save" in config else True


        self.dataset = dataset
        self.feature = feature

    def identify(self):
        """
        Identify the feature and dataset being processed.
        This method can be overridden by subclasses to provide specific identification.
        """
        return f"{self.__class__.__name__}"

    def run(self):
        logger.info(
            f"Processing {self.feature.identify()} features for {self.dataset.identify()}."
        )
        zipped = self.dataset.get_zipped()
        if self.parallel:
            from joblib import Parallel, delayed

            all_outs = Parallel(n_jobs=self.num_processes, batch_size=self.batch_size)(
                delayed(self.process_scenario)(
                    scenario=scenario, scenario_meta=scenario_meta
                )
                for scenario, scenario_meta in tqdm(zipped, total=len(self.dataset))
            )
        else:
            all_outs = []
            for scenario, scenario_meta in tqdm(zipped, total=len(self.dataset)):
                out = self.process_scenario(
                    scenario=scenario, scenario_meta=scenario_meta
                )
                all_outs.append(out)

        if self.save:
            self.save_cache()

    def process_scenario(self, scenario, scenario_meta):
        """
        Base method to process a file.
        Should be overridden by subclasses.

        :param file_path: Path to the file to process
        """
        pass

    def save_cache(self):
        pass
