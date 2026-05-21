from omegaconf import DictConfig
from tqdm import tqdm

from characterization.datasets import BaseDataset
from characterization.features import BaseFeature
from characterization.processors.base_processor import BaseProcessor
from characterization.schemas import ScenarioFeatures
from characterization.utils.io_utils import get_logger, to_pickle

logger = get_logger(__name__)


class FeatureProcessor(BaseProcessor[BaseFeature]):
    """Processor for computing and saving features from a dataset using a feature characterizer."""

    def __init__(self, config: DictConfig, dataset: BaseDataset, characterizer: BaseFeature) -> None:
        """Initializes the FeatureProcessor with configuration, dataset, and feature characterizer.

        Args:
            config (DictConfig): Configuration for the feature processor, including parameters such as
                batch size, number of workers, shuffle, save, and output path.
            dataset (Dataset): The dataset to process. Must be a subclass of torch.utils.data.Dataset.
            characterizer (BaseFeature): The feature extractor to apply across the dataset scenarios.
        """
        super().__init__(config, dataset, characterizer)

    def run(self) -> None:
        """Runs the feature processing on the dataset.

        Iterates over the dataset and computes features for each scenario using the characterizer.
        If saving is enabled, features are serialized and saved to disk.

        Returns:
            None
        """
        logger.info("Processing %s %s for %s", self.dataset.name, self.characterizer.name, self.scenario_type)

        # TODO: Need more elegant iteration over the dataset to avoid the two-level for loop.
        # for scenario_batch in track(self.dataloader, total=len(self.dataloader), description="Processing features"):
        for scenario_batch in tqdm(self.dataloader, total=len(self.dataloader), desc="Processing features..."):
            for scenario in scenario_batch["scenario"]:
                scenario_id = scenario.metadata.scenario_id
                features: ScenarioFeatures = self.characterizer.compute(scenario)

                if self.save:
                    to_pickle(
                        self.output_path,
                        features.model_dump(),
                        scenario_id,
                        overwrite=self.overwrite,
                        update=self.update,
                    )

        logger.info("Finished processing %s features for %s.", self.characterizer.name, self.dataset.name)
