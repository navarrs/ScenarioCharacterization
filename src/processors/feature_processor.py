from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from features.base_feature import BaseFeature
from processors.base_processor import BaseProcessor
from scorer.base_scorer import BaseScorer
from utils.common import get_logger

logger = get_logger(__name__)


class FeatureProcessor(BaseProcessor):
    def __init__(
        self,
        config: DictConfig,
        dataset: Dataset,
        characterizer: BaseFeature | BaseScorer,
    ) -> None:
        """Initializes the FeatureProcessor with configuration, dataset, and feature.

        Args:
            config (DictConfig): Configuration for the feature processor, including parameters like
                batch size, number of workers, and whether to save the output.
            dataset (Dataset): The dataset to process, which should be a subclass of 
                torch.utils.data.Dataset.
            characterizer (BaseFeature | BaseScorer): An instance of BaseFeature or its subclass that 
                defines the feature to compute.

        Raises:
            AssertionError: If the characterizer is not of type 'feature'.
        """
        super(FeatureProcessor, self).__init__(config, dataset, characterizer)
        assert (
            self.characterizer.characterizer_type == "feature"
        ), f"Expected characterizer of type 'feature', got {self.characterizer.characterizer_type}."

    def run(self):
        """Runs the feature processing on the dataset.

        Iterates over the dataset and computes features for each scenario. If saving is enabled,
        features are saved to disk.

        Returns:
            None
        """
        logger.info(
            f"Processing {self.characterizer.name} features for {self.dataset.name}."
        )

        # TODO: Need more elegant iteration over the dataset to avoid the two-level for loop.
        for scenario_batch in tqdm(self.dataloader, desc="Processing scenarios"):
            for scenario in scenario_batch["scenario"]:
                # At this point, the scenario dictionary should be standarized regardless of the
                # dataset type. See docstring for the expected keys.
                feature = self.characterizer.compute(scenario)

                if self.save:
                    self.to_pickle(feature, scenario["scenario_id"])

        logger.info(
            f"Finished processing {self.characterizer.name} features for {self.dataset.name}."
        )
