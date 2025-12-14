from omegaconf import DictConfig
from tqdm import tqdm

from characterization.datasets import BaseDataset
from characterization.features import BaseFeature
from characterization.processors.base_processor import BaseProcessor
from characterization.schemas import ScenarioFeatures
from characterization.scorer.base_scorer import BaseScorer
from characterization.utils.io_utils import get_logger, to_pickle

logger = get_logger(__name__)


class FeatureProcessor(BaseProcessor):
    """Processor for computing and saving features from a dataset using a feature characterizer."""

    def __init__(
        self,
        config: DictConfig,
        dataset: BaseDataset,
        characterizer: BaseFeature | BaseScorer,
    ) -> None:
        """Initializes the FeatureProcessor with configuration, dataset, and feature characterizer.

        Args:
            config (DictConfig): Configuration for the feature processor, including parameters such as
                batch size, number of workers, shuffle, save, and output path.
            dataset (Dataset): The dataset to process. Must be a subclass of torch.utils.data.Dataset.
            characterizer (BaseFeature | BaseScorer): An instance of BaseFeature or BaseScorer that
                defines the feature computation logic.

        Raises:
            AssertionError: If the characterizer is not of type 'feature'.
        """
        super().__init__(config, dataset, characterizer)
        if self.characterizer.characterizer_type != "feature":
            error_message = f"Expected characterizer of type 'feature', got {self.characterizer.characterizer_type}."
            raise AssertionError(error_message)

    def run(self) -> None:
        """Runs the feature processing on the dataset.

        Iterates over the dataset and computes features for each scenario using the characterizer.
        If saving is enabled, features are serialized and saved to disk.

        Returns:
            None
        """
        logger.info(
            "Processing %s %s for %s",
            self.dataset.name,  # pyright: ignore[reportAttributeAccessIssue]
            self.characterizer.name,
            self.scenario_type,
        )

        # TODO: Need more elegant iteration over the dataset to avoid the two-level for loop.
        # for scenario_batch in track(self.dataloader, total=len(self.dataloader), description="Processing features"):
        for scenario_batch in tqdm(self.dataloader, total=len(self.dataloader), desc="Processing features..."):
            for scenario in scenario_batch["scenario"]:
                scenario_id = scenario.metadata.scenario_id
                features: ScenarioFeatures = self.characterizer.compute(scenario)  # pyright: ignore[reportCallIssue]

                if self.save:
                    to_pickle(
                        self.output_path,
                        features.model_dump(),
                        scenario_id,
                        overwrite=self.overwrite,
                        update=self.update,
                    )

        logger.info("Finished processing %s features for %s.", self.characterizer.name, self.dataset.name)  # pyright: ignore[reportAttributeAccessIssue]
