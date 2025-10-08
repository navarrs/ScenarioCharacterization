import os

from omegaconf import DictConfig
from tqdm import tqdm

from characterization.features import SUPPORTED_FEATURES, BaseFeature
from characterization.processors.base_processor import BaseProcessor
from characterization.schemas import ScenarioFeatures, ScenarioScores
from characterization.scorer import BaseScorer
from characterization.utils.datasets import BaseDataset
from characterization.utils.io_utils import from_pickle, get_logger, to_pickle

logger = get_logger(__name__)


class ScoresProcessor(BaseProcessor):
    """Processor for computing and saving scores from a dataset using a scoring characterizer."""

    def __init__(
        self,
        config: DictConfig,
        dataset: BaseDataset,
        characterizer: BaseFeature | BaseScorer,
    ) -> None:
        """Initializes the ScoresProcessor with configuration, dataset, and scorer.

        Args:
            config (DictConfig): Configuration for the scores processor, including parameters such as
                batch size, number of workers, features to use, feature paths, and whether to save the output.
            dataset (Dataset): The dataset to process. Must be a subclass of torch.utils.data.Dataset.
            characterizer (BaseFeature | BaseScorer): An instance of BaseFeature or BaseScorer that
                defines the scoring logic.

        Raises:
            ValueError: If features or feature paths are not specified, or if unsupported features are requested.
            AssertionError: If the characterizer is not of type 'score'.
        """
        super().__init__(config, dataset, characterizer)
        if self.characterizer.characterizer_type != "score":
            error_message = f"Expected characterizer of type 'score', got {self.characterizer.characterizer_type}."
            raise AssertionError(error_message)

        self.features = config.get("features", None)
        if self.features is None:
            error_message = "Features must be specified in the configuration."
            raise ValueError(error_message)

        unsupported = [f for f in self.features if f not in SUPPORTED_FEATURES]
        if unsupported:
            error_message = f"Features {unsupported} not in supported list {SUPPORTED_FEATURES}"
            raise ValueError(error_message)

        self.feature_path = config.get("feature_path", None)
        if not self.feature_path:
            error_message = "Feature paths must be specified in the configuration."
            raise ValueError(error_message)
        logger.info("Features will be loaded from %s", self.feature_path)

    def run(self) -> None:
        """Runs the score processing on the dataset.

        Iterates over the dataset, loads features for each scenario, checks for missing features,
        computes scores using the characterizer, and saves them if required.

        Returns:
            None
        """
        logger.info("Processing %s %s for %s.", self.features, self.characterizer.name, self.dataset.name)

        # TODO: Need more elegant iteration over the dataset to avoid the two-level for loop.
        # for scenario_batch in track(self.dataloader, total=len(self.dataloader), description="Processing scores..."):
        for scenario_batch in tqdm(self.dataloader, total=len(self.dataloader), desc="Processing scores..."):
            for scenario in scenario_batch["scenario"]:
                scenario_id = scenario.metadata.scenario_id
                scenario_feature_file = os.path.join(self.feature_path, f"{scenario_id}.pkl")
                scenario_features = from_pickle(scenario_feature_file)

                # TODO: pre-check that features have been computed
                scenario_features = ScenarioFeatures.model_validate(scenario_features)

                scores: ScenarioScores = self.characterizer.compute(
                    scenario=scenario,
                    scenario_features=scenario_features,  # pyright: ignore[reportCallIssue]
                )

                if self.save:
                    to_pickle(self.output_path, scores.model_dump(), scenario_id, overwrite=self.overwrite)
