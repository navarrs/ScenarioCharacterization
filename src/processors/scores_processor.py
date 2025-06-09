import os

from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from features import SUPPORTED_FEATURES
from features.base_feature import BaseFeature
from processors.base_processor import BaseProcessor
from scorer.base_scorer import BaseScorer
from utils.common import get_logger

logger = get_logger(__name__)


class ScoresProcessor(BaseProcessor):
    def __init__(
        self,
        config: DictConfig,
        dataset: Dataset,
        characterizer: BaseFeature | BaseScorer,
    ) -> None:
        """Initializes the ScoresProcessor with a configuration, dataset, and scorer.

        Args:
            config (DictConfig): Configuration for the scores processor, including parameters like
                batch size, number of workers, and whether to save the output.
            dataset (Dataset): The dataset to process, which should be a subclass of torch.utils.data.Dataset.
            characterizer (BaseFeature | BaseScorer): An instance of BaseScorer or its subclass that defines the scoring method.

        Raises:
            ValueError: If features or feature paths are not specified, or if unsupported features are requested.
            AssertionError: If the characterizer is not of type 'score'.
        """
        super(ScoresProcessor, self).__init__(config, dataset, characterizer)
        assert (
            self.characterizer.characterizer_type == "score"
        ), f"Expected characterizer of type 'feature', got {self.characterizer.characterizer_type}."

        self.features = config.get("features", None)
        if self.features is None:
            logger.error("Features must be specified in the configuration.")
            raise ValueError

        unsupported = [f for f in self.features if f not in SUPPORTED_FEATURES]
        if unsupported:
            logger.error(
                f"Features {unsupported} not in supported list {SUPPORTED_FEATURES}"
            )
            raise ValueError

        self.feature_path = config.get("feature_path", None)
        if not self.feature_path:
            logger.error("Feature paths must be specified in the configuration.")
            raise ValueError
        else:
            logger.info(f"Features will be loaded from {self.feature_path}")

    def run(self):
        """Runs the score processing on the dataset.

        Iterates over the dataset, loads features for each scenario, checks for missing features,
        computes scores, and saves them if required.

        Returns:
            None
        """
        logger.info(
            f"Processing {self.features} {self.characterizer.name} scores for {self.dataset.name}."
        )

        # TODO: Need more elegant iteration over the dataset to avoid the two-level for loop.
        for scenario_batch in tqdm(self.dataloader, desc="Processing scenarios"):
            for scenario in scenario_batch["scenario"]:

                scenario_id = scenario["scenario_id"]
                scenario_feature_file = os.path.join(
                    self.feature_path, f"{scenario_id}.pkl"
                )
                scenario_features = self.from_pickle(scenario_feature_file)

                # TODO: pre-check that features have been computed
                missing_features = [
                    f for f in self.features if f not in scenario_features
                ]
                if missing_features:
                    logger.error(
                        f"Scenario {scenario_id} is missing features: {missing_features}"
                    )
                    raise ValueError

                scores = self.characterizer.compute(
                    scenario=scenario, scenario_features=scenario_features
                )

                if self.save:
                    self.to_pickle(scores, scenario["scenario_id"])
