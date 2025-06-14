import os

from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from characterization.features import SUPPORTED_FEATURES
from characterization.features.base_feature import BaseFeature
from characterization.processors.base_processor import BaseProcessor
from characterization.scorer.base_scorer import BaseScorer
from characterization.utils.common import from_pickle, get_logger, to_pickle
from characterization.utils.schemas import ScenarioFeatures, ScenarioScores

logger = get_logger(__name__)


class ScoresProcessor(BaseProcessor):
    def __init__(
        self,
        config: DictConfig,
        dataset: Dataset,
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
        super(ScoresProcessor, self).__init__(config, dataset, characterizer)
        if not self.characterizer.characterizer_type == "score":
            raise AssertionError(
                f"Expected characterizer of type 'score', got {self.characterizer.characterizer_type}."
            )

        self.features = config.get("features", None)
        if self.features is None:
            raise ValueError("Features must be specified in the configuration.")

        unsupported = [f for f in self.features if f not in SUPPORTED_FEATURES]
        if unsupported:
            raise ValueError(f"Features {unsupported} not in supported list {SUPPORTED_FEATURES}")

        self.feature_path = config.get("feature_path", None)
        if not self.feature_path:
            raise ValueError("Feature paths must be specified in the configuration.")
        else:
            logger.info(f"Features will be loaded from {self.feature_path}")

    def run(self):
        """Runs the score processing on the dataset.

        Iterates over the dataset, loads features for each scenario, checks for missing features,
        computes scores using the characterizer, and saves them if required.

        Returns:
            None
        """
        logger.info(f"Processing {self.features} {self.characterizer.name} scores for {self.dataset.name}.")

        # TODO: Need more elegant iteration over the dataset to avoid the two-level for loop.
        for scenario_batch in tqdm(self.dataloader, desc="Processing scenarios"):
            for scenario in scenario_batch["scenario"]:

                scenario_id = scenario.scenario_id
                scenario_feature_file = os.path.join(self.feature_path, f"{scenario_id}.pkl")
                scenario_features = from_pickle(scenario_feature_file)

                # TODO: pre-check that features have been computed
                missing_features = [f for f in self.features if f not in scenario_features]
                if missing_features:
                    raise ValueError(f"Scenario {scenario_id} is missing features: {missing_features}")
                scenario_features = ScenarioFeatures.model_validate(scenario_features)
                if scenario_features.num_agents != scenario.num_agents:
                    raise ValueError(
                        f"Scenario {scenario_id} has {scenario.num_agents} agents",
                        f"but features indicate {scenario_features.num_agents}.",
                    )

                scores: ScenarioScores = self.characterizer.compute(
                    scenario=scenario, scenario_features=scenario_features
                )

                if self.save:
                    to_pickle(self.output_path, scores.model_dump(), scenario_id)
