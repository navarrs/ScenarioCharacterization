from abc import ABC, abstractmethod

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from characterization.features.base_feature import BaseFeature
from characterization.scorer.base_scorer import BaseScorer
from characterization.utils.common import get_logger

logger = get_logger(__name__)


class BaseProcessor(ABC):
    def __init__(
        self,
        config: DictConfig,
        dataset: Dataset,
        characterizer: BaseFeature | BaseScorer,
    ) -> None:
        """Initializes the BaseProcessor with configuration, dataset, and characterizer.

        Args:
            config (DictConfig): Configuration for the processor. Should include parameters such as batch size,
                number of workers, shuffle, save, output path, and scenario type.
            dataset (Dataset): The dataset to process. Must be a subclass of torch.utils.data.Dataset and implement
                a collate_batch method.
            characterizer (BaseFeature | BaseScorer): An instance of a feature extractor or scorer to apply across the
                dataset scenarios.

        Raises:
            ValueError: If saving is enabled but no output path is specified in the configuration.
        """
        super(BaseProcessor, self).__init__()

        self.scenario_type = config.scenario_type if "scenario_type" in config else "gt"
        self.dataset = dataset
        self.characterizer = characterizer

        # DataLoader parameters
        self.batch_size = config.get("batch_size", 4)
        self.num_workers = config.get("num_workers", 4)
        self.shuffle = config.get("shuffle", False)

        self.save = config.get("save", True)
        self.output_path = config.get("output_path", None)
        if self.save:
            if self.output_path is None:
                raise ValueError("Output path must be specified in the configuration.")
            else:
                logger.info(f"Features {self.characterizer.name} will be saved to {self.output_path}")

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_batch,
        )

    @property
    def name(self):
        """Returns the name of the processor class.

        Returns:
            str: The name of the processor class.
        """
        return f"{self.__class__.__name__}"

    @abstractmethod
    def run(self):
        """Runs the processor on the dataset.

        This method must be implemented by subclasses to define the processing logic.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass
