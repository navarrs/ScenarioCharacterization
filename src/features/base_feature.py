from abc import ABC, abstractmethod
from omegaconf import DictConfig

class BaseFeature(ABC):
    def __init__(self, config: DictConfig) -> None:
        """
        Initialize the BaseFeature with a configuration.

        :param config: Configuration for the feature.
        """
        self.config = config

    def identify(self) -> str:
        """
        Identify the dataset.
        """
        return f"{self.__class__.__name__}"

    def compute(self, *args, **kwargs):
        pass
