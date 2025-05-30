from abc import ABC, abstractmethod


class AbstractDataset(ABC):
    def __init__(self):
        super().__init__()
        self.scenario_base_path = None

    def identify(self) -> str:
        """
        Identify the dataset.
        """
        return f"{self.__class__.__name__}: from {self.scenario_base_path}"

    @abstractmethod
    def __len__(self):
        """Return the number of items in the dataset."""
        raise NotImplementedError("Method __len__ is not implemented yet.")

    @abstractmethod
    def load_data(self):
        """Load the dataset."""
        raise NotImplementedError("Method load_data is not implemented yet.")
