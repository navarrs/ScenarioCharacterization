import re
from abc import ABC, abstractmethod


class BaseFeature(ABC):
    """Domain-agnostic abstract base class for scenario feature extractors."""

    characterizer_type: str = "feature"

    @property
    def name(self) -> str:
        """Return the class name formatted as lowercase with spaces."""
        return re.sub(r"(?<!^)([A-Z])", r" \1", self.__class__.__name__).lower()

    @abstractmethod
    def compute(self, scenario: object) -> object:
        """Compute features for a given scenario."""
