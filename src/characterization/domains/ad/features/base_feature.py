from abc import abstractmethod

from omegaconf import DictConfig

from characterization.domains.ad.schemas import Scenario, ScenarioFeatures
from characterization.features.base_feature import BaseFeature
from characterization.utils.common import ReturnCriterion
from characterization.utils.logging_utils import get_pylogger

logger = get_pylogger(__name__)


class ADBaseFeature(BaseFeature):
    """Base class for AD (autonomous driving) feature extractors.

    Attributes:
        config (DictConfig): Configuration parameters for the feature extractor.
        characterizer_type (str): Type identifier for the characterizer, always "feature".
        return_criterion (ReturnCriterion): Criterion determining when to return results.
        compute_agent_to_agent_closest_dists (bool): Whether to compute pairwise closest distances.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize with Hydra config; sets characterizer_type, return_criterion, and closest-dist flag."""
        self.config = config
        self.characterizer_type = "feature"
        self.return_criterion = ReturnCriterion[config.get("return_criterion", "critical").upper()]
        self.compute_agent_to_agent_closest_dists = config.get("compute_agent_to_agent_closest_dists", False)

    @abstractmethod
    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Compute features for the given AD scenario."""
        ...
