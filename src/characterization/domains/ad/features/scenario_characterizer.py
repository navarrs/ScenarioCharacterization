"""Combined AD feature extractor: individual + interaction features in a single call."""

from omegaconf import DictConfig, OmegaConf

from characterization.domains.ad.features.individual_features import IndividualFeatures
from characterization.domains.ad.features.interaction_features import InteractionFeatures
from characterization.domains.ad.schemas import Scenario, ScenarioFeatures


class ADScenarioCharacterizer:
    """Extracts both individual and interaction features from an AD scenario.

    Wraps :class:`IndividualFeatures` and :class:`InteractionFeatures` and merges their outputs
    into a single :class:`ScenarioFeatures` object, mirroring the interface of
    :class:`~characterization.domains.aviation.features.base_feature.AviationFeatures`.

    Args:
        config: Hydra DictConfig forwarded to both sub-extractors. Supports all keys accepted by
            :class:`IndividualFeatures` and :class:`InteractionFeatures` (e.g. ``return_criterion``,
            ``feature_type``, ``compute_agent_to_agent_closest_dists``).
        n_jobs: Number of worker processes for interaction feature computation. ``-1`` (default)
            uses all available CPUs; positive values cap the pool size.

    Example::

        extractor = ADScenarioCharacterizer(OmegaConf.create({"return_criterion": "critical"}))
        features = extractor.compute(scenario)
        print(features.model_dump())
    """

    def __init__(self, config: DictConfig | None = None, n_jobs: int = -1) -> None:
        """Instantiate sub-extractors from *config* and store the worker cap."""
        cfg = config if config is not None else OmegaConf.create({})
        self._individual = IndividualFeatures(cfg)
        self._interaction = InteractionFeatures(cfg)
        self._max_workers: int | None = None if n_jobs == -1 else n_jobs

    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Extract individual and interaction features from *scenario*.

        Args:
            scenario: The AD scenario to characterize.

        Returns:
            ScenarioFeatures with both ``individual_features`` and ``interaction_features`` populated.
        """
        ind = self._individual.compute(scenario)
        int_ = self._interaction.compute(scenario, max_workers=self._max_workers)
        return ScenarioFeatures(
            metadata=scenario.metadata,
            individual_features=ind.individual_features,
            interaction_features=int_.interaction_features,
            agent_to_agent_closest_dists=ind.agent_to_agent_closest_dists,
        )
