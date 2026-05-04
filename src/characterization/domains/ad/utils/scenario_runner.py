"""AD-domain scenario runner implementation."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
from omegaconf import DictConfig
from pydantic import BaseModel

from characterization.domains.ad.schemas.scenario import Scenario
from characterization.domains.ad.schemas.scenario_features import ScenarioFeatures
from characterization.domains.ad.schemas.scenario_scores import Score
from characterization.domains.ad.utils.scenario_characterization_utils import (
    build_scenario_summary,
    generate_group_summaries,
)
from characterization.domains.ad.utils.scenario_characterization_utils import (
    load_scenario as _load_scenario,
)
from characterization.schemas.scenario import BaseScenario
from characterization.schemas.scenario_scores import ScenarioScores
from characterization.utils.logging_utils import get_pylogger
from characterization.utils.scenario_runner import BaseScenarioRunner

if TYPE_CHECKING:
    from characterization.domains.ad.utils.scenario_visualizer.base_visualizer import ADBaseVisualizer

_LOGGER = get_pylogger(__name__)

_VIZ_CACHE_KEY = "visualizer"


class ADScenarioRunner(BaseScenarioRunner):
    """Scenario runner for the AD (autonomous driving) domain.

    Args:
        visualizer: Hydra sub-config for the scenario visualizer, or ``None`` to skip visualization.
    """

    def __init__(self, visualizer: DictConfig | None = None) -> None:
        """Store the optional visualizer Hydra sub-config."""
        super().__init__()
        self._visualizer_cfg = visualizer
        # _viz_cache[_VIZ_CACHE_KEY]: single ADBaseVisualizer instance (or None on failure)

    def load_scenario(self, pkl_path: Path, _group_id: str) -> Scenario | None:
        """Load an AD scenario from *pkl_path* (group ID is unused in the AD domain)."""
        return _load_scenario(pkl_path)

    def build_summary(
        self,
        scenario: BaseScenario,
        pkl_path: Path,
        features: BaseModel,
        scores: ScenarioScores,
        max_agents: int,
        max_pairs: int,
    ) -> str:
        """Build a human-readable text summary for a single AD scenario."""
        return build_scenario_summary(
            cast("Scenario", scenario),
            pkl_path,
            cast("ScenarioFeatures", features),
            scores,
            max_agents,
            max_pairs,
        )

    def visualize(
        self,
        scenario: BaseScenario,
        group_id: str,
        viz_dir: Path,
        *,
        scores: ScenarioScores | None = None,
    ) -> None:
        """Render and save a visualization for the AD *scenario*."""
        if self._visualizer_cfg is None:
            return

        if _VIZ_CACHE_KEY not in self._viz_cache:
            try:
                self._viz_cache[_VIZ_CACHE_KEY] = hydra.utils.instantiate(self._visualizer_cfg)
            except Exception:
                _LOGGER.exception("Failed to instantiate AD visualizer")
                self._viz_cache[_VIZ_CACHE_KEY] = None

        visualizer: ADBaseVisualizer | None = self._viz_cache[_VIZ_CACHE_KEY]  # type: ignore[assignment]
        if visualizer is None:
            return

        ad_scenario = cast("Scenario", scenario)
        viz_out_dir = viz_dir / group_id
        viz_out_dir.mkdir(parents=True, exist_ok=True)
        try:
            visualizer.visualize_scenario(ad_scenario, scores=_to_score(scores), output_dir=viz_out_dir)
        except Exception:
            _LOGGER.exception("Failed to visualize scenario %s", ad_scenario.metadata.scenario_id)

    def generate_group_summaries(
        self,
        group_id: str,
        group_features: list[BaseModel],
        group_scores: list[ScenarioScores],
        summaries_dir: Path | None,
        plots_dir: Path | None,
    ) -> None:
        """Write aggregate text summary and feature plot for an AD group."""
        ad_features = cast("list[ScenarioFeatures]", group_features)
        generate_group_summaries(group_id, ad_features, group_scores, summaries_dir, plots_dir)


def _to_score(scores: ScenarioScores | None) -> Score | None:
    """Convert the canonical ScenarioScores to the AD-internal Score type."""
    if scores is None:
        return None
    return Score(scene_score=scores.scene_score)
