"""Aviation-domain scenario runner: thin wrapper over existing aviation utilities."""

import copy
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
from omegaconf import DictConfig
from pydantic import BaseModel

from characterization.domains.aviation.schemas.scenario import MapData, Scenario
from characterization.domains.aviation.schemas.scenario_features import ScenarioFeatures
from characterization.domains.aviation.utils.file_io_utils import load_scenario as _load_scenario
from characterization.domains.aviation.utils.scenario_characterization_utils import (
    build_scenario_summary,
    generate_airport_summaries,
)
from characterization.schemas.scenario import BaseScenario
from characterization.schemas.scenario_scores import ScenarioScores
from characterization.utils.logging_utils import get_pylogger
from characterization.utils.scenario_runner import BaseScenarioRunner

if TYPE_CHECKING:
    from characterization.domains.aviation.utils.scenario_visualizer.scenario_visualizer import ScenarioVisualizer

_LOGGER = get_pylogger(__name__)


class AviationScenarioRunner(BaseScenarioRunner):
    """Scenario runner for the aviation domain.

    Wraps the existing aviation utility functions for loading, summarizing, visualizing, and
    generating group-level summaries. All domain-specific configuration is stored at construction.

    Args:
        maps_dir: Directory containing per-airport map pickle files, or ``None`` to skip map loading.
        visualizer: Hydra sub-config for the scenario visualizer, or ``None`` to skip visualization.
    """

    def __init__(
        self,
        maps_dir: Path | str | None = None,
        visualizer: DictConfig | None = None,
    ) -> None:
        """Store maps directory and optional visualizer Hydra sub-config."""
        super().__init__()
        self._maps_dir = Path(maps_dir) if maps_dir else None
        self._visualizer_cfg = visualizer
        # _group_data_cache: MapData | None per airport_id
        # _viz_cache: ScenarioVisualizer | None per airport_id

    def load_scenario(self, pkl_path: Path, group_id: str) -> Scenario | None:
        """Load an aviation scenario from *pkl_path*, injecting map data for *group_id*."""
        map_cache: dict[str, MapData | None] = self._group_data_cache  # type: ignore[assignment]
        return _load_scenario(pkl_path, self._maps_dir, group_id, map_cache)

    def build_summary(
        self,
        scenario: BaseScenario,
        pkl_path: Path,
        features: BaseModel,
        scores: ScenarioScores,
        max_agents: int,
        max_pairs: int,
    ) -> str:
        """Build a human-readable text summary for a single aviation scenario."""
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
        """Render and save a visualization for the aviation *scenario*."""
        if self._visualizer_cfg is None:
            return
        viz_cache: dict[str, ScenarioVisualizer | None] = self._viz_cache  # type: ignore[assignment]
        if group_id not in viz_cache:
            visualizer_cfg = copy.deepcopy(self._visualizer_cfg)
            visualizer_cfg.airport = group_id
            try:
                viz_cache[group_id] = hydra.utils.get_class(visualizer_cfg._target_)(visualizer_cfg)
            except Exception:
                _LOGGER.exception("Failed to instantiate visualizer for airport %s", group_id)
                viz_cache[group_id] = None

        visualizer = viz_cache[group_id]
        if visualizer is None:
            return

        viz_out_dir = viz_dir / group_id
        viz_out_dir.mkdir(parents=True, exist_ok=True)
        try:
            visualizer.visualize_scenario(cast("Scenario", scenario), scores=scores, output_dir=viz_out_dir)
        except Exception:
            _LOGGER.exception(
                "Failed to visualize scenario %s",
                cast("Scenario", scenario).metadata.scenario_id,
            )

    def generate_group_summaries(
        self,
        group_id: str,
        group_features: list[BaseModel],
        group_scores: list[ScenarioScores],
        summaries_dir: Path | None,
        plots_dir: Path | None,
    ) -> None:
        """Write aggregate text summary and feature plot for an aviation airport group."""
        aviation_features = cast("list[ScenarioFeatures]", group_features)
        generate_airport_summaries(
            {group_id: aviation_features},
            {group_id: group_scores},
            summaries_dir,
            plots_dir,
        )
