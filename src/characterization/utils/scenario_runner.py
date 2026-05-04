"""Abstract base class for domain-specific scenario runner implementations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from characterization.schemas.scenario import BaseScenario
from characterization.schemas.scenario_scores import ScenarioScores


class BaseScenarioRunner(ABC):
    """Encapsulates all domain-specific operations needed by the scenario characterization runner.

    Subclasses implement scenario loading, per-scenario summary/visualization, and group-level
    aggregate generation. Internal caches live inside the runner so the main loop stays clean.
    Domain-specific configuration (e.g. map directories, visualizer settings) is passed at
    construction time and stored as instance state.
    """

    def __init__(self) -> None:
        """Initialise the per-group data and visualizer caches."""
        self._group_data_cache: dict[str, Any] = {}
        self._viz_cache: dict[str, Any] = {}

    def group_key(self, pkl_path: Path) -> str:
        """Return the group identifier for a scenario file (defaults to parent directory name)."""
        return pkl_path.parent.name

    @abstractmethod
    def load_scenario(self, pkl_path: Path, group_id: str) -> BaseScenario | None:
        """Load a scenario from *pkl_path*.

        Args:
            pkl_path: Path to the scenario ``.pkl`` file.
            group_id: Group identifier (e.g. airport or dataset split) for cache lookups.

        Returns:
            The loaded scenario, or ``None`` if loading fails.
        """

    @abstractmethod
    def build_summary(
        self,
        scenario: BaseScenario,
        pkl_path: Path,
        features: BaseModel,
        scores: ScenarioScores,
        max_agents: int,
        max_pairs: int,
    ) -> str:
        """Build a human-readable text summary for a single scenario."""

    @abstractmethod
    def visualize(
        self,
        scenario: BaseScenario,
        group_id: str,
        viz_dir: Path,
        *,
        scores: ScenarioScores | None = None,
    ) -> None:
        """Render and save a visualization for *scenario*."""

    @abstractmethod
    def generate_group_summaries(
        self,
        group_id: str,
        group_features: list[BaseModel],
        group_scores: list[ScenarioScores],
        summaries_dir: Path | None,
        plots_dir: Path | None,
    ) -> None:
        """Write aggregate text summary and feature plot for an entire group."""
