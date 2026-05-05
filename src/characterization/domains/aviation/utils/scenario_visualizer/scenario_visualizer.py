"""Static multi-pane scenario visualizer for aviation scenarios."""

from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig

from characterization.domains.aviation.schemas.scenario import Scenario
from characterization.domains.aviation.schemas.scenario_scores import ScenarioScores
from characterization.domains.aviation.utils.scenario_visualizer.base_visualizer import (
    AviationBaseVisualizer,
    SupportedPanes,
)
from characterization.utils.logging_utils import get_pylogger

_LOGGER = get_pylogger(__name__)


class ScenarioVisualizer(AviationBaseVisualizer):
    """Static multi-pane scenario visualizer for aviation scenarios.

    Renders one pane per entry in ``panes_to_plot``. Panes that cannot be drawn (e.g. missing
    probe metadata) are skipped and the remaining panes are still rendered.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes the ScenarioVisualizer."""
        super().__init__(config)

    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: ScenarioScores | None = None,
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Renders a static multi-pane PNG for the scenario.

        Args:
            scenario: Aviation scenario to visualize.
            scores: Optional scores for SCORED_INDIVIDUAL / SCORED_INTERACTION panes.
            output_dir: Directory where the PNG will be saved.

        Returns:
            Path to the saved PNG file.
        """
        scenario_id = scenario.metadata.scenario_id
        output_filepath = output_dir / f"{scenario_id}.png"

        axs = plt.subplots(1, self.num_panes_to_plot, figsize=(5 * self.num_panes_to_plot, 5))[1]
        self.plot_map_data(axs, self.num_panes_to_plot)

        for i, pane in enumerate(self.panes_to_plot):
            ax = axs[i] if self.num_panes_to_plot > 1 else axs
            match pane:
                case SupportedPanes.ALL_AGENTS:
                    self.plot_sequences(ax, scenario, title="All Agents")
                case SupportedPanes.HIGHLIGHT_RELEVANT_AGENTS:
                    self.plot_sequences(ax, scenario, show_relevant=True, title="Highlighted Relevant Trajectories")
                case SupportedPanes.SCORED_INDIVIDUAL:
                    if scores is None or not scores.individual_scores:
                        _LOGGER.warning(
                            "SCORED_INDIVIDUAL pane requested but no individual scores available; skipping.",
                        )
                        continue
                    self.plot_sequences_with_scores(ax, scenario, scores.individual_scores, title="Individual Scores")
                case SupportedPanes.SCORED_INTERACTION:
                    if scores is None or not scores.interaction_scores:
                        _LOGGER.warning(
                            "SCORED_INTERACTION pane requested but no interaction scores available; skipping.",
                        )
                        continue
                    self.plot_sequences_with_scores(ax, scenario, scores.interaction_scores, title="Interaction Scores")
                case SupportedPanes.COUNTERFACTUAL_PROBE:
                    if scenario.critical_probe is None:
                        _LOGGER.warning(
                            "COUNTERFACTUAL_PROBE pane requested but scenario.critical_probe is None; skipping.",
                        )
                        continue
                    self.plot_sequences_with_probe(ax, scenario)
                case _:
                    error_message = f"Unsupported pane type: {pane}"
                    raise ValueError(error_message)

        self.configure_axes(axs, scenario, self.num_panes_to_plot)
        if self.add_title:
            plt.suptitle(f"Scenario: {scenario_id}")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        plt.close()

        _LOGGER.info("Visualized scenario to %s", output_filepath)
        return output_filepath
