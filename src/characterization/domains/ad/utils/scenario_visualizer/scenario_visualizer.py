# pyright: reportUnknownMemberType=false
"""Static multi-pane scenario visualizer for AD scenarios."""

from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig

from characterization.domains.ad.schemas import Scenario, Score
from characterization.domains.ad.utils.scenario_visualizer.base_visualizer import ADBaseVisualizer, SupportedPanes
from characterization.utils.logging_utils import get_pylogger

logger = get_pylogger(__name__)


class ScenarioVisualizer(ADBaseVisualizer):
    """Static multi-pane scenario visualizer for AD scenarios."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the ScenarioVisualizer."""
        super().__init__(config)

    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: Score | None = None,
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Renders a static multi-pane PNG for the scenario.

        Args:
            scenario: AD scenario to visualize.
            scores: Optional scores for transparency weighting or categorical display.
            output_dir: Directory where the PNG will be saved.

        Returns:
            Path to the saved PNG file.
        """
        scenario_id = scenario.metadata.scenario_id
        suffix = (
            ""
            if SupportedPanes.HIGHLIGHT_RELEVANT_AGENTS not in self.panes_to_plot
            or scores is None
            or scores.scene_score is None
            else f"_{round(scores.scene_score, 2)}"
        )
        output_filepath = output_dir / f"{scenario_id}{suffix}.png"
        logger.info("Visualizing scenario to %s", output_filepath)

        axs = plt.subplots(1, self.num_panes_to_plot, figsize=(5 * self.num_panes_to_plot, 5))[1]
        self.plot_map_data(axs, scenario, self.num_panes_to_plot)

        for i, pane in enumerate(self.panes_to_plot):
            ax = axs[i] if self.num_panes_to_plot > 1 else axs
            match pane:
                case SupportedPanes.ALL_AGENTS:
                    self.plot_sequences(ax, scenario, scores, title="All Agents Trajectories")
                case SupportedPanes.HIGHLIGHT_RELEVANT_AGENTS:
                    if self.plot_categorical:
                        self.plot_sequences_categorical(
                            ax,
                            scenario,
                            scores,
                            title="Scenario with Agent Categorical Scores",
                        )
                    else:
                        self.plot_sequences(
                            ax,
                            scenario,
                            scores,
                            show_relevant=True,
                            title="Highlighted Relevant and SDC Agent Trajectories",
                        )
                case (
                    SupportedPanes.SCORED_INDIVIDUAL
                    | SupportedPanes.SCORED_INTERACTION
                    | SupportedPanes.COUNTERFACTUAL_PROBE
                ):
                    logger.warning("%s pane is not yet implemented for AD scenarios; skipping.", pane)
                case _:
                    error_message = f"Unsupported pane type: {pane}"
                    raise ValueError(error_message)

        self.set_axes(axs, scenario, self.num_panes_to_plot)
        if self.add_title:
            plt.suptitle(f"Scenario: {scenario_id}")
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        plt.close()
        return output_filepath
