from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig

from characterization.schemas import Scenario, ScenarioScores
from characterization.utils.io_utils import get_logger
from characterization.utils.viz.visualizer import BaseVisualizer, SupportedPanes

logger = get_logger(__name__)


class ScenarioVisualizer(BaseVisualizer):
    """Visualizer for scenarios."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the ScenarioVisualizer with the given configuration."""
        super().__init__(config)

    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: ScenarioScores | None = None,
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Visualizes a single scenario and saves the output to a file.

        WaymoVisualizer visualizes the scenario on two windows:
            window 1: displays the full scene zoomed out
            window 2: displays the scene with relevant agents in different colors.

        Args:
            scenario (Scenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            output_dir (str): the directory where to save the scenario visualization.

        Returns:
            Path: The path to the saved visualization file.
        """
        scenario_id = scenario.metadata.scenario_id
        suffix = (
            ""
            if SupportedPanes.HIGHLIGHT_RELEVANT_AGENTS not in self.panes_to_plot
            or scores is None
            or scores.safeshift_scores is None
            or scores.safeshift_scores.scene_score is None
            else f"_{round(scores.safeshift_scores.scene_score, 2)}"
        )
        output_filepath = output_dir / f"{scenario_id}{suffix}.png"
        logger.info("Visualizing scenario to %s", output_filepath)

        # Plot static and dynamic map information in the scenario
        axs = plt.subplots(1, self.num_panes_to_plot, figsize=(5 * self.num_panes_to_plot, 5 * 1))[1]
        self.plot_map_data(axs, scenario, self.num_panes_to_plot)

        for i, pane in enumerate(self.panes_to_plot):
            match pane:
                case SupportedPanes.ALL_AGENTS:
                    self.plot_sequences(
                        axs[i] if self.num_panes_to_plot > 1 else axs, scenario, scores, title="All Agents Trajectories"
                    )
                case SupportedPanes.HIGHLIGHT_RELEVANT_AGENTS:
                    # Plot trajectory data with relevant agents in a different color
                    self.plot_sequences(
                        axs[i] if self.num_panes_to_plot > 1 else axs,
                        scenario,
                        scores,
                        show_relevant=True,
                        title="Highlighted Relevant and SDC Agent Trajectories",
                    )
                case _:
                    error_message = f"Pane '{pane}' is not supported or not implemented yet."
                    raise ValueError(error_message)

        # Prepare and save plot
        self.set_axes(axs, scenario, self.num_panes_to_plot)
        if self.add_title:
            plt.suptitle(f"Scenario: {scenario_id}")
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        plt.close()
        return output_filepath
