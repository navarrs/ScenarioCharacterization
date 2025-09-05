import matplotlib.pyplot as plt
from characterization.schemas import Scenario, ScenarioScores
from characterization.utils.io_utils import get_logger
from omegaconf import DictConfig

from characterization.utils.viz.visualizer import BaseVisualizer

logger = get_logger(__name__)


class WaymoVisualizer(BaseVisualizer):
    """Visualizer for Waymo scenarios."""
    def __init__(self, config: DictConfig) -> None:
        """Initializes the WaymoVisualizer with the given configuration."""
        super().__init__(config)

    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: ScenarioScores | None = None,
        output_dir: str = "temp",
    ) -> None:
        """Visualizes a single scenario and saves the output to a file.

        WaymoVisualizer visualizes the scenario on two windows:
            window 1: displays the full scene zoomed out
            window 2: displays the scene with relevant agents in different colors.

        Args:
            scenario (Scenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            output_dir (str): the directory where to save the scenario visualization.
        """
        scenario_id = scenario.metadata.scenario_id
        suffix = (
            "" if scores is None or scores.safeshift_scores is None or scores.safeshift_scores.scene_score is None
            else f"_{round(scores.safeshift_scores.scene_score, 2)}"
        )
        output_filepath = f"{output_dir}/{scenario_id}{suffix}.png"
        logger.info("Visualizing scenario to %s", output_filepath)

        num_windows = 2
        axs = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))[1]

        # Plot static and dynamic map information in the scenario
        self.plot_map_data(axs, scenario, num_windows)

        # Window 1: Plot trajectory data
        self.plot_sequences(axs[0], scenario, scores)

        # Window 2: Plot trajectory data with relevant agents in a different color
        self.plot_sequences(axs[1], scenario, scores, show_relevant=True)

        # Prepare and save plot
        self.set_axes(axs, scenario, num_windows)
        plt.suptitle(f"Scenario: {scenario_id}")
        axs[0].set_title("All Agents Trajectories")
        axs[1].set_title("Highlighted Relevant and SDC Agent Trajectories")
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        plt.close()
