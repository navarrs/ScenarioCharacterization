import matplotlib.pyplot as plt
from omegaconf import DictConfig

from characterization.schemas import Scenario, ScenarioScores
from characterization.utils.io_utils import get_logger
from characterization.utils.viz.visualizer import BaseVisualizer

logger = get_logger(__name__)


class AnimatedScenarioVisualizer(BaseVisualizer):
    """Animated Visualizer for scenarios."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the AnimatedScenarioVisualizer with the given configuration."""
        super().__init__(config)

    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: ScenarioScores | None = None,
        output_dir: str = "temp",
    ) -> None:
        """Visualizes a single scenario and saves the output to a file.

        WaymoAnimatedVisualizer visualizes the scenario as an per-timestep animation.

        Args:
            scenario (Scenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            output_dir (str): the directory where to save the scenario visualization.
        """
        scenario_id = scenario.metadata.scenario_id
        suffix = (
            ""
            if scores is None or scores.safeshift_scores is None or scores.safeshift_scores.scene_score is None
            else f"_{round(scores.safeshift_scores.scene_score, 2)}"
        )
        output_filepath = f"{output_dir}/{scenario_id}{suffix}.gif"
        logger.info("Visualizing scenario to %s", output_filepath)

        total_timesteps = scenario.metadata.track_length
        for timestep in range(2, total_timesteps):
            _, ax = plt.subplots(1, 1, figsize=(5, 5))

            # Plot static and dynamic map information in the scenario
            self.plot_map_data(ax, scenario)

            self.plot_sequences(ax, scenario, scores, show_relevant=True, end_timestep=timestep)

            # Prepare and save plot
            self.set_axes(ax, scenario)
            ax.set_title(f"Scenario: {scenario_id}")
            plt.subplots_adjust(wspace=0.05)
            plt.savefig(f"{output_dir}/temp_{timestep}.png", dpi=300, bbox_inches="tight")
            plt.close()

        BaseVisualizer.to_gif(output_dir, output_filepath)
