from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

from characterization.schemas import Scenario, ScenarioScores
from characterization.utils.io_utils import get_logger
from characterization.utils.viz.visualizer import BaseVisualizer

logger = get_logger(__name__)


class AnimatedScenarioVisualizer(BaseVisualizer):
    """Animated Visualizer for scenarios."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the AnimatedScenarioVisualizer with the given configuration."""
        super().__init__(config)

    def plot_single_step(
        self, scenario: Scenario, scores: ScenarioScores | None, output_dir: str, timestep: int
    ) -> None:
        """Plots a single timestep of the scenario.

        Args:
            scenario (Scenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            output_dir (str): the directory where to save the scenario visualization.
            timestep (int): the timestep to visualize.
        """
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        scenario_id = scenario.metadata.scenario_id

        # Plot static and dynamic map information in the scenario
        self.plot_map_data(ax, scenario)

        self.plot_sequences(ax, scenario, scores, show_relevant=True, end_timestep=timestep)

        # Prepare and save plot
        self.set_axes(ax, scenario)
        if self.add_title:
            ax.set_title(f"Scenario: {scenario_id}")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"{output_dir}/temp_{timestep}.png", dpi=300, bbox_inches="tight")
        plt.close()

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
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self.plot_single_step, scenario, scores, output_dir, timestep)
                for timestep in range(2, total_timesteps)
            ]

        # tqdm progress bar
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Generating plots"):
            pass

        BaseVisualizer.to_gif(output_dir, output_filepath)
