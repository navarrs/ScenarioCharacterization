import pathlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

    def _plot_single_step(
        self,
        scenario: Scenario,
        scores: ScenarioScores | None,
        output_dir: Path,
        timestep_idx: int,
        timestamp: float,
    ) -> None:
        """Plots a single timestep of the scenario.

        Args:
            scenario (Scenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            output_dir (str): the directory where to save the scenario visualization.
            timestep_idx (int): the timestep index (in the timestamps array) to visualize.
            timestamp (float): the timestamp corresponding to the timestep.
        """
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        scenario_id = scenario.metadata.scenario_id

        # Plot static and dynamic map information in the scenario
        self.plot_map_data(ax, scenario)

        self.plot_sequences(ax, scenario, scores, show_relevant=True, end_timestep=timestep_idx)

        # Add timestamp annotation in the upper right corner
        if self.display_time:
            timestamp_str = f"t-elapsed: {timestamp:.2f}s (t-scale={self.time_scale_factor})"
            ax.annotate(
                timestamp_str,
                xy=(0.98, 0.98),
                xycoords="axes fraction",
                fontsize=8,
                ha="right",
                va="top",
                bbox={"boxstyle": "round,pad=0.3", "fc": "gray", "ec": "gray", "alpha": 0.3},
            )

        # Prepare and save plot
        self.set_axes(ax, scenario)
        if self.add_title:
            ax.set_title(f"Scenario: {scenario_id}")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"{output_dir}/temp_{timestep_idx}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: ScenarioScores | None = None,
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Visualizes a single scenario and saves the output to a file.

        WaymoAnimatedVisualizer visualizes the scenario as an per-timestep animation.

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
            if scores is None or scores.safeshift_scores is None or scores.safeshift_scores.scene_score is None
            else f"_{round(scores.safeshift_scores.scene_score, 2)}"
        )
        output_filepath = output_dir / f"{scenario_id}{suffix}.gif"

        timestamp_seconds = scenario.metadata.timestamps_seconds
        scenario_fps = min(self.fps, scenario.metadata.frequency_hz)
        total_timesteps = scenario.metadata.track_length
        total_time = timestamp_seconds[-1] - timestamp_seconds[0]
        num_frames = scenario_fps * total_time * self.time_scale_factor
        step_size = max(1, total_timesteps // int(num_frames))

        logger.info(
            "Saving scenario to %s [Num. timesteps: %d, Total time: %.2fs, Generating ~%d frames with step size %d]",
            output_filepath,
            total_timesteps,
            total_time,
            int(num_frames),
            step_size,
        )

        # Use a temporary dir to save individual frames rather than the output_dir to avoid contamination with multiple
        # runs as temp* files are globbed
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = pathlib.Path(tmp_dir)
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(
                        self._plot_single_step,
                        scenario,
                        scores,
                        tmp_dir_path,
                        timestep,
                        timestamp_seconds[timestep],
                    )
                    # For simplicity, the frames between the last timestamp
                    # and the end of the scenario are ignored (range does not
                    # include total_timesteps-1 when step_size > 1).
                    # This is fine for typical scenarios around 30 s at 100 Hz
                    # where we lose ~100 frames or up to 1 s, representing
                    # ~3% of the scenario duration.
                    for timestep in range(0, total_timesteps, step_size)
                ]

            # tqdm progress bar
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Generating plots"):
                pass
            BaseVisualizer.to_gif(tmp_dir_path, output_filepath, fps=self.fps)
        return output_filepath
