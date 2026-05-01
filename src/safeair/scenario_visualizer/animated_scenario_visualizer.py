# pyright: reportUnknownMemberType=false

import pathlib
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

from characterization.utils.logging_utils import get_pylogger
from safeair.scenario_visualizer.base_visualizer import BaseVisualizer, SupportedPanes
from safeair.schemas import ModelOutput, Scenario
from safeair.schemas.scenario_scores import ScenarioScores

_LOGGER = get_pylogger(__name__)


class AnimatedScenarioVisualizer(BaseVisualizer):
    """Animated Visualizer for scenarios."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the AnimatedScenarioVisualizer with the given configuration."""
        super().__init__(config)

        # Set up pane to plot. It will fail if an invalid pane is provided.
        self.pane_to_plot = SupportedPanes[config.get("pane_to_plot", "HIGHLIGHT_RELEVANT_AGENTS")]

    def _plot_single_step(
        self,
        scenario: Scenario,
        model_output: ModelOutput | None,
        output_dir: Path,
        timestep_idx: int,
        timestamp: float,
    ) -> None:
        """Plots a single timestep of the scenario.

        Args:
            scenario: Scenario to visualize.
            model_output: Model output for the scenario.
            output_dir: Directory where the frame will be saved.
            timestep_idx: Timestep index (in the timestamps array) to visualize.
            timestamp: Timestamp corresponding to the timestep.
        """
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        scenario_id = scenario.metadata.scenario_id

        # Plot static and dynamic map information in the scenario
        self.plot_map_data(ax, alpha=0.8)

        match self.pane_to_plot:
            case SupportedPanes.ALL_AGENTS:
                self.plot_sequences(ax, scenario, end_timestep=timestep_idx)
            case SupportedPanes.HIGHLIGHT_RELEVANT_AGENTS:
                self.plot_sequences(ax, scenario, show_relevant=True, end_timestep=timestep_idx)
            case SupportedPanes.CRITICAL_SCENARIO:
                if scenario.critical_metadata is None:
                    error_message = (
                        f"{self.pane_to_plot} pane requires critical_metadata in the scenario, but it is None."
                    )
                    raise ValueError(error_message)
                self.plot_sequences_with_critical_metadata(ax, scenario, end_timestep=timestep_idx)
            case SupportedPanes.CRITICAL_SCENARIO_IDENTIFICATION_PREDICTION:
                if scenario.critical_metadata is None or model_output is None:
                    error_message = (
                        f"{self.pane_to_plot} pane requires critical_metadata in the scenario and model_output, but "
                        "one of them is None."
                    )
                    raise ValueError(error_message)
                self.plot_sequences_with_critical_scenario_prediction(
                    ax, scenario, model_output=model_output, end_timestep=timestep_idx
                )
            case _:
                error_message = f"Unsupported pane type: {self.pane_to_plot}"
                raise ValueError(error_message)

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
        self.configure_axes(ax, scenario)
        if self.add_title:
            ax.set_title(f"Scenario: {scenario_id}")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"{output_dir}/temp_{timestep_idx}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_scenario(
        self,
        scenario: Scenario,
        model_output: ModelOutput | None,
        scores: ScenarioScores | None = None,  # noqa: ARG002
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Visualizes a single scenario and saves the output to a file.

        AnimatedScenarioVisualizer visualizes the scenario as a per-timestep animation.

        Args:
            scenario: Scenario to visualize.
            model_output: Model output for the scenario.
            scores: Unused; accepted for interface compatibility.
            output_dir: Directory where the visualization will be saved.

        Returns:
            Path to the saved visualization file.
        """
        scenario_id = scenario.metadata.scenario_id
        output_filepath = output_dir / f"{scenario_id}.gif"

        timestamp_seconds = scenario.metadata.timestamps_seconds
        scenario_fps = min(self.fps, scenario.metadata.frequency_hz)
        total_timesteps = scenario.metadata.track_length
        total_time = timestamp_seconds[-1] - timestamp_seconds[0]
        num_frames = scenario_fps * total_time * self.time_scale_factor
        step_size = max(1, total_timesteps // int(num_frames))

        _LOGGER.info(
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
            # matplotlib is not thread-safe
            # ThreadPoolExecutor would result in corrupt images
            if self.num_workers > 1:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [
                        executor.submit(
                            self._plot_single_step,
                            scenario,
                            model_output,
                            tmp_dir_path,
                            timestep,
                            timestamp_seconds[timestep],
                        )
                        # For simplicity, the frames between the last timestamp and the end of the scenario are ignored
                        # (range does not include total_timesteps-1 when step_size > 1). This is fine for typical
                        # scenarios around 30 s at 100 Hz where we lose ~100 frames or up to 1 s, representing ~3% of
                        # the scenario duration.
                        for timestep in range(0, total_timesteps, step_size)
                    ]

                # tqdm progress bar
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Generating plots"):
                    pass
            else:
                for timestep in tqdm(range(0, total_timesteps, step_size), desc="Generating plots"):
                    self._plot_single_step(
                        scenario,
                        model_output,
                        tmp_dir_path,
                        timestep,
                        timestamp_seconds[timestep],
                    )
            BaseVisualizer.to_gif(tmp_dir_path, output_filepath, fps=self.fps)
        return output_filepath
