# pyright: reportUnknownMemberType=false
"""Animated per-timestep scenario visualizer for aviation scenarios."""

import concurrent.futures
import multiprocessing as mp
import pathlib
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

from characterization.domains.aviation.schemas.scenario_scores import ScenarioScores
from characterization.domains.aviation.utils.scenario_visualizer.base_visualizer import (
    AviationBaseVisualizer,
    SupportedPanes,
)
from characterization.utils.logging_utils import get_pylogger
from safeair.schemas import Scenario

_LOGGER = get_pylogger(__name__)


class AnimatedScenarioVisualizer(AviationBaseVisualizer):
    """Animated per-timestep scenario visualizer for aviation scenarios.

    Generates one PNG frame per selected timestep and stitches them into a GIF.
    Frame generation is parallelized via ``ProcessPoolExecutor`` with the ``spawn``
    start method to avoid matplotlib threading issues.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes the AnimatedScenarioVisualizer."""
        super().__init__(config)
        self.pane_to_plot = SupportedPanes[config.get("pane_to_plot", "HIGHLIGHT_RELEVANT_AGENTS")]

    def _plot_single_step(
        self,
        scenario: Scenario,
        output_dir: Path,
        timestep_idx: int,
        timestamp: float,
    ) -> None:
        """Renders and saves a single animation frame.

        Args:
            scenario: Aviation scenario to visualize.
            output_dir: Directory where the frame PNG will be saved.
            timestep_idx: Index into the timestamps array for this frame.
            timestamp: Wall-clock timestamp in seconds for this frame.
        """
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        scenario_id = scenario.metadata.scenario_id

        self.plot_map_data(ax, alpha=0.8)

        match self.pane_to_plot:
            case SupportedPanes.ALL_AGENTS:
                self.plot_sequences(ax, scenario, end_timestep=timestep_idx)
            case SupportedPanes.HIGHLIGHT_RELEVANT_AGENTS:
                self.plot_sequences(ax, scenario, show_relevant=True, end_timestep=timestep_idx)
            case _:
                error_message = f"Unsupported pane type for animation: {self.pane_to_plot}"
                raise ValueError(error_message)

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

        self.configure_axes(ax, scenario)
        if self.add_title:
            ax.set_title(f"Scenario: {scenario_id}")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"{output_dir}/temp_{timestep_idx}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: ScenarioScores | None = None,  # noqa: ARG002
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Generates an animated GIF by rendering per-timestep frames.

        Args:
            scenario: Aviation scenario to visualize.
            scores: Unused; accepted for interface compatibility.
            output_dir: Directory where the GIF will be saved.

        Returns:
            Path to the saved GIF file.
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = pathlib.Path(tmp_dir)
            if self.num_workers > 1:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.num_workers,
                    mp_context=mp.get_context("spawn"),
                    initializer=AviationBaseVisualizer.init_worker_matplotlib,
                ) as executor:
                    futures = [
                        executor.submit(
                            self._plot_single_step,
                            scenario,
                            tmp_dir_path,
                            timestep,
                            timestamp_seconds[timestep],
                        )
                        for timestep in range(0, total_timesteps, step_size)
                    ]
                    for _ in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="Generating plots",
                    ):
                        pass
            else:
                for timestep in tqdm(range(0, total_timesteps, step_size), desc="Generating plots"):
                    self._plot_single_step(scenario, tmp_dir_path, timestep, timestamp_seconds[timestep])

            AviationBaseVisualizer.to_gif(tmp_dir_path, output_filepath, fps=self.fps)

        return output_filepath
