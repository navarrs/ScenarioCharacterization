"""Trajectory prediction scenario visualizer."""

from abc import abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from numpy.typing import NDArray
from omegaconf import DictConfig

from characterization.domains.aviation.scenario_types import (
    AGENT_COLORS,
    MAP_COLORS_DARK,
    AgentType,
    MapType,
)
from characterization.utils.geometric_utils import XYZ_DIMENSIONS, transform_to_reference_frame
from characterization.utils.logging_utils import get_pylogger
from safeair.scenario_visualizer.base_visualizer import BaseVisualizer
from safeair.schemas import ModelOutput, Scenario
from safeair.schemas.scenario_scores import ScenarioScores

_LOGGER = get_pylogger(__name__)


class BaseLocalVisualizer(BaseVisualizer):
    """Visualizer for trajectory prediction scenarios.

    Displays a single ego-centric pane with map features, agent histories colored by type, ground truth future
    trajectories as dashed lines, and predicted future trajectory modes colored by probability.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseLocalVisualizer."""
        super().__init__(config)

        self.pred_colormap = cm.get_cmap(config.get("pred_colormap", "hot"))

        self.agent_zoom = {
            AgentType.AIRCRAFT: 0.015,
            AgentType.VEHICLE: 0.15,
            AgentType.UNKNOWN: 0.015,
        }

        # NOTE: self.plot_agent applies ndimage.rotate(icon, θ) then np.fliplr, so the effective icon direction is
        # (sin θ, cos θ). To align with local_heading h (where 0 = forward = plot +x), we need θ = 90 - h.
        self.heading_offset = 90.0

    def plot_map_data_in_local_frame(
        self, ax: Axes, reference_point: NDArray[np.float32], reference_heading: NDArray[np.float32], alpha: float = 0.3
    ) -> None:
        """Plots map features transformed into the ego-centric local coordinate frame.

        Silently skips if the raster map is active (no graph map data available).

        Args:
            ax: Axes to plot on.
            reference_point: Reference point for the local coordinate frame, shape (2,).
            reference_heading: Reference heading for the local coordinate frame, shape (1,).
            alpha: Alpha value for map feature points.
        """
        if self.show_raster:
            _LOGGER.warning("Raster map not supported.")
            return

        for polylines, color in [
            (self.map_data.taxiway_polylines_xy, MAP_COLORS_DARK[MapType.TAXIWAY]),
            (self.map_data.runway_polylines_xy, MAP_COLORS_DARK[MapType.RUNWAY]),
            (self.map_data.exit_polylines_xy, MAP_COLORS_DARK[MapType.EXIT]),
            (self.map_data.ramp_polylines_xy, MAP_COLORS_DARK[MapType.RAMP]),
        ]:
            local = transform_to_reference_frame(polylines[:, :2], reference_point, reference_heading)
            ax.scatter(local[:, 0], local[:, 1], c=color, s=1, alpha=alpha, zorder=1, edgecolors="none")

        hold = transform_to_reference_frame(
            self.map_data.hold_short_points_xy[:, :2], reference_point, reference_heading
        )
        color = MAP_COLORS_DARK[MapType.HOLD_SHORT_LINE]
        ax.scatter(hold[:, 0], hold[:, 1], c=color, s=2, alpha=alpha, zorder=1, marker="s", edgecolors="none")

    def plot_sequences_local(self, ax: Axes, history: NDArray[np.float32], future: NDArray[np.float32]) -> None:
        """Plots agent histories (solid) and ground truth futures (dashed) colored by agent type.

        Agent type, heading, and ego identity are read directly from the history feature vector:
        ``[x, y, z, type_onehot x5, time_embed x(H+1), sin_heading, cos_heading, speed, accel, mask]``.
        Heading is already in local frame (histories are ego-centric), so no global subtraction is needed.

        Args:
            ax: Axes to plot on.
            history: All-agent history array ``(N, H, S+1)``, last feature is validity mask.
            future: All-agent future array ``(N, F, S+1)``, last feature is validity mask.
        """
        hist_len = history.shape[1]
        n_types = len(AgentType)
        # History layout: [x, y, z | type_onehot x n_types | time_embed x (H+1) | sin_h, cos_h | speed | accel | mask]
        type_start = XYZ_DIMENSIONS  # index of first type_onehot slot
        sin_h_idx = XYZ_DIMENSIONS + n_types + hist_len + 1  # index of sin(heading)

        for hist, fut in zip(history, future, strict=False):
            hist_mask = hist[:, -1].astype(bool)
            if hist_mask.sum() < self.min_valid_timesteps:
                continue

            last_valid = hist[hist_mask][-1]
            # Base type from AIRCRAFT/VEHICLE/UNKNOWN slots (indices 0-2 in one-hot)
            atype = AgentType(int(last_valid[type_start : type_start + 3].argmax()))
            color = AGENT_COLORS.get(atype, AGENT_COLORS[AgentType.UNKNOWN])
            is_ego = bool(last_valid[type_start + AgentType.EGO_AGENT.value])

            local_heading_deg = float(np.rad2deg(np.arctan2(last_valid[sin_h_idx], last_valid[sin_h_idx + 1])))
            icon_rotation_deg = self.heading_offset - local_heading_deg

            hx, hy = hist[hist_mask, 0], hist[hist_mask, 1]
            ax.plot(hx, hy, color=color, linewidth=1, zorder=5)

            zorder = 1000 if is_ego else 100
            self.plot_agent(ax, hx[-1], hy[-1], icon_rotation_deg, atype, alpha=0.9, add_halo=is_ego, zorder=zorder)

            fut_mask = fut[:, -1].astype(bool)
            if fut_mask.sum() >= self.min_valid_timesteps:
                fx, fy = fut[fut_mask, 0], fut[fut_mask, 1]
                ax.plot(fx, fy, color=color, linewidth=1, linestyle="--", zorder=5)

    def configure_axes_local(
        self,
        ax: Axes,
        trajectories: NDArray[np.float32],
        spine_linewidth: float = 0.3,
        spine_color: str = "#cccccc",
    ) -> None:
        """Configures axes for the local frame by zooming to the ego trajectory extent.

        Sets axis limits to the ego agent's observed min/max positions plus ``buffer_distance``, hides ticks, enforces
        equal aspect ratio, and styles spines.

        Args:
            ax: Axes to configure.
            trajectories: Array of trajectory positions ``(M, 2)``.
            spine_linewidth: Line width for the axes spines. Defaults to 0.3.
            spine_color: Color for the axes spines. Defaults to ``"#cccccc"``.
        """
        min_x, min_y = np.inf, np.inf
        max_x, max_y = -np.inf, -np.inf
        for trajectory in trajectories:
            mask = trajectory[:, -1].astype(bool)
            if mask.sum() == 0:
                continue

            x, y = trajectory[mask, 0], trajectory[mask, 1]
            min_x, max_x = min(min_x, x.min()), max(max_x, x.max())
            min_y, max_y = min(min_y, y.min()), max(max_y, y.max())

        ax.set_xlim(min_x - self.buffer_distance, max_x + self.buffer_distance)
        ax.set_ylim(min_y - self.buffer_distance, max_y + self.buffer_distance)
        ax.set_autoscale_on(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)
            spine.set_color(spine_color)

    def visualize_scenario(
        self,
        scenario: Scenario,
        model_output: ModelOutput | None,
        scores: ScenarioScores | None = None,  # noqa: ARG002
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Visualizes a single scenario and saves the output to a file.

        Renders a single ego-centric pane showing map features, agent histories with type-appropriate colors and icons,
        ground truth future trajectories as dashed lines, and predicted future trajectory modes.

        Args:
            scenario: Scenario to visualize.
            model_output: Model output containing trajectory prediction results.
            scores: Unused; accepted for interface compatibility.
            output_dir: Directory where the visualization will be saved.

        Returns:
            Path to the saved visualization file.

        Raises:
            ValueError: If model_output or trajectory_prediction_output is None.
        """
        if model_output is None or model_output.trajectory_prediction_output is None:
            error_message = "Trajectory prediction output is required for ScenarioTrajpredVisualizer."
            raise ValueError(error_message)

        scenario_id = scenario.metadata.scenario_id
        output_filepath = output_dir / f"{scenario_id}.png"

        _, ax = plt.subplots(1, 1, figsize=(5, 5))

        self.visualize_scenario_local(ax, scenario, model_output)

        if self.add_title:
            ax.set_title(f"Scenario: {scenario_id}")

        plt.tight_layout()
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        plt.close()

        _LOGGER.info("Visualized scenario to %s", output_filepath)
        return output_filepath

    @abstractmethod
    def visualize_scenario_local(self, ax: Axes, scenario: Scenario, model_output: ModelOutput) -> None:
        """Visualizes a single scenario and saves the output to a file.

        This method should be implemented by subclasses to provide scenario-specific visualization,
        supporting flexible titles and output paths. It is designed to handle both static and dynamic map
        features, as well as agent trajectories and attributes.

        Args:
            ax: Axes to plot on.
            scenario: Scenario to visualize.
            model_output: Model output for the scenario.
        """
