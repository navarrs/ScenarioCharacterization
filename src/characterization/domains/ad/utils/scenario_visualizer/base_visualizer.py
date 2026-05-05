# pyright: reportUnknownMemberType=false
"""AD-specific base visualizer."""

import json
import time
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
from pathlib import Path
from typing import cast

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from natsort import natsorted
from numpy.typing import NDArray
from omegaconf import DictConfig
from PIL import Image

from characterization.domains.ad.scenario_types import AgentTrajectoryMasker, AgentType
from characterization.domains.ad.schemas import DynamicMapData, Scenario, Score, StaticMapData
from characterization.utils.common import (
    STATIONARY_SPEED_THRESHOLD,
    SUPPORTED_SCENARIO_TYPES,
    categorize_from_thresholds,
)
from characterization.utils.constants import MIN_VALID_POINTS
from characterization.utils.logging_utils import get_pylogger

logger = get_pylogger(__name__)


class SupportedPanes(Enum):
    """Panes supported for AD scenario visualization.

    Options are:
    - ALL_AGENTS: Displays all agents in the scenario.
    - HIGHLIGHT_RELEVANT_AGENTS: Highlights relevant and ego agents.
    - SCORED_INDIVIDUAL: Visualizes per-agent individual (kinematic) scores as score-colored halos.
    - SCORED_INTERACTION: Visualizes per-agent interaction scores as score-colored halos.
    - COUNTERFACTUAL_PROBE: Overlays counterfactual probe trajectory with criticality markers.
    """

    ALL_AGENTS = 0
    HIGHLIGHT_RELEVANT_AGENTS = 1
    SCORED_INDIVIDUAL = 4
    SCORED_INTERACTION = 5
    COUNTERFACTUAL_PROBE = 6


class ADBaseVisualizer(ABC):
    """Base visualizer for AD (autonomous driving) scenarios.

    Provides AD-specific rendering: static/dynamic map feature plotting (lanes, crosswalks,
    stop signs, stop points), agent rectangle overlays with heading, score-based and categorical coloring,
    and ego-centered axis zoom.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes AD-specific visualization state on top of the generic base.

        Args:
            config: Visualizer configuration.

        Raises:
            AssertionError: If ``scenario_type`` is not in ``SUPPORTED_SCENARIO_TYPES``.
            FileNotFoundError: If ``plot_categorical=True`` and the categories file is missing.
        """
        super().__init__()

        panes_cfg = config.get("panes_to_plot", ["HIGHLIGHT_RELEVANT_AGENTS"])
        self.panes_to_plot: list[SupportedPanes] = [SupportedPanes[p] for p in panes_cfg]
        self.num_panes_to_plot: int = len(self.panes_to_plot)
        self.add_title: bool = config.get("add_title", False)
        self.title_fontsize: int = config.get("title_fontsize", 12)
        self.update_limits: bool = config.get("update_limits", True)
        self.buffer_distance: float = config.get("buffer_distance", 5.0)
        self.distance_to_ego_zoom_in: float = config.get("distance_to_ego_zoom_in", 100.0)
        self.num_workers: int = config.get("num_workers", 10)
        self.fps: int = config.get("fps", 10)
        self.time_scale_factor: float = config.get("time_scale_factor", 1.0)
        self.display_time: bool = config.get("display_time", True)
        self.show_relevant: bool = config.get("show_relevant", False)

        self.scenario_type: str = config.scenario_type
        if self.scenario_type not in SUPPORTED_SCENARIO_TYPES:
            error_message = f"Scenario type {self.scenario_type} not in supported types: {SUPPORTED_SCENARIO_TYPES}"
            raise AssertionError(error_message)

        self.map_colors: dict[str, str] = {
            "lane": "black",
            "crosswalk": "gray",
            "speed_bump": "orange",
            "road_edge": "black",
            "road_line": "black",
            "stop_sign": "red",
            "stop_point": "purple",
        }

        self.map_alphas: dict[str, float] = {
            "lane": 0.1,
            "crosswalk": 0.6,
            "speed_bump": 0.6,
            "road_edge": 0.1,
            "road_line": 0.1,
            "stop_sign": 0.8,
            "stop_point": 0.8,
        }

        self.agent_colors: dict[AgentType, str] = {
            AgentType.TYPE_UNSET: "gray",
            AgentType.TYPE_VEHICLE: "slategray",
            AgentType.TYPE_PEDESTRIAN: "plum",
            AgentType.TYPE_CYCLIST: "forestgreen",
            AgentType.TYPE_OTHER: "gray",
            AgentType.TYPE_EGO_AGENT: "dodgerblue",
            AgentType.TYPE_RELEVANT: "coral",
        }

        self.plot_categorical: bool = config.get("plot_categorical", False)
        if self.plot_categorical:
            categories_filepath = Path(config.categories_file)
            if not categories_filepath.is_file():
                error_message = f"Categories file not found at {categories_filepath}"
                raise FileNotFoundError(error_message)
            with categories_filepath.open("r") as f:
                self.categories_values: list[float] = list(json.load(f).values())
            self.num_categories: int = len(self.categories_values) + 1
            from matplotlib import cm as _cm  # noqa: PLC0415

            color_map = _cm.get_cmap(config.get("categorical_color_map", "Spectral_r"))
            vals = np.linspace(0, 1, self.num_categories)
            colors = [color_map(v) for v in vals]
            hex_colors = [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b, _ in colors]
            self.categorical_color_map: dict[int, str] = {i: hex_colors[i] for i in range(self.num_categories + 1)}
            self.categorical_color_map[-1] = "lightgray"

        self._warned_missing_map_data: bool = False

    def _warn_missing_map_data(self, scenario: Scenario) -> None:
        """Emits once-per-instance warnings when static or dynamic map data is absent."""
        if self._warned_missing_map_data:
            return
        if scenario.static_map_data is None:
            warnings.warn(
                "Scenario does not contain static_map_data, skipping static map visualization.",
                UserWarning,
                stacklevel=3,
            )
        if scenario.dynamic_map_data is None:
            warnings.warn(
                "Scenario does not contain dynamic_map_data, skipping dynamic map visualization.",
                UserWarning,
                stacklevel=3,
            )
        self._warned_missing_map_data = True

    def plot_map_data(self, ax: Axes, scenario: Scenario, num_windows: int = 1) -> None:
        """Plots static and dynamic map features (lanes, crosswalks, stop points, etc.).

        Args:
            ax: Axes to plot on.
            scenario: Scenario containing map data.
            num_windows: Number of subplot windows.
        """
        self._warn_missing_map_data(scenario)
        if scenario.static_map_data is not None:
            self.plot_static_map_data(ax, static_map_data=scenario.static_map_data, num_windows=num_windows)
        if scenario.dynamic_map_data is not None:
            self.plot_dynamic_map_data(ax, dynamic_map_data=scenario.dynamic_map_data, num_windows=num_windows)

    def plot_sequences_categorical(
        self,
        ax: Axes,
        scenario: Scenario,
        scores: Score | None = None,
        *,
        start_timestep: int = 0,
        end_timestep: int = -1,
        title: str = "",
    ) -> None:
        """Plots agent trajectories colored by discrete risk categories.

        Args:
            ax: Axes to plot on.
            scenario: AD scenario.
            scores: Must contain ``agent_scores`` for categorical mapping.
            start_timestep: First timestep to include.
            end_timestep: Last timestep (exclusive; -1 = all).
            title: Axes title.

        Raises:
            ValueError: If ``scores`` or ``scores.agent_scores`` is None.
        """
        agent_data = scenario.agent_data
        agent_types = np.asarray(agent_data.agent_types)

        if scores is None or scores.agent_scores is None:
            error_message = "Scores with agent_scores are required for categorical visualization."
            raise ValueError(error_message)

        agent_scores = scores.agent_scores
        agent_scores = ADBaseVisualizer.convert_to_risk_levels(agent_scores, self.categories_values)

        agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
        zipped = zip(
            agent_trajectories.agent_xy_pos,
            agent_trajectories.agent_lengths,
            agent_trajectories.agent_widths,
            agent_trajectories.agent_headings,
            agent_trajectories.agent_valid.squeeze(-1).astype(bool),
            agent_scores,
            agent_types,
            strict=False,
        )

        timestamps = np.asarray(scenario.metadata.timestamps_seconds)
        norm_timestamps = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
        for apos, alen, awid, ahead, amask, score, atype in zipped:
            mask = amask[start_timestep:end_timestep]
            alpha_t = norm_timestamps[start_timestep:end_timestep][mask]
            if not mask.any() or mask.sum() < MIN_VALID_POINTS:
                continue

            pos = apos[start_timestep:end_timestep][mask]
            heading = ahead[end_timestep]
            length = alen[end_timestep]
            width = awid[end_timestep]
            color = self.agent_colors[atype] if atype == AgentType.TYPE_EGO_AGENT else self.categorical_color_map[score]
            ax.scatter(pos[:, 0], pos[:, 1], color=color, s=0.5, alpha=alpha_t)
            self.plot_agent(ax, pos[-1, 0], pos[-1, 1], heading, length, width, 1.0, color, plot_rectangle=True)

        if self.add_title:
            ax.set_title(title, fontsize=self.title_fontsize)

    def plot_sequences(
        self,
        ax: Axes,
        scenario: Scenario,
        scores: Score | None = None,
        *,
        show_relevant: bool = False,
        start_timestep: int = 0,
        end_timestep: int = -1,
        title: str = "",
    ) -> None:
        """Plots agent trajectories with optional relevance highlighting and score transparency.

        Args:
            ax: Axes to plot on.
            scenario: AD scenario.
            scores: Optional scores for transparency weighting.
            show_relevant: If True, marks relevant agents with ``TYPE_RELEVANT`` color.
            start_timestep: First timestep to include.
            end_timestep: Last timestep (exclusive; -1 = all).
            title: Axes title.
        """
        agent_data = scenario.agent_data
        agent_relevance = agent_data.agent_relevance
        agent_types = np.asarray(agent_data.agent_types)
        ego_index = scenario.metadata.ego_agent_index

        agent_scores = np.ones(agent_data.num_agents, float)
        if scores is not None and scores.agent_scores is not None:
            agent_scores = scores.agent_scores
            agent_scores[ego_index] = 0.0
            agent_scores = ADBaseVisualizer.get_normalized_agent_scores(agent_scores, ego_index)

        if show_relevant and agent_relevance is not None:
            relevant_indices = np.where(agent_relevance > 0.0)[0]
            agent_types[relevant_indices] = AgentType.TYPE_RELEVANT
        agent_types[ego_index] = AgentType.TYPE_EGO_AGENT

        agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
        zipped = zip(
            agent_trajectories.agent_xy_pos,
            agent_trajectories.agent_xy_vel,
            agent_trajectories.agent_lengths,
            agent_trajectories.agent_widths,
            agent_trajectories.agent_headings,
            agent_trajectories.agent_valid.squeeze(-1).astype(bool),
            agent_types,
            agent_scores,
            strict=False,
        )

        timestamps = np.asarray(scenario.metadata.timestamps_seconds)
        norm_timestamps = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])

        for apos, avel, alen, awid, ahead, amask, atype, score in zipped:
            mask = amask[start_timestep:end_timestep]
            alpha_t = norm_timestamps[start_timestep:end_timestep][mask]
            if not mask.any() or mask.sum() < MIN_VALID_POINTS:
                continue

            pos = apos[start_timestep:end_timestep][mask]
            vel = avel[start_timestep:end_timestep][mask]
            heading = ahead[end_timestep]
            length = alen[end_timestep]
            width = awid[end_timestep]
            color = self.agent_colors.get(atype, self.agent_colors[AgentType.TYPE_UNSET])

            mean_speed = np.linalg.norm(vel, axis=1).mean()
            if mean_speed > STATIONARY_SPEED_THRESHOLD:
                ax.scatter(pos[:, 0], pos[:, 1], color=color, s=2, alpha=alpha_t * score)

            alpha = alpha_t[-1] * score if alpha_t.size > 0 else score
            self.plot_agent(ax, pos[-1, 0], pos[-1, 1], heading, length, width, alpha, color, plot_rectangle=True)

        if self.add_title:
            ax.set_title(title, fontsize=self.title_fontsize)

    def plot_agent(
        self,
        ax: Axes,
        x: float,
        y: float,
        heading: float,
        width: float,
        height: float,
        alpha: float,
        color: str = "magenta",
        *,
        plot_rectangle: bool = False,
        linewidth: float = 0.5,
        edgecolor: str = "black",
        zorder: int = 100,
        marker: str = "o",
        marker_size: int = 8,
    ) -> None:
        """Plots a single agent as a rotated rectangle or a scatter marker.

        Args:
            ax: Axes to plot on.
            x: Agent center X position.
            y: Agent center Y position.
            heading: Heading angle in radians.
            width: Agent width (used when ``plot_rectangle=True``).
            height: Agent height/length (used when ``plot_rectangle=True``).
            alpha: Transparency.
            color: Fill color.
            plot_rectangle: If True, renders a rotated rectangle; otherwise a scatter marker.
            linewidth: Edge linewidth for rectangle mode.
            edgecolor: Edge color for rectangle mode.
            zorder: Z-order for layering.
            marker: Marker type for scatter mode.
            marker_size: Marker size for scatter mode.
        """
        if plot_rectangle:
            angle_deg = np.rad2deg(heading)
            cx, cy = -width / 2.0, -height / 2.0
            x_offset = cx * np.cos(heading) - cy * np.sin(heading)
            y_offset = cx * np.sin(heading) + cy * np.cos(heading)
            rect = Rectangle(
                (x + x_offset, y + y_offset),
                width,
                height,
                angle=angle_deg,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor=color,
                alpha=alpha,
                zorder=zorder,
            )
            ax.add_patch(rect)
        else:
            ax.scatter(x, y, s=marker_size, zorder=zorder, c=color, marker=marker, alpha=alpha)

    def plot_static_map_data(
        self,
        ax: Axes,
        static_map_data: StaticMapData,
        num_windows: int = 1,
        dim: int = 2,
    ) -> None:
        """Plots static map features: lanes, road lines, road edges, crosswalks, speed bumps, stop signs.

        Args:
            ax: Axes to plot on.
            static_map_data: Static map features.
            num_windows: Number of subplot windows.
            dim: Number of spatial dimensions to use (2 = XY).
        """
        if static_map_data.map_polylines is None:
            logger.warning("Scenario does not contain map_polylines, skipping static map visualization.")
            return

        road_graph = static_map_data.map_polylines[:, :dim]
        for attr, key in [
            ("lane_polyline_idxs", "lane"),
            ("road_line_polyline_idxs", "road_line"),
            ("road_edge_polyline_idxs", "road_edge"),
            ("crosswalk_polyline_idxs", "crosswalk"),
            ("speed_bump_polyline_idxs", "speed_bump"),
            ("stop_sign_polyline_idxs", "stop_sign"),
        ]:
            idxs = getattr(static_map_data, attr)
            if idxs is not None:
                ADBaseVisualizer.plot_polylines(
                    ax,
                    road_graph,
                    idxs,
                    num_windows,
                    color=self.map_colors[key],
                    alpha=self.map_alphas[key],
                )

    def plot_dynamic_map_data(self, ax: Axes, dynamic_map_data: DynamicMapData, num_windows: int = 0) -> None:
        """Plots dynamic map features (stop points).

        Args:
            ax: Axes to plot on.
            dynamic_map_data: Dynamic map features.
            num_windows: Number of subplot windows.
        """
        stop_points = dynamic_map_data.stop_points
        if stop_points is None:
            return
        x_pos = stop_points[0][0][:, 0]
        y_pos = stop_points[0][0][:, 1]
        color = self.map_colors["stop_point"]
        alpha = self.map_alphas["stop_point"]
        if num_windows == 1:
            ax.scatter(x_pos, y_pos, s=6, c=color, marker="s", alpha=alpha)
        else:
            for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                a.scatter(x_pos, y_pos, s=6, c=color, marker="s", alpha=alpha)

    @staticmethod
    def plot_stop_signs(
        ax: Axes,
        road_graph: NDArray[np.float32],
        polyline_idxs: NDArray[np.int32],
        num_windows: int = 0,
        color: str = "red",
        dim: int = 2,
    ) -> None:
        """Plots stop signs as hexagon markers on the road graph.

        Args:
            ax: Axes to plot on.
            road_graph: Road graph point array.
            polyline_idxs: (N, 2) start/end index pairs for stop sign polylines.
            num_windows: Number of subplot windows.
            color: Marker color.
            dim: Spatial dimensions to extract from ``road_graph``.
        """
        for polyline in polyline_idxs:
            start_idx, end_idx = cast("NDArray[np.int32]", polyline)
            pos = road_graph[start_idx:end_idx, :dim]
            if num_windows == 1:
                ax.scatter(pos[:, 0], pos[:, 1], s=16, c=color, marker="H", alpha=1.0)
            else:
                for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                    a.scatter(pos[:, 0], pos[:, 1], s=16, c=color, marker="H", alpha=1.0)

    @staticmethod
    def plot_polylines(
        ax: Axes,
        road_graph: NDArray[np.float32],
        polyline_idxs: NDArray[np.int32],
        num_windows: int = 0,
        color: str = "k",
        alpha: float = 1.0,
        linewidth: float = 0.5,
    ) -> None:
        """Plots polylines (lanes, crosswalks, etc.) from indexed road graph data.

        Args:
            ax: Axes to plot on.
            road_graph: Road graph point array.
            polyline_idxs: (N, 2) start/end index pairs for each polyline.
            num_windows: Number of subplot windows.
            color: Line color.
            alpha: Line transparency.
            linewidth: Line width.
        """
        for polyline in polyline_idxs:
            start_idx, end_idx = cast("NDArray[np.int32]", polyline)
            pos = road_graph[start_idx:end_idx]
            if num_windows == 1:
                ax.plot(pos[:, 0], pos[:, 1], color, alpha=alpha, linewidth=linewidth)
            else:
                for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                    a.plot(pos[:, 0], pos[:, 1], color, alpha=alpha, linewidth=linewidth)

    @staticmethod
    def get_first_and_last_ego_position(
        scenario: Scenario,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[None, None]:
        """Returns the first and last valid XY positions of the ego vehicle.

        Args:
            scenario: AD scenario.

        Returns:
            ``(first_position, last_position)`` arrays, or ``(None, None)`` if the ego
            trajectory has fewer than ``MIN_VALID_POINTS`` valid points.
        """
        ego_index = scenario.metadata.ego_agent_index
        agent_trajectories = AgentTrajectoryMasker(scenario.agent_data.agent_trajectories)
        agent_valid = agent_trajectories.agent_valid.squeeze(-1).astype(bool)
        agent_positions = agent_trajectories.agent_xy_pos

        ego_traj = agent_positions[ego_index]
        valid_mask = agent_valid[ego_index] & np.all(np.isfinite(ego_traj), axis=-1)
        valid_ego_traj = ego_traj[valid_mask]
        if valid_ego_traj.shape[0] < MIN_VALID_POINTS:
            return None, None

        return valid_ego_traj[0], valid_ego_traj[-1]

    @staticmethod
    def convert_to_risk_levels(agent_scores: NDArray[np.float32], categories: list[float]) -> NDArray[np.int32]:
        """Converts continuous scores to discrete risk level indices.

        Args:
            agent_scores: Continuous score array.
            categories: Threshold list defining risk level boundaries.

        Returns:
            Integer risk level array of the same shape as ``agent_scores``.
        """
        return np.array(
            [categorize_from_thresholds(float(score), categories) for score in agent_scores],
            dtype=np.int32,
        )

    def set_axes(self, ax: Axes, scenario: Scenario, num_windows: int = 1) -> None:
        """Configures axis limits centered around the ego vehicle's last position.

        Args:
            ax: Axes (or array of Axes) to configure.
            scenario: Scenario used to compute ego displacement.
            num_windows: Number of subplot windows.
        """
        first_ego_position, last_ego_position = ADBaseVisualizer.get_first_and_last_ego_position(scenario)
        if first_ego_position is None or last_ego_position is None:
            return

        ego_displacement = np.linalg.norm(first_ego_position - last_ego_position, axis=-1)
        distance = max(self.distance_to_ego_zoom_in, ego_displacement) + self.buffer_distance

        if num_windows == 1:
            ax = np.asarray([ax])  # pyright: ignore[reportAssignmentType]

        for n, a in enumerate(ax.reshape(-1)):  # pyright: ignore[reportAttributeAccessIssue]
            a.set_xticks([])
            a.set_yticks([])
            if n == 0:
                continue
            if self.update_limits:
                a.set_xlim(last_ego_position[0] - distance, last_ego_position[0] + distance)
                a.set_ylim(last_ego_position[1] - distance, last_ego_position[1] + distance)

    @staticmethod
    def to_gif(
        tmp_dir_frames: Path,
        output_filepath: Path,
        *,
        fps: int = 10,
        disposal: int = 2,
        loop: int = 0,
    ) -> None:
        """Converts a directory of per-frame PNG files into an animated GIF."""
        t_i = time.time()
        files = natsorted(glob(f"{tmp_dir_frames}/temp_*.png"))
        if not files:
            err_msg = f"No frames found in {tmp_dir_frames}, cannot create GIF."
            raise RuntimeError(err_msg)
        images_to_append = (Image.open(f) for f in files[1:])
        duration = 1000 / fps
        Image.open(files[0]).save(
            output_filepath,
            format="GIF",
            append_images=images_to_append,
            save_all=True,
            duration=duration,
            disposal=disposal,
            loop=loop,
        )
        t_f = time.time()
        logger.info("Saved GIF to %s [Time taken: %.2fs]", output_filepath, t_f - t_i)

    @staticmethod
    def get_normalized_agent_scores(
        agent_scores: NDArray[np.float32],
        ego_index: int,
        amin: float = 0.05,
        amax: float = 1.0,
        global_min_score: float | None = None,
        global_max_score: float | None = None,
    ) -> NDArray[np.float32]:
        """Normalizes agent scores to [amin, amax] and pins the ego agent to amax."""
        min_score = np.nanmin(agent_scores) if global_min_score is None else global_min_score
        max_score = np.nanmax(agent_scores) if global_max_score is None else global_max_score
        if max_score > min_score:
            agent_scores = (agent_scores - min_score) / (max_score - min_score)
        else:
            agent_scores = np.ones_like(agent_scores, dtype=np.float32) / len(agent_scores)  # pyright: ignore[reportAssignmentType]
        agent_scores = np.clip(agent_scores, a_min=amin, a_max=amax).astype(np.float32)
        agent_scores[ego_index] = amax
        return agent_scores

    @staticmethod
    def init_worker_matplotlib() -> None:
        """Sets the non-interactive Agg backend in spawned worker processes."""
        mpl.use("Agg")

    @abstractmethod
    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: Score | None = None,
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Visualizes a scenario and saves the output to a file."""
