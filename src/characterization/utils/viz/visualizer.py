"""Base class for scenario visualizers."""

from abc import ABC, abstractmethod
from glob import glob
from enum import Enum
from pathlib import Path
import time
import numpy as np
from characterization.schemas import DynamicMapData, Scenario, ScenarioScores, StaticMapData
from characterization.utils.common import SUPPORTED_SCENARIO_TYPES, AgentTrajectoryMasker, MIN_VALID_POINTS
from characterization.utils.io_utils import get_logger
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from natsort import natsorted
from omegaconf import DictConfig
from PIL import Image

from warnings import warn

logger = get_logger(__name__)

class SupportedPanes(Enum):
    ALL_AGENTS = 0
    HIGHLIGHT_RELEVANT_AGENTS = 1

class BaseVisualizer(ABC):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseVisualizer with visualization configuration and validates required keys.

        This base class provides a flexible interface for scenario visualizers, supporting custom map and agent color
        schemes, transparency, and scenario type validation. Subclasses should implement scenario-specific visualization
        logic.

        Args:
            config (DictConfig): Configuration for the visualizer, including scenario type, map/agent keys, colors, and
                alpha values.

        Raises:
            AssertionError: If the scenario type or any required configuration key is missing or unsupported.
        """
        self.config = config
        self.scenario_type = config.scenario_type
        if self.scenario_type not in SUPPORTED_SCENARIO_TYPES:
            error_message = f"Scenario type {self.scenario_type} not in supported types: {SUPPORTED_SCENARIO_TYPES}"
            raise AssertionError(error_message)

        self.static_map_keys = config.get("static_map_keys", None)
        if self.static_map_keys is None:
            error_message = "static_map_keys must be provided in the configuration."
            raise AssertionError(error_message)

        self.dynamic_map_keys = config.get("dynamic_map_keys", None)
        if self.dynamic_map_keys is None:
            error_message = "dynamic_map_keys must be provided in the configuration."
            raise AssertionError(error_message)

        self.map_colors = config.get("map_colors", None)
        if self.map_colors is None:
            error_message = "map_colors must be provided in the configuration."
            raise AssertionError(error_message)

        self.map_alphas = config.get("map_alphas", None)
        if self.map_alphas is None:
            error_message = "map_alphas must be provided in the configuration."
            raise AssertionError(error_message)

        self.agent_colors = config.get("agent_colors", None)
        if self.agent_colors is None:
            error_message = "agent_colors must be provided in the configuration."
            raise AssertionError(error_message)

        # Set up panes to plot. It will fail if an invalid pane is provided.
        panes = config.get("panes_to_plot", ["ALL_AGENTS"])
        self.panes_to_plot = [SupportedPanes[pane] for pane in panes]
        self.num_panes_to_plot = len(self.panes_to_plot)

        # Number of workers for processing animations in parallel
        self.num_workers = config.get("num_workers", 10)
        self.fps = config.get("fps", 10)
        self.time_scale_factor = config.get("time_scale_factor", 1.0)
        self.display_time = config.get("display_time", True)

        # Other visualization options
        self.add_title = config.get("add_title", True)
        self.update_limits = config.get("update_limits", False)
        self.buffer_distance = config.get("distance_to_ego_zoom_in", 5.0)  # in meters
        self.distance_to_ego_zoom_in = config.get("distance_to_ego_zoom_in", 100.0)  # in meters

    def plot_map_data(self, ax: Axes, scenario: Scenario, num_windows: int = 1) -> None:
        """Plots the map data.

        Args:
            ax (Axes): Axes to plot on.
            scenario (Scenario): encapsulates the scenario to visualize.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
        """
        # Plot static map information
        if scenario.static_map_data is None:
            warn("Scenario does not contain map_polylines, skipping static map visualization.", UserWarning)
        else:
            self.plot_static_map_data(ax, static_map_data=scenario.static_map_data, num_windows=num_windows)

        # Plot dynamic map information
        if scenario.dynamic_map_data is None:
            warn("Scenario does not contain dynamic_map_info, skipping dynamic map visualization.", UserWarning)
        else:
            self.plot_dynamic_map_data(ax, dynamic_map_data=scenario.dynamic_map_data, num_windows=num_windows)

    def plot_sequences(  # noqa: PLR0913
        self,
        ax: Axes,
        scenario: Scenario,
        scores: ScenarioScores | None = None,
        *,
        show_relevant: bool = False,
        start_timestep: int = 0,
        end_timestep: int = -1,
        title: str = "",
    ) -> None:
        """Plots agent trajectories for a scenario, with optional highlighting and score-based transparency.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            scenario (Scenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            show_relevant (bool, optional): If True, highlights relevant and SDC agents. Defaults to False.
            start_timestep (int): starting timestep to plot the sequences.
            end_timestep (int): ending timestep to plot the sequences.
            title (str, optional): Title for the plot. Defaults to "".
        """
        agent_data = scenario.agent_data
        agent_relevance = agent_data.agent_relevance
        agent_types = np.asarray([atype.name for atype in agent_data.agent_types])
        ego_index = scenario.metadata.ego_vehicle_index

        # Get the agent normalized scores
        agent_scores = (
            np.ones(agent_data.num_agents, float)
            if scores is None or scores.safeshift_scores is None or scores.safeshift_scores.agent_scores is None
            else scores.safeshift_scores.agent_scores
        )
        agent_scores = BaseVisualizer.get_normalized_agent_scores(agent_scores, ego_index)

        # Mark any agents with a relevance score > 0 as "TYPE_RELEVANT"
        if show_relevant and agent_relevance is not None:
            relevant_indices = np.where(agent_relevance > 0.0)[0]
            agent_types[relevant_indices] = "TYPE_RELEVANT"
        agent_types[ego_index] = "TYPE_SDC"

        # Zip information to plot
        agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
        zipped = zip(
            agent_trajectories.agent_xy_pos,
            agent_trajectories.agent_lengths,
            agent_trajectories.agent_widths,
            agent_trajectories.agent_headings,
            agent_trajectories.agent_valid.squeeze(-1).astype(bool),
            agent_types,
            agent_scores,
            strict=False,
        )

        for apos, alen, awid, ahead, amask, atype, score in zipped:
            # Skip if there are less than 2 valid points
            mask = amask[start_timestep:end_timestep]
            if not mask.any() or mask.sum() < MIN_VALID_POINTS:
                continue

            pos = apos[start_timestep:end_timestep][mask]
            heading = ahead[end_timestep]
            length = alen[end_timestep]
            width = awid[end_timestep]
            atype = "TYPE_RELEVANT" if atype == "TYPE_RELEVAN" else atype
            color = self.agent_colors[atype]

            # Plot the trajectory
            ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, alpha=score)

            # Plot the agent
            self.plot_agent(ax, pos[-1, 0], pos[-1, 1], heading, length, width, score, color, plot_rectangle=True)

        if self.add_title:
            ax.set_title(title)

    def plot_agent(  # noqa: PLR0913
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
        """Plots a single agent as a point (optionally as a rectangle) on the axes.

        Args:
            ax (matplotlib.axes.Axes): axes to plot on.
            x (float): x position of the agent.
            y (float): y position of the agent.
            heading (float): heading angle of the agent.
            width (float): width of the agent.
            height (float): height of the agent.
            alpha (float): transparency for the agent marker.
            color (str): color of the agent marker.
            plot_rectangle (bool): if true it will plot the agent as rectangle, otherwise it will plot it as 'marker'.
            edgecolor (str): color of the agent's edge if 'plot_rectangle' is True.
            linewidth (float): width of the agent's edge if 'plot_rectangle' is True.
            zorder (int): z order of agent to plot.
            marker (str): marker type of the agent if 'plot_rectangle' is False.
            marker_size (int): size of the marker if to plot the agent.
        """
        if plot_rectangle:
            # Compute the agent’s orientation
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
        self, ax: Axes, static_map_data: StaticMapData, num_windows: int = 1, dim: int = 2
    ) -> None:
        """Plots static map features (lanes, road lines, crosswalks, etc.) for a scenario.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            static_map_data (StaticMapData): static map information.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
            dim (int, optional): Number of dimensions to plot. Defaults to 2.
        """
        if static_map_data.map_polylines is None:
            logger.warning("Scenario does not contain map_polylines, skipping static map visualization.")
            return

        road_graph = static_map_data.map_polylines[:, :dim]
        if static_map_data.lane_polyline_idxs is not None:
            color, alpha = self.map_colors["lane"], self.map_alphas["lane"]
            BaseVisualizer.plot_polylines(
                ax, road_graph, static_map_data.lane_polyline_idxs, num_windows, color=color, alpha=alpha
            )

        if static_map_data.road_line_polyline_idxs is not None:
            color, alpha = self.map_colors["road_line"], self.map_alphas["road_line"]
            BaseVisualizer.plot_polylines(
                ax, road_graph, static_map_data.road_line_polyline_idxs, num_windows, color=color, alpha=alpha
            )

        if static_map_data.road_edge_polyline_idxs is not None:
            color, alpha = self.map_colors["road_edge"], self.map_alphas["road_edge"]
            BaseVisualizer.plot_polylines(
                ax, road_graph, static_map_data.road_edge_polyline_idxs, num_windows, color=color, alpha=alpha
            )

        if static_map_data.crosswalk_polyline_idxs is not None:
            color, alpha = self.map_colors["crosswalk"], self.map_alphas["crosswalk"]
            BaseVisualizer.plot_polylines(
                ax, road_graph, static_map_data.crosswalk_polyline_idxs, num_windows, color, alpha
            )

        if static_map_data.speed_bump_polyline_idxs is not None:
            color, alpha = self.map_colors["speed_bump"], self.map_alphas["speed_bump"]
            BaseVisualizer.plot_polylines(
                ax, road_graph, static_map_data.speed_bump_polyline_idxs, num_windows, color, alpha
            )

        if static_map_data.stop_sign_polyline_idxs is not None:
            color, alpha = self.map_colors["stop_sign"], self.map_alphas["stop_sign"]
            BaseVisualizer.plot_polylines(
                ax, road_graph, static_map_data.stop_sign_polyline_idxs, num_windows, color, alpha
            )

    def plot_dynamic_map_data(self, ax: Axes, dynamic_map_data: DynamicMapData, num_windows: int = 0) -> None:
        """Plots dynamic map features (e.g., stop points) for a scenario.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            dynamic_map_data (DynamicMapData): Dynamic map information.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
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
            # If there are multiple windows, propagate the polyline visualization.
            for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                a.scatter(x_pos, y_pos, s=6, c=color, marker="s", alpha=alpha)

    @staticmethod
    def plot_stop_signs(  # noqa: PLR0913
        ax: Axes,
        road_graph: np.ndarray,
        polyline_idxs: np.ndarray,
        num_windows: int = 0,
        color: str = "red",
        dim: int = 2,
    ) -> None:
        """Plots stop signs on the axes for a scenario using polyline indices.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            road_graph (np.ndarray): Road graph points.
            polyline_idxs (np.ndarray): Indices for stop sign polylines.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
            color (str, optional): Color for stop signs. Defaults to "red".
            dim (int, optional): Number of dimensions to plot. Defaults to 2.
        """
        for polyline in polyline_idxs:
            start_idx, end_idx = polyline
            pos = road_graph[start_idx:end_idx, :dim]
            if num_windows == 1:
                ax.scatter(pos[:, 0], pos[:, 1], s=16, c=color, marker="H", alpha=1.0)
            else:
                # If there are multiple windows, propagate the polyline visualization.
                for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                    a.scatter(pos[:, 0], pos[:, 1], s=16, c=color, marker="H", alpha=1.0)

    @staticmethod
    def plot_polylines(
        ax: Axes,
        road_graph: np.ndarray,
        polyline_idxs: np.ndarray,
        num_windows: int = 0,
        color: str = "k",
        alpha: float = 1.0,
        linewidth: float = 0.5,
    ) -> None:
        """Plots polylines (e.g., lanes, crosswalks) on the axes for a scenario.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            road_graph (np.ndarray): Road graph points.
            polyline_idxs (np.ndarray): Indices for polylines to plot.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
            color (str, optional): Color for polylines. Defaults to "k".
            alpha (float, optional): Alpha transparency. Defaults to 1.0.
            linewidth (float, optional): Line width. Defaults to 0.5.
        """
        for polyline in polyline_idxs:
            start_idx, end_idx = polyline
            pos = road_graph[start_idx:end_idx]
            if num_windows == 1:
                ax.plot(pos[:, 0], pos[:, 1], color, alpha=alpha, linewidth=linewidth)
            else:
                # If there are multiple windows, propagate the polyline visualization.
                for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                    a.plot(pos[:, 0], pos[:, 1], color, alpha=alpha, linewidth=linewidth)

    @staticmethod
    def to_gif(
        tmp_dir_frames: Path,
        output_filepath: Path,
        *,
        fps: int = 10,
        disposal: int = 2,
        loop: int = 0,
    ) -> None:
        """Saves scenario as a GIF.

        Args:
            tmp_dir_frames (Path): temporary directory where scenario image frames have been saved.
            output_filepath (Path): output filepath to save the GIF.
            fps (int): frames per second for the GIF.
            disposal (int): specifies how the previous frame should be treated before displaying the next frame.
                (Default value is 2 (restores background color, clear the previous frame))
            loop (int): number of times the GIF should loop.
        """
        t_i = time.time()
        # Load all the temporary files
        files = natsorted(glob(f"{tmp_dir_frames}/temp_*.png"))  # noqa: PTH207
        if not files:
            err_msg = f"No frames found in {tmp_dir_frames}, cannot create GIF."
            raise RuntimeError(err_msg)
        images_to_append = (Image.open(f) for f in files[1:])

        duration = 1000 / fps

        # Saves them into a GIF
        Image.open(files[0]).save(
            output_filepath,
            format="GIF",
            append_images=images_to_append,
            save_all=True,  # Ensures all frames are saved. Needed for preserving animation.
            duration=duration,
            disposal=disposal,
            loop=loop,
        )

        t_f = time.time()
        logger.info("Saved GIF to %s [Time taken: %.2fs]", output_filepath, t_f - t_i)

    @staticmethod
    def get_normalized_agent_scores(
        agent_scores: np.ndarray, ego_index: int, amin: float = 0.05, amax: float = 1.0
    ) -> np.ndarray:
        """Gets the agent scores and returns a normalized score array.

        Args:
            agent_scores (np.ndarray): array containing the agent scores.
            ego_index (int): index of the ego vehicle.
            amin (float): minimum value to clip the array.
            amax (float): maximum value to clip the array.

        Returns:
            np.ndarray: normalized agent scores.
        """
        min_score = np.nanmin(agent_scores)
        max_score = np.nanmax(agent_scores)

        if max_score > min_score:
            agent_scores = (agent_scores - min_score) / (max_score - min_score)
        else:
            # If all scores are identical, assign equal scores to all agents
            agent_scores = np.ones_like(agent_scores) / len(agent_scores)

        # Clip scores to avoid zero alpha values
        agent_scores = np.clip(agent_scores, a_min=amin, a_max=amax)

        # Set ego-agent to maximum alpha
        agent_scores[ego_index] = amax
        return agent_scores

    @staticmethod
    def get_first_and_last_ego_position(scenario: Scenario) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Gets the first and last valid positions of the ego vehicle.

        Args:
            scenario (Scenario): encapsulates the scenario to visualize.
            distance_to_zoom_in (float): minimum distance to zoom in around the ego vehicle.
            buffer_distance (float): additional buffer distance to add to the zoom-in distance.
        Returns:
            tuple[np.ndarray, float]: ego vehicle position and distance to set the plot limits.
        """
        ego_index = scenario.metadata.ego_vehicle_index
        agent_trajectories = AgentTrajectoryMasker(scenario.agent_data.agent_trajectories)
        agent_valid = agent_trajectories.agent_valid.squeeze(-1).astype(bool)
        agent_positions = agent_trajectories.agent_xy_pos

        # Get first valid ego position
        ego_traj = agent_positions[ego_index]

        valid_mask = agent_valid[ego_index] & np.all(np.isfinite(ego_traj), axis=-1)
        valid_ego_traj = ego_traj[valid_mask]
        if valid_ego_traj.shape[0] < 2:
            return None, None

        # Return first and last valid positions
        return valid_ego_traj[0],  valid_ego_traj[-1]

    def set_axes(self, ax: Axes, scenario: Scenario, num_windows: int = 1) -> None:
        """Plots dynamic map features (e.g., stop points) for a scenario.

        Args:
            ax (Axes): Axes to plot on.
            scenario (Scenario): encapsulates the scenario to visualize.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
        """
        first_ego_position, last_ego_position = BaseVisualizer.get_first_and_last_ego_position(scenario)
        if first_ego_position is None or last_ego_position is None:
            return

        ego_displacement = np.linalg.norm(first_ego_position - last_ego_position, axis=-1)
        distance = max(self.distance_to_ego_zoom_in, ego_displacement) + self.buffer_distance

        if num_windows == 1:
            ax.set_xticks([])
            ax.set_yticks([])

            if self.update_limits:
                ax.set_xlim(first_ego_position[0] - distance, first_ego_position[0] + distance)
                ax.set_ylim(first_ego_position[1] - distance, first_ego_position[1] + distance)

        else:
            for n, a in enumerate(ax.reshape(-1)):
                a.set_xticks([])
                a.set_yticks([])
                if n == 0:
                    continue

                if self.update_limits:
                    a.set_xlim(first_ego_position[0] - distance, first_ego_position[0] + distance)
                    a.set_ylim(first_ego_position[1] - distance, first_ego_position[1] + distance)

    @abstractmethod
    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: ScenarioScores | None = None,
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Visualizes a single scenario and saves the output to a file.

        This method should be implemented by subclasses to provide scenario-specific visualization,
        supporting flexible titles and output paths. It is designed to handle both static and dynamic map
        features, as well as agent trajectories and attributes.

        Args:
            scenario (Scenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            output_dir (str): the directory where to save the scenario visualization.

        Returns:
            Path: The path to the saved visualization file.
        """
