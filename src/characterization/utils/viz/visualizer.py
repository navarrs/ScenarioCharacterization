import os
from abc import ABC, abstractmethod
from glob import glob

import numpy as np
from characterization.schemas import DynamicMapData, Scenario, ScenarioScores, StaticMapData
from characterization.utils.common import SUPPORTED_SCENARIO_TYPES, AgentTrajectoryMasker, MIN_VALID_POINTS
from characterization.utils.io_utils import get_logger
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from natsort import natsorted
from omegaconf import DictConfig
from PIL import Image

logger = get_logger(__name__)


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

        self.buffer_distance = config.get("distance_to_ego_zoom_in", 5.0)  # in meters
        self.distance_to_ego_zoom_in = config.get("distance_to_ego_zoom_in", 50.0)  # in meters

    def plot_map_data(self, ax: Axes, scenario: Scenario, num_windows: int = 1) -> None:
        """Plots the map data.

        Args:
            ax (Axes): Axes to plot on.
            scenario (Scenario): encapsulates the scenario to visualize.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
        """
        # Plot static map information
        if scenario.static_map_data is None:
            logger.warning("Scenario does not contain map_polylines, skipping static map visualization.")
        else:
            self.plot_static_map_data(ax, static_map_data=scenario.static_map_data, num_windows=num_windows)

        # Plot dynamic map information
        if scenario.dynamic_map_data is None:
            logger.warning("Scenario does not contain dynamic_map_info, skipping dynamic map visualization.")
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
    ) -> None:
        """Plots agent trajectories for a scenario, with optional highlighting and score-based transparency.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            scenario (Scenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            show_relevant (bool, optional): If True, highlights relevant and SDC agents. Defaults to False.
            start_timestep (int): starting timestep to plot the sequences.
            end_timestep (int): ending timestep to plot the sequences.
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
            color = self.agent_colors[atype]

            # Plot the trajectory
            ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, alpha=score)

            # Plot the agent
            self.plot_agent(ax, pos[-1, 0], pos[-1, 1], heading, length, width, score, color, plot_rectangle=True)

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
        output_dir: str,
        output_filepath: str,
        *,
        duration: int = 100,
        disposal: int = 2,
        loop: int = 0,
    ) -> None:
        """Saves scenario as a GIF.

        Args:
            output_dir (str): directory where temporary scenario files have been saved.
            output_filepath (str): output filepath to save the GIF.
            duration (int): duration of each frame.
            disposal (int): specifies how the previous frame should be treated before displaying the next frame.
                (Default value is 2 (restores background color, clear the previous frame))
            loop (int): number of times the GIF should loop.
        """
        # Load all the temporary files
        files = glob(f"{output_dir}/temp_*.png")  # noqa: PTH207
        imgs = [Image.open(f) for f in natsorted(files)]

        # Saves them into a GIF
        imgs[0].save(
            output_filepath,
            format="GIF",
            append_images=imgs[1:],
            save_all=True,  # Ensures all frames are saved. Needed for preserving animation.
            duration=duration,
            disposal=disposal,
            loop=loop,
        )

        # Removes the temporary files
        for f in files:
            os.remove(f)  # noqa: PTH107

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
            agent_scores = np.clip((agent_scores - min_score) / (max_score - min_score), a_min=amin, a_max=amax)
        else:
            agent_scores = 1.0 - 2 * np.ones_like(agent_scores) / len(agent_scores)

        agent_scores[ego_index] = amax
        return agent_scores

    def set_axes(self, ax: Axes, scenario: Scenario, num_windows: int = 1) -> None:
        """Plots dynamic map features (e.g., stop points) for a scenario.

        Args:
            ax (Axes): Axes to plot on.
            scenario (Scenario): encapsulates the scenario to visualize.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
        """
        ego_index = scenario.metadata.ego_vehicle_index
        agent_positions = AgentTrajectoryMasker(scenario.agent_data.agent_trajectories).agent_xy_pos
        ego_position = agent_positions[ego_index, 0]
        last_ego_position = agent_positions[ego_index, -1]
        ego_displacement = np.linalg.norm(ego_position - last_ego_position, axis=-1)
        distance = max(self.distance_to_ego_zoom_in, ego_displacement) + self.buffer_distance

        if num_windows == 1:
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlim(ego_position[0] - distance, ego_position[0] + distance)
            ax.set_ylim(ego_position[1] - distance, ego_position[1] + distance)

        else:
            for n, a in enumerate(ax.reshape(-1)):
                a.set_xticks([])
                a.set_yticks([])
                if n == 0:
                    continue

                a.set_xlim(ego_position[0] - distance, ego_position[0] + distance)
                a.set_ylim(ego_position[1] - distance, ego_position[1] + distance)

    @abstractmethod
    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: ScenarioScores | None = None,
        output_dir: str = "temp",
    ) -> None:
        """Visualizes a single scenario and saves the output to a file.

        This method should be implemented by subclasses to provide scenario-specific visualization,
        supporting flexible titles and output paths. It is designed to handle both static and dynamic map
        features, as well as agent trajectories and attributes.

        Args:
            scenario (Scenario): encapsulates the scenario to visualize.
            scores (ScenarioScores | None): encapsulates the scenario and agent scores.
            output_dir (str): the directory where to save the scenario visualization.
        """
