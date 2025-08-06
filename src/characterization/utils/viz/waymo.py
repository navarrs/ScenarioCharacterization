import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from omegaconf import DictConfig

from characterization.utils.common import get_logger
from characterization.utils.schemas.schemas import Scenario
from characterization.utils.viz.visualizer import BaseVisualizer

# from matplotlib.patches import Rectangle


logger = get_logger(__name__)


class WaymoVisualizer(BaseVisualizer):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: np.ndarray,
        title: str = "Scenario",
        output_filepath: str = "temp.png",
    ) -> None:
        """Visualizes a Waymo scenario, including static/dynamic map elements and agent trajectories.

        Creates a two-panel plot: one for all agent trajectories, one highlighting relevant and SDC agents.
        Overlays static and dynamic map features. Optionally uses agent scores for transparency.

        Args:
            scenario (Scenario): Scenario data to visualize.
            scores (dict, optional): Optional agent scores for transparency. Defaults to empty dict.
            title (str, optional): Title for the visualization. Defaults to "Scenario".
            output_filepath (str, optional): Path to save the visualization. Defaults to "temp.png".
        """
        num_windows = 2
        _, axs = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))

        # Plot static map information
        if scenario.map_polylines is None:
            logger.warning("Scenario does not contain map_polylines, skipping static map visualization.")
        else:
            self.plot_static_map_infos(
                axs,
                map_polylines=scenario.map_polylines,
                lane_polyline_idxs=scenario.lane_polyline_idxs,
                road_line_polyline_idxs=scenario.road_line_polyline_idxs,
                road_edge_polyline_idxs=scenario.road_edge_polyline_idxs,
                crosswalk_polyline_idxs=scenario.crosswalk_polyline_idxs,
                speed_bump_polyline_idxs=scenario.speed_bump_polyline_idxs,
                stop_sign_polyline_idxs=scenario.stop_sign_polyline_idxs,
                num_windows=num_windows,
            )

        # Plot dynamic map information
        if scenario.dynamic_stop_points is None:
            logger.warning("Scenario does not contain dynamic_map_info, skipping dynamic map visualization.")
        else:
            self.plot_dynamic_map_infos(axs, scenario.dynamic_stop_points, num_windows=num_windows)

        self.plot_sequences(axs[0], scenario, scores)
        self.plot_sequences(axs[1], scenario, scores, show_relevant=True)

        for ax in axs.reshape(-1):
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(title)
        axs[0].set_title("All Agents Trajectories")
        axs[1].set_title("Highlighted Relevant and SDC Agent Trajectories")
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        axs.cla()
        plt.close()

    def plot_agent(
        self,
        ax: Axes,
        x: float,
        y: float,
        heading: float,
        width: float,
        height: float,
        alpha: float,
    ) -> None:
        """Plots a single agent as a point (optionally as a rectangle) on the axes.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            x (float): X position of the agent.
            y (float): Y position of the agent.
            heading (float): Heading angle of the agent.
            width (float): Width of the agent.
            height (float): Height of the agent.
            alpha (float): Transparency for the agent marker.
        """
        ax.scatter(x, y, s=8, zorder=1000, c="magenta", marker="o", alpha=alpha)
        # angle_deg = np.rad2deg(heading)
        # rect = Rectangle(
        #     (x - width / 2, y - height / 2),
        #     width,
        #     height,
        #     angle=angle_deg,
        #     # linewidth=2,
        #     # edgecolor='blue',
        #     facecolor='magenta',
        #     alpha=alpha,
        #     zorder=100,
        # )
        # ax.add_patch(rect)

    def plot_sequences(self, ax: Axes, scenario: Scenario, scores: np.ndarray, show_relevant: bool = False) -> None:
        """Plots agent trajectories for a scenario, with optional highlighting and score-based transparency.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            scenario (Scenario): Scenario data with agent positions, types, and relevance.
            scores (array-like): Per-agent scores for transparency (normalized).
            show_relevant (bool, optional): If True, highlights relevant and SDC agents. Defaults to False.
        """
        agent_positions = scenario.agent_positions
        agent_lengths = scenario.agent_lengths
        agent_widths = scenario.agent_widths
        agent_headings = scenario.agent_headings
        agent_types = scenario.agent_types
        agent_valid = scenario.agent_valid
        agent_relevance = scenario.agent_relevance
        ego_index = scenario.ego_index
        relevant_indeces = np.where(agent_relevance > 0.0)[0]

        min_score = np.nanmin(scores)
        max_score = np.nanmax(scores)
        if max_score > min_score:
            scores = np.clip((scores - min_score) / (max_score - min_score), a_min=0.05, a_max=1.0)
        else:
            scores = 0.05 * np.ones_like(scores)

        if show_relevant:
            # TODO: make agent_types a numpy array
            for idx in relevant_indeces:
                agent_types[idx] = "TYPE_RELEVANT"
            agent_types[ego_index] = "TYPE_SDC"  # Mark ego agent for visualization

        zipped = zip(
            agent_positions, agent_lengths, agent_widths, agent_headings, agent_valid, agent_types, scores, strict=False
        )
        for apos, alen, awid, ahead, amask, atype, score in zipped:
            amask = amask.squeeze(-1)
            if not amask.any() or amask.sum() < 2:
                continue

            pos = apos[amask, :]
            heading = ahead[amask][0]
            length = alen[amask]
            width = awid[amask]
            color = self.agent_colors[atype]
            ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, alpha=score)
            # Plot the agent
            self.plot_agent(ax, pos[0, 0], pos[0, 1], heading, length, width, score)

    def plot_static_map_infos(
        self,
        ax: Axes,
        map_polylines: np.ndarray,
        lane_polyline_idxs: np.ndarray | None = None,
        road_line_polyline_idxs: np.ndarray | None = None,
        road_edge_polyline_idxs: np.ndarray | None = None,
        crosswalk_polyline_idxs: np.ndarray | None = None,
        speed_bump_polyline_idxs: np.ndarray | None = None,
        stop_sign_polyline_idxs: np.ndarray | None = None,
        num_windows: int = 0,
        dim: int = 2,
    ) -> None:
        """Plots static map features (lanes, road lines, crosswalks, etc.) for a scenario.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            map_polylines (np.ndarray): All map polylines.
            lane_polyline_idxs (np.ndarray, optional): Lane polyline indices.
            road_line_polyline_idxs (np.ndarray, optional): Road line polyline indices.
            road_edge_polyline_idxs (np.ndarray, optional): Road edge polyline indices.
            crosswalk_polyline_idxs (np.ndarray, optional): Crosswalk polyline indices.
            speed_bump_polyline_idxs (np.ndarray, optional): Speed bump polyline indices.
            stop_sign_polyline_idxs (np.ndarray, optional): Stop sign polyline indices.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
            dim (int, optional): Number of dimensions to plot. Defaults to 2.
        """
        road_graph = map_polylines[:, :dim]
        if lane_polyline_idxs is not None:
            color, alpha = self.map_colors["lane"], self.map_alphas["lane"]
            self.plot_polylines(ax, road_graph, lane_polyline_idxs, num_windows, color=color, alpha=alpha)
        if road_line_polyline_idxs is not None:
            color, alpha = self.map_colors["road_line"], self.map_alphas["road_line"]
            self.plot_polylines(ax, road_graph, road_line_polyline_idxs, num_windows, color=color, alpha=alpha)
        if road_edge_polyline_idxs is not None:
            color, alpha = self.map_colors["road_edge"], self.map_alphas["road_edge"]
            self.plot_polylines(ax, road_graph, road_edge_polyline_idxs, num_windows, color=color, alpha=alpha)
        if crosswalk_polyline_idxs is not None:
            color, alpha = self.map_colors["crosswalk"], self.map_alphas["crosswalk"]
            self.plot_polylines(ax, road_graph, crosswalk_polyline_idxs, num_windows, color, alpha)
        if speed_bump_polyline_idxs is not None:
            color, alpha = self.map_colors["speed_bump"], self.map_alphas["speed_bump"]
            self.plot_polylines(ax, road_graph, speed_bump_polyline_idxs, num_windows, color=color, alpha=alpha)
        if stop_sign_polyline_idxs is not None:
            color, alpha = self.map_colors["stop_sign"], self.map_alphas["stop_sign"]
            self.plot_stop_signs(ax, road_graph, stop_sign_polyline_idxs, num_windows, color=color)

    def plot_dynamic_map_infos(
        self,
        ax: Axes,
        dynamic_stop_points: np.ndarray,
        num_windows: int = 0,
        dim: int = 2,
    ) -> None:
        """Plots dynamic map features (e.g., stop points) for a scenario.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            dynamic_stop_points (np.ndarray): Array of stop point positions.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
            dim (int, optional): Number of dimensions to plot. Defaults to 2.
        """
        x_pos = dynamic_stop_points[:, 0]
        y_pos = dynamic_stop_points[:, 1]
        color = self.map_colors["stop_point"]
        alpha = self.map_alphas["stop_point"]
        if num_windows == 1:
            ax.scatter(x_pos, y_pos, s=6, c=color, marker="s", alpha=alpha)
        else:
            for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                a.scatter(x_pos, y_pos, s=6, c=color, marker="s", alpha=alpha)

    def plot_stop_signs(
        self,
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
                for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                    a.scatter(pos[:, 0], pos[:, 1], s=16, c=color, marker="H", alpha=1.0)

    def plot_polylines(
        self,
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
                for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
                    a.plot(pos[:, 0], pos[:, 1], color, alpha=alpha, linewidth=linewidth)
