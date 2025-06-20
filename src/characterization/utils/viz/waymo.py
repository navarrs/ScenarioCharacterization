import matplotlib.pyplot as plt
import numpy as np

from characterization.utils.schemas import Scenario
from characterization.utils.viz.visualizer import BaseVisualizer

# from matplotlib.patches import Rectangle


class WaymoVisualizer(BaseVisualizer):
    def __init__(self, config):
        super().__init__(config)

    def visualize_scenario(
        self,
        scenario: Scenario,
        scores: dict = {},  # Optional scores for the scenario
        title: str = "Scenario",
        output_filepath: str = "temp.png",
    ) -> None:
        """
        Visualizes a single Waymo scenario, including static and dynamic map elements, and agent trajectories.

        This method creates a two-panel visualization: one showing all agent trajectories, and one highlighting relevant
        and SDC (self-driving car) agents. It overlays static and dynamic map features and saves the visualization to a
        file.

        Args:
            scenario (Scenario): The scenario data to visualize.
            title (str, optional): Title for the visualization. Defaults to "Scenario".
            output_filepath (str, optional): Path to save the visualization output. Defaults to "temp.png".

        Returns:
            None
        """
        num_windows = 2
        fig, axs = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))

        # Plot static map information
        if scenario.map_polylines is None:
            print("[WARNING] Scenario does not contain map_polylines, skipping static map visualization.")
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

        breakpoint()
        self.plot_dynamic_map_infos(axs, scenario.dynamic_map_info, num_windows=num_windows)

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
        ax.cla()
        plt.close()

    def plot_agent(
        self, ax: plt.Axes, x: float, y: float, heading: float, width: float, height: float, alpha: float
    ) -> None:
        """
        Plots a single agent on the given axes.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on.
            pos (np.ndarray): Position of the agent.
            heading (float): Heading angle of the agent.
            width (float): Width of the agent.
            height (float): Height of the agent.

        Returns:
            None
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

    def plot_sequences(self, ax: plt.Axes, scenario: Scenario, scores, show_relevant: bool = False) -> None:
        """
        Plots agent trajectories for a scenario, with optional highlighting of relevant and SDC agents.

        Args:
            scenario (Scenario): Scenario data containing agent positions, types, and relevance.
            ax (matplotlib.axes.Axes): Axes to plot on.
            show_relevant (bool, optional): If True, highlights relevant and SDC agents. Defaults to False.

        Returns:
            None
        """
        agent_positions = scenario.agent_positions
        agent_dimensions = scenario.agent_dimensions  # length, width, height
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

        zipped = zip(agent_positions, agent_dimensions, agent_headings, agent_valid, agent_types, scores)
        for apos, adim, ahead, amask, atype, score in zipped:
            amask = amask.squeeze(-1)
            if not amask.any() or amask.sum() < 2:
                continue

            pos = apos[amask, :]
            heading = ahead[amask][0]
            lenght = adim[0, 0]
            width = adim[0, 1]
            color = self.agent_colors[atype]
            ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, alpha=score)
            # Plot the agent
            self.plot_agent(ax, pos[0, 0], pos[0, 1], heading, lenght, width, score)

    def plot_static_map_infos(
        self,
        ax: plt.Axes,
        map_polylines: np.ndarray | None = None,
        lane_polyline_idxs: np.ndarray | None = None,
        road_line_polyline_idxs: np.ndarray | None = None,
        road_edge_polyline_idxs: np.ndarray | None = None,
        crosswalk_polyline_idxs: np.ndarray | None = None,
        speed_bump_polyline_idxs: np.ndarray | None = None,
        stop_sign_polyline_idxs: np.ndarray | None = None,
        num_windows: int = 0,
        dim: int = 2,
    ) -> None:
        """
        Plots static map information such as lanes, stop signs, and crosswalks for a scenario.

        Args:
            map_information (dict): Dictionary containing static map information.
            ax (matplotlib.axes.Axes): Axes to plot on.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
            dim (int, optional): Number of dimensions to plot. Defaults to 2.

        Returns:
            dict: Dictionary of plotted map info positions.
        """
        road_graph = map_polylines[:, :dim]

        # Plot lanes
        if lane_polyline_idxs is not None:
            self.plot_polylines(road_graph, lane_polyline_idxs, ax, num_windows, color=self.map_colors["lane"], dim=dim)
        if road_line_polyline_idxs is not None:
            self.plot_polylines(
                road_graph, road_line_polyline_idxs, ax, num_windows, color=self.map_colors["road_line"], dim=dim
            )
        if road_edge_polyline_idxs is not None:
            self.plot_polylines(
                road_graph, road_edge_polyline_idxs, ax, num_windows, color=self.map_colors["road_edge"], dim=dim
            )
        if crosswalk_polyline_idxs is not None:
            self.plot_polylines(
                road_graph, crosswalk_polyline_idxs, ax, num_windows, color=self.map_colors["crosswalk"], dim=dim
            )
        if speed_bump_polyline_idxs is not None:
            self.plot_polylines(
                road_graph, speed_bump_polyline_idxs, ax, num_windows, color=self.map_colors["speed_bump"], dim=dim
            )

        # map_infos_pos = {}
        # for key in self.static_map_keys:
        #     if key not in map_information.keys():
        #         continue

        #     if key == "stop_sign":
        #         map_infos_pos[key] = self.plot_stop_signs(
        #             map_information[key], ax, num_windows, color=self.map_colors[key], dim=dim
        #         )
        #     else:
        #         map_infos_pos[key] = self.plot_polylines(
        #             map_information[key],
        #             road_graph,
        #             ax,
        #             num_windows,
        #             color=self.map_colors[key],
        #             alpha=self.map_alphas[key],
        #             dim=dim,
        #         )

    def plot_dynamic_map_infos(self, ax: plt.Axes, map_information: dict, num_windows: int = 0, dim: int = 2):
        """
        Plots dynamic map information such as stop points for a scenario.

        Args:
            map_information (dict): Dictionary containing dynamic map information.
            ax (matplotlib.axes.Axes): Axes to plot on.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
            dim (int, optional): Number of dimensions to plot. Defaults to 2.

        Returns:
            dict: Dictionary of plotted dynamic map info positions.
        """
        for key in self.dynamic_map_keys:
            if key not in map_information.keys():
                continue
            if key == "stop_point":
                if len(map_information[key]) <= 0:
                    continue
                stop_points = map_information[key][0]
                for i in range(stop_points.shape[1]):
                    pos = stop_points[0, i, :2]
                    if ax is None:
                        continue

                    if num_windows == 1:
                        ax.scatter(pos[0], pos[1], s=6, c=self.map_colors[key], marker="s", alpha=self.map_alphas[key])
                    else:
                        for a in ax.reshape(-1):
                            a.scatter(
                                pos[0], pos[1], s=6, c=self.map_colors[key], marker="s", alpha=self.map_alphas[key]
                            )

    def plot_stop_signs(
        self, stop_signs, ax: plt.Axes = None, num_windows: int = 0, color: str = "red", dim: int = 2
    ) -> np.ndarray:
        """
        Plots stop signs on the given axes for a scenario.

        Args:
            stop_signs (list): List of stop sign dictionaries with 'position'.
            ax (matplotlib.axes.Axes, optional): Axes to plot on.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
            color (str, optional): Color for the stop signs. Defaults to "red".
            dim (int, optional): Number of dimensions to plot. Defaults to 2.

        Returns:
            np.ndarray: Array of stop sign positions.
        """
        stop_sign_xy = np.zeros(shape=(len(stop_signs), dim))
        for i, stop_sign in enumerate(stop_signs):
            pos = stop_sign["position"]
            stop_sign_xy[i] = pos[:dim]
            if ax is None:
                continue

            if num_windows == 1:
                ax.scatter(pos[0], pos[1], s=16, c=color, marker="H", alpha=1.0)
            else:
                for a in ax.reshape(-1):
                    a.scatter(pos[0], pos[1], s=16, c=color, marker="H", alpha=1.0)

        return stop_sign_xy

    def plot_polylines(
        self,
        road_graph: np.ndarray,
        polyline_idxs: np.ndarray,
        ax: plt.Axes,
        num_windows=0,
        color="k",
        alpha=1.0,
        linewidth=0.5,
        dim=2,
    ):
        """
        Plots polylines (e.g., lanes, crosswalks) on the given axes for a scenario.

        Args:
            polylines (list): List of polyline dictionaries with 'polyline_index'.
            road_graph (np.ndarray): Array of road graph points.
            ax (matplotlib.axes.Axes, optional): Axes to plot on.
            num_windows (int, optional): Number of subplot windows. Defaults to 0.
            color (str, optional): Color for the polylines. Defaults to "k".
            alpha (float, optional): Alpha transparency. Defaults to 1.0.
            linewidth (float, optional): Line width. Defaults to 0.5.
            dim (int, optional): Number of dimensions to plot. Defaults to 2.

        Returns:
            list: List of polyline position arrays.
        """
        for polyline in polyline_idxs:
            start_idx, end_idx = polyline
            polyline_pos = road_graph[start_idx:end_idx, :dim]
            if num_windows == 1:
                ax.plot(
                    polyline_pos[:, 0],
                    polyline_pos[:, 1],
                    color,
                    alpha=alpha,
                    linewidth=linewidth,
                    ms=2,
                )
            else:
                for a in ax.reshape(-1):
                    a.plot(
                        polyline_pos[:, 0],
                        polyline_pos[:, 1],
                        color,
                        alpha=alpha,
                        linewidth=linewidth,
                        ms=2,
                    )
