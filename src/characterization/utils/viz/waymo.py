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

        self.plot_static_map_infos(axs, scenario.static_map_info, num_windows=num_windows)
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

    def plot_static_map_infos(self, ax: plt.Axes, map_information: dict, num_windows: int = 0, dim: int = 2) -> None:
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
        road_graph = map_information["all_polylines"][:, :dim]

        map_infos_pos = {}
        for key in self.static_map_keys:
            if key not in map_information.keys():
                continue

            if key == "stop_sign":
                map_infos_pos[key] = self.plot_stop_signs(
                    map_information[key], ax, num_windows, color=self.map_colors[key], dim=dim
                )
            else:
                map_infos_pos[key] = self.plot_polylines(
                    map_information[key],
                    road_graph,
                    ax,
                    num_windows,
                    color=self.map_colors[key],
                    alpha=self.map_alphas[key],
                    dim=dim,
                )

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
        polylines: np.ndarray,
        road_graph: np.ndarray,
        ax: plt.Axes = None,
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
        polyline_pos_list = []
        for pl in polylines:
            start_idx, end_idx = pl["polyline_index"]
            polyline_pos = road_graph[start_idx:end_idx, :dim]
            polyline_pos_list.append(polyline_pos)
            if ax is None:
                continue
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
        return polyline_pos_list


# --------------------------------------------------------------------------------------------------
# NOTE: Unused functions for now
# import itertools
# import uuid
# import matplotlib.animation as animation
# from matplotlib import cm
# --------------------------------------------------------------------------------------------------
# def get_color_map(num_colors):
#     """Returns a color map dictionary with a specified number of unique colors.

#     Args:
#         num_colors (int): The number of colors to include in the color map.

#     Returns:
#         dict: A dictionary mapping indices to color names.

#     Raises:
#         AssertionError: If num_colors is not in the valid range.
#     """
#     import matplotlib.colors as mcolors

#     color_dict = mcolors.CSS4_COLORS
#     max_colors = len(color_dict.keys())
#     if num_colors < 1 or num_colors > max_colors:
#         raise AssertionError(
#             f"num_colors must be between 1 and {max_colors}, but got {num_colors}."
#         )
#     assert num_colors > 0 and num_colors <= len(
#         color_dict.keys()
#     ), f"Max. num, of colors is {max_colors}; requested {num_colors}"

#     color_map = {}
#     for i, (k, v) in enumerate(color_dict.items()):
#         if i > num_colors:
#             break
#         color_map[i] = k
#     return color_map


# def plot_cluster_overlap(num_clusters, num_components, labels, scores, shards, tag):
#     """Plots the overlap between clusters in a 2D PCA space.

#     Args:
#         num_clusters (int): Number of clusters.
#         num_components (int): Number of PCA components.
#         labels (np.ndarray): Cluster labels for each sample.
#         scores (np.ndarray): 2D PCA scores for each sample.
#         shards (list): List of shard indices.
#         tag (str): Tag for the output filename.

#     Returns:
#         None
#     """
#     fig, ax = plt.subplots(num_clusters, num_clusters, figsize=(15, 15))
#     plt.subplots_adjust(wspace=0.05, hspace=0.05)
#     for a in ax.reshape(-1):
#         a.set_xticks([])
#         a.set_yticks([])

#     for i, j in itertools.combinations(range(num_clusters), 2):
#         color_i, color_j = "blue", "orange"
#         idx_i = np.where(labels == i)
#         ax[i, j].scatter(scores[idx_i, 0], scores[idx_i, 1], color=color_i)
#         idx_j = np.where(labels == j)
#         ax[i, j].scatter(scores[idx_j, 0], scores[idx_j, 1], color=color_j)

#     filename = f"{tag}_kmeans-{num_clusters}_pca-{num_components}_overlap_shards{shards[0]}-{shards[-1]}.png"
#     plt.savefig(filename, dpi=300, bbox_inches="tight")
#     plt.close()


# def plot_lanes_by_distance(lanes, order, dists, ax, k=-1):
#     """Plots lanes colored by their distance.

#     Args:
#         lanes (list): List of lane arrays.
#         order (np.ndarray): Order of lane indices.
#         dists (np.ndarray): Distances for each lane.
#         ax (matplotlib.axes.Axes): Axes to plot on.
#         k (int, optional): Number of lanes to plot. If -1, plot all.

#     Returns:
#         None
#     """
#     if k == -1:
#         # ndists = 1 - np.clip((dists - dists.mean()) / dists.std(), 0.0, 0.1)
#         ndists = 1 - np.clip(dists / dists.max(), 0.0, 1.0)
#         for lane_idx in order:
#             lane = lanes[lane_idx].T
#             ax.plot(
#                 lane[0],
#                 lane[1],
#                 c=cm.winter(ndists[lane_idx]),
#                 alpha=ndists[lane_idx],
#                 linewidth=0.5,
#             )
#     else:
#         order = order[:k]
#         dists = dists[order]
#         # ndists = 1 - np.clip((dists - dists.mean()) / dists.std(), 0.0, 0.1)
#         ndists = 1 - np.clip(dists / dists.max(), 0.0, 1.0)

#         for i, lane_idx in enumerate(order):
#             lane = lanes[lane_idx].T
#             ax.plot(lane[0], lane[1], c=cm.winter(ndists[i]), alpha=1.0, linewidth=0.5)


# def plot_interaction(
#     ax,
#     ax_idx,
#     pos_i,
#     agent_type_i,
#     i_idx,
#     traj_i,
#     pos_j,
#     agent_type_j,
#     j_idx,
#     traj_j,
#     title,
# ):
#     """Plots the interaction between two agents.

#     Args:
#         ax (np.ndarray): Array of matplotlib axes.
#         ax_idx (int): Index of the subplot.
#         pos_i (np.ndarray): Positions of agent i.
#         agent_type_i (str): Type of agent i.
#         i_idx (int): Index of conflict point for agent i.
#         traj_i (np.ndarray): Trajectory of agent i.
#         pos_j (np.ndarray): Positions of agent j.
#         agent_type_j (str): Type of agent j.
#         j_idx (int): Index of conflict point for agent j.
#         traj_j (np.ndarray): Trajectory of agent j.
#         title (str): Title for the subplot.

#     Returns:
#         None
#     """
#     ax[ax_idx].scatter(
#         pos_i[0, 0],
#         pos_i[0, 1],
#         color=AGENT_COLOR[agent_type_i],
#         marker="*",
#         s=10,
#         label="Start",
#     )
#     ax[ax_idx].plot(
#         pos_i[:, 0],
#         pos_i[:, 1],
#         color=AGENT_COLOR[agent_type_i],
#         alpha=0.6,
#         linewidth=1,
#     )

#     ax[ax_idx].scatter(pos_j[0, 0], pos_j[0, 1], color=AGENT_COLOR[agent_type_j], marker="*", s=10)
#     ax[ax_idx].plot(
#         pos_j[:, 0],
#         pos_j[:, 1],
#         color=AGENT_COLOR[agent_type_j],
#         alpha=1,
#         linewidth=1,
#         linestyle="dashed",
#     )

#     if i_idx != -1:
#         ax[ax_idx].scatter(
#             traj_i[i_idx, 0],
#             traj_i[i_idx, 1],
#             color="red",
#             marker="+",
#             s=10,
#             label="Conflict Point",
#         )
#     if j_idx != -1:
#         ax[ax_idx].scatter(traj_j[j_idx, 0], traj_j[j_idx, 1], color="red", marker="+", s=10)

#     ax[ax_idx].legend()
#     ax[ax_idx].set_title(title)


# # --------------------------------------------------------------------------------------------------
# # Unused so far
# # --------------------------------------------------------------------------------------------------


# # Taken from: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_motion.ipynb
# def create_figure_and_axes(size_pixels):
#     """Initializes a unique figure and axes for plotting.

#     Args:
#         size_pixels (int): Size of the figure in pixels.

#     Returns:
#         tuple: (fig, ax) Matplotlib figure and axes.
#     """
#     fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

#     # Sets output image to pixel resolution.
#     dpi = 100
#     size_inches = size_pixels / dpi
#     fig.set_size_inches([size_inches, size_inches])
#     fig.set_dpi(dpi)
#     fig.set_facecolor("white")
#     ax.set_facecolor("white")
#     ax.xaxis.label.set_color("black")
#     ax.tick_params(axis="x", colors="black")
#     ax.yaxis.label.set_color("black")
#     ax.tick_params(axis="y", colors="black")
#     fig.set_tight_layout(True)
#     ax.grid(False)
#     return fig, ax


# def fig_canvas_image(fig):
#     """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb().

#     Args:
#         fig (matplotlib.figure.Figure): The figure to convert.

#     Returns:
#         np.ndarray: Image array of the figure canvas.
#     """
#     # Just enough margin in the figure to display xticks and yticks.
#     fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
#     fig.canvas.draw()
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


# def get_colormap(num_agents):
#     """Compute a color map array of shape [num_agents, 4].

#     Args:
#         num_agents (int): Number of agents.

#     Returns:
#         np.ndarray: Array of RGBA colors.
#     """
#     colors = cm.get_cmap("jet", num_agents)
#     colors = colors(range(num_agents))
#     np.random.shuffle(colors)
#     return colors


# def get_viewport(all_states, all_states_mask):
#     """Gets the region containing the data.

#     Args:
#         all_states (np.ndarray): States of agents as an array of shape [num_agents, num_steps, 2].
#         all_states_mask (np.ndarray): Binary mask of shape [num_agents, num_steps] for all_states.

#     Returns:
#         tuple: (center_y, center_x, width) for the viewport.
#     """
#     valid_states = all_states[all_states_mask]
#     all_y = valid_states[..., 1]
#     all_x = valid_states[..., 0]

#     center_y = (np.max(all_y) + np.min(all_y)) / 2
#     center_x = (np.max(all_x) + np.min(all_x)) / 2

#     range_y = np.ptp(all_y)
#     range_x = np.ptp(all_x)

#     width = max(range_y, range_x)

#     return center_y, center_x, width


# def visualize_one_step(
#     states,
#     mask,
#     roadgraph,
#     title,
#     center_y,
#     center_x,
#     width,
#     color_map,
#     size_pixels=1000,
# ):
#     """Generate visualization for a single step.

#     Args:
#         states (np.ndarray): Agent states for the current step.
#         mask (np.ndarray): Mask for valid agent states.
#         roadgraph (np.ndarray): Road graph points.
#         title (str): Title for the plot.
#         center_y (float): Center y-coordinate for the viewport.
#         center_x (float): Center x-coordinate for the viewport.
#         width (float): Width of the viewport.
#         color_map (np.ndarray): Color map for agents.
#         size_pixels (int, optional): Size of the output image in pixels.

#     Returns:
#         np.ndarray: Image array of the visualization.
#     """
#     # Create figure and axes.
#     fig, ax = create_figure_and_axes(size_pixels=size_pixels)

#     # Plot roadgraph.
#     rg_pts = roadgraph[:, :2].T
#     ax.plot(rg_pts[0, :], rg_pts[1, :], "k.", alpha=1, ms=2)

#     masked_x = states[:, 0][mask]
#     masked_y = states[:, 1][mask]
#     colors = color_map[mask]

#     # Plot agent current position.
#     ax.scatter(
#         masked_x,
#         masked_y,
#         marker="o",
#         linewidths=3,
#         color=colors,
#     )

#     # Title.
#     ax.set_title(title)

#     # Set axes.  Should be at least 10m on a side and cover 160% of agents.
#     size = max(10, width * 1.0)
#     ax.axis(
#         [
#             -size / 2 + center_x,
#             size / 2 + center_x,
#             -size / 2 + center_y,
#             size / 2 + center_y,
#         ]
#     )
#     ax.set_aspect("equal")

#     image = fig_canvas_image(fig)
#     plt.close(fig)
#     return image


# def create_animation(images):
#     """Creates a Matplotlib animation of the given images.

#     Args:
#         images (list): A list of numpy arrays representing the images.

#     Returns:
#         matplotlib.animation.Animation: The created animation.
#     """
#     plt.ioff()
#     fig, ax = plt.subplots()
#     dpi = 100
#     size_inches = 1000 / dpi
#     fig.set_size_inches([size_inches, size_inches])
#     plt.ion()

#     def animate_func(i):
#         ax.imshow(images[i])
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.grid("off")

#     anim = animation.FuncAnimation(fig, animate_func, frames=len(images) // 2, interval=100)
#     plt.close(fig)
#     return anim


# def visualize_all_agents_smooth(
#     decoded_example,
#     size_pixels=1000,
# ):
#     """Visualizes all agent predicted trajectories in a series of images.

#     Args:
#         decoded_example (dict): Dictionary containing scenario information.
#         size_pixels (int, optional): The size in pixels of the output image.

#     Returns:
#         list: List of [H, W, 3] uint8 np.arrays of the drawn matplotlib's figure canvas.
#     """
#     # [num_agents, num_past_steps, 2] float32.
#     # past_states = tf.stack(
#     #     [decoded_example['state/past/x'], decoded_example['state/past/y']],
#     #     -1).numpy()
#     # past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0
#     past_states = decoded_example["track_infos"]["trajs"][:, :10, :2]
#     past_states_mask = decoded_example["track_infos"]["trajs"][:, :10, -1] > 0.0

#     # [num_agents, 1, 2] float32.
#     # current_states = tf.stack(
#     #     [decoded_example['state/current/x'], decoded_example['state/current/y']],
#     #     -1).numpy()
#     # current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0
#     current_states = decoded_example["track_infos"]["trajs"][:, 10, :2][:, np.newaxis, :]
#     current_states_mask = decoded_example["track_infos"]["trajs"][:, 10, -1][:, np.newaxis] > 0.0

#     # [num_agents, num_future_steps, 2] float32.
#     # future_states = tf.stack(
#     #     [decoded_example['state/future/x'], decoded_example['state/future/y']],
#     #     -1).numpy()
#     # future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0
#     future_states = decoded_example["track_infos"]["trajs"][:, 11:, :2]
#     future_states_mask = decoded_example["track_infos"]["trajs"][:, 11:, -1] > 0.0

#     # [num_points, 3] float32.
#     # roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()
#     roadgraph_xyz = decoded_example["map_infos"]["all_polylines"][:, :3]

#     num_agents, num_past_steps, _ = past_states.shape
#     num_future_steps = future_states.shape[1]

#     color_map = get_colormap(num_agents)

#     # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
#     all_states = np.concatenate([past_states, current_states, future_states], 1)

#     # [num_agens, num_past_steps + 1 + num_future_steps] float32.
#     all_states_mask = np.concatenate([past_states_mask, current_states_mask, future_states_mask], 1) > 0.0

#     center_y, center_x, width = get_viewport(all_states, all_states_mask)

#     images = []

#     # Generate images from past time steps.
#     for i, (s, m) in enumerate(
#         zip(
#             np.split(past_states, num_past_steps, 1),
#             np.split(past_states_mask, num_past_steps, 1),
#         )
#     ):
#         im = visualize_one_step(
#             s[:, 0],
#             m[:, 0],
#             roadgraph_xyz,
#             "past: %d" % (num_past_steps - i),
#             center_y,
#             center_x,
#             width,
#             color_map,
#             size_pixels,
#         )
#         images.append(im)

#     # Generate one image for the current time step.
#     s = current_states
#     m = current_states_mask

#     im = visualize_one_step(
#         s[:, 0],
#         m[:, 0],
#         roadgraph_xyz,
#         "current",
#         center_y,
#         center_x,
#         width,
#         color_map,
#         size_pixels,
#     )
#     images.append(im)

#     # Generate images from future time steps.
#     for i, (s, m) in enumerate(
#         zip(
#             np.split(future_states, num_future_steps, 1),
#             np.split(future_states_mask, num_future_steps, 1),
#         )
#     ):
#         im = visualize_one_step(
#             s[:, 0],
#             m[:, 0],
#             roadgraph_xyz,
#             "future: %d" % (i + 1),
#             center_y,
#             center_x,
#             width,
#             color_map,
#             size_pixels,
#         )
#         images.append(im)

#     return images
