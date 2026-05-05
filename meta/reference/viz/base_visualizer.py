"""Base class for scenario visualizers."""

import time
from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
from pathlib import Path

import cv2
import imageio
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from natsort import natsorted
from numpy.typing import NDArray
from omegaconf import DictConfig
from PIL import Image
from scipy import ndimage

from safeair.schemas import CriticalScenarioMetadata, MapData, ModelOutput, ReferenceMetadata, Scenario
from safeair.schemas.critical_probe import CriticalProbe
from safeair.schemas.scenario_scores import AgentScore, ScenarioScores
from safeair.utils import SCALE_FACTOR_TO_M, XYZScale, get_pylogger
from safeair.utils.scenario_types import (
    AGENT_COLORS,
    MAP_COLORS,
    STRING_TO_AGENT_TYPE,
    AgentTrajectory,
    AgentType,
    MapType,
)

_LOGGER = get_pylogger(__name__)


class SupportedPanes(Enum):
    """Panes supported for visualization.

    Options are:
    - ALL_AGENTS: Displays all agents in the scenario.
    - HIGHLIGHT_RELEVANT_AGENTS: Highlights relevant and ego-agents agents.
    - CRITICAL_SCENARIO: Highlights critical scenario information
    - CRITICAL_SCENARIO_IDENTIFICATION_PREDICTION: Highlights predicted critical scenario information based on the
        corresponding model output.
    - SCORED_INDIVIDUAL: Visualizes per-agent individual (kinematic) scores as score-colored halos.
    - SCORED_INTERACTION: Visualizes per-agent interaction scores as score-colored halos.
    """

    ALL_AGENTS = 0
    HIGHLIGHT_RELEVANT_AGENTS = 1
    CRITICAL_SCENARIO = 2
    CRITICAL_SCENARIO_IDENTIFICATION_PREDICTION = 3
    SCORED_INDIVIDUAL = 4
    SCORED_INTERACTION = 5
    COUNTERFACTUAL_PROBE = 6


class BaseVisualizer(ABC):
    """Base class for scenario visualizers. Provides common functionality for visualizing scenarios. Subclasses should
    implement scenario-specific visualization logic.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseVisualizer with visualization configuration and validates required keys.

        This base class provides a flexible interface for scenario visualizers, supporting custom map and agent color
        schemes, transparency, and scenario type validation. Subclasses should implement scenario-specific visualization
        logic.

        Args:
            config: Visualizer configuration, including scenario type, map/agent keys, colors, and alpha values.

        Raises:
            AssertionError: If the scenario type or any required configuration key is missing or unsupported.
        """
        self.config = config

        assets_path = Path(config.get("assets_path", "assets"))

        self.use_icons = config.get("use_icons", False)
        if self.use_icons:
            self.agent_icons = {
                AgentType.AIRCRAFT: imageio.imread(assets_path / "ac.png"),
                AgentType.VEHICLE: imageio.imread(assets_path / "vc.png"),
                AgentType.UNKNOWN: imageio.imread(assets_path / "uk_ac.png"),
            }

            agent_zoom_cfg = config.get("agent_zoom", {})
            self.agent_zoom = {
                AgentType.AIRCRAFT: agent_zoom_cfg.get("AIRCRAFT", 0.010),
                AgentType.VEHICLE: agent_zoom_cfg.get("VEHICLE", 0.15),
                AgentType.UNKNOWN: agent_zoom_cfg.get("UNKNOWN", 0.010),
            }

        self.airport = config.airport

        self.use_latlon = config.get("use_latlon", True)
        self.show_raster = config.get("show_raster", False)
        self.xy_offset = config.get("xy_offset", 0.1)
        self.latlon_offset = config.get("latlon_offset", 0.02)
        if self.show_raster:
            raster_map_filepath = assets_path / self.airport / "bkg_map.png"
            raster_map = cv2.imread(str(raster_map_filepath))
            if raster_map is None:
                error_message = f"Could not read raster map from {raster_map_filepath}"
                _LOGGER.error(error_message)
                raise FileNotFoundError(error_message)
            raster_map = cv2.resize(raster_map, (raster_map.shape[1] // 2, raster_map.shape[0] // 2))
            self.raster_map = cv2.cvtColor(raster_map, cv2.COLOR_BGR2RGB)
        else:
            # TODO: load graph data and visualize
            graph_data_path = Path(config.get("graph_data_path", "graph_data"))
            map_filepath = graph_data_path / self.airport / f"{self.airport}.pkl"
            self.map_data = MapData.from_pickle(map_filepath)

        limits_filepath = assets_path / self.airport / "limits.json"
        self.ref_data = ReferenceMetadata.from_json_file(limits_filepath)
        self.ref_data.set_reference(config.espg)

        # Initialize the color map for categorical visualization
        colormap = config.get("colormap", "autumn_r")
        self.color_map = cm.get_cmap(colormap)

        # Score visualization config
        score_colormap = config.get("score_colormap", "YlOrRd")
        self.score_color_map = cm.get_cmap(score_colormap)
        self.score_halo_size = config.get("score_halo_size", 100)
        self.show_score_colorbar = config.get("show_score_colorbar", True)

        # Number of workers for processing animations in parallel
        self.num_workers = config.get("num_workers", 10)
        self.fps = config.get("fps", 1)
        self.time_scale_factor = config.get("time_scale_factor", 1.0)
        self.display_time = config.get("display_time", True)
        self.display_agent_ids = config.get("display_agent_ids", False)

        # Other visualization options
        self.min_valid_timesteps = config.get("min_valid_timesteps", 2)
        self.probe_color = config.get("probe_color", "black")
        self.add_title = config.get("add_title", True)
        self.update_limits = config.get("update_limits", False)
        self.buffer_distance = config.get("distance_to_ego_zoom_in", 50.0)  # in meters
        self.distance_to_ego_zoom_in = config.get("distance_to_ego_zoom_in", 1000.0)  # in meters
        self.display_only_critical_actypes = config.get("display_only_critical_actypes", False)

        # Probe visualization options
        self.show_probe_metadata = config.get("show_probe_metadata", False)
        self.show_criticality_label = config.get("show_criticality_label", False)
        self.probe_halo_size = config.get("probe_halo_size", 80)
        self.probe_ring_size = config.get("probe_ring_size", 120)
        self.probe_criticality_marker_size = config.get("probe_criticality_marker_size", 50)
        self.show_agent_distance_lines = config.get("show_agent_distance_lines", False)

    def plot_map_data(self, ax: Axes, num_windows: int = 1, alpha: float = 0.2) -> None:
        """Plots static and dynamic map features for a scenario.

        Args:
            ax: Axes to plot on.
            num_windows: Number of subplot windows. Defaults to 1.
            alpha: Transparency level for the map features. Defaults to 0.2.
        """
        if self.show_raster:
            # Plot raster map as background
            self._plot_raster_map(ax, num_windows, alpha)
        else:
            # Plot hold line information
            self._plot_graph_map(ax, num_windows)

    def _plot_raster_map(self, ax: Axes, num_windows: int = 1, alpha: float = 0.6) -> None:
        assert self.ref_data.reference_system is not None, "Reference system must be set before plotting map data."
        if num_windows == 1:
            ax = np.asarray([ax])  # pyright: ignore[reportAssignmentType]

        for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
            a.imshow(
                self.raster_map,
                extent=(
                    self.ref_data.reference_system.west,
                    self.ref_data.reference_system.east,
                    self.ref_data.reference_system.south,
                    self.ref_data.reference_system.north,
                ),
                zorder=0,
                alpha=alpha,
            )

    def _plot_graph_map(self, ax: Axes, num_windows: int = 1, alpha: float = 0.7) -> None:
        # Plot hold lines
        if num_windows == 1:
            ax = np.asarray([ax])  # pyright: ignore[reportAssignmentType]

        x_idx, y_idx = 1, 0
        if self.use_latlon:
            taxiways = self.map_data.taxiway_polylines_latlon
            runways = self.map_data.runway_polylines_latlon
            exits = self.map_data.exit_polylines_latlon
            ramps = self.map_data.ramp_polylines_latlon
            hold_lines = self.map_data.hold_short_points_latlon
        else:
            taxiways = self.map_data.taxiway_polylines_xy
            runways = self.map_data.runway_polylines_xy
            exits = self.map_data.exit_polylines_xy
            ramps = self.map_data.ramp_polylines_xy
            hold_lines = self.map_data.hold_short_points_xy

        polyline_layers = [
            (taxiways, MAP_COLORS[MapType.TAXIWAY], 1, "o"),
            (runways, MAP_COLORS[MapType.RUNWAY], 1, "o"),
            (exits, MAP_COLORS[MapType.EXIT], 1, "o"),
            (ramps, MAP_COLORS[MapType.RAMP], 1, "o"),
            (hold_lines, MAP_COLORS[MapType.HOLD_SHORT_LINE], 2, "s"),
        ]

        for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
            for points, color, size, marker in polyline_layers:
                a.scatter(
                    points[:, x_idx],
                    points[:, y_idx],
                    c=color,
                    s=size,
                    alpha=alpha,
                    zorder=1,
                    marker=marker,
                    edgecolors="none",
                )

    def _repack_agent_sequences(
        self, scenario: Scenario, *, show_relevant: bool = False
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[AgentType],
        NDArray[np.object_],
        NDArray[np.int32],
        NDArray[np.float32],
    ]:
        """Repackages agent trajectory data for plotting, with optional highlighting of relevant agents.

        Args:
            scenario: Scenario to visualize.
            show_relevant: If True, highlights relevant and SDC agents. Defaults to False.
        """
        agent_data = scenario.agent_data
        agent_ids = agent_data.agent_ids
        agent_types = agent_data.agent_types
        ac_types = agent_data.aircraft_types
        if ac_types is None:
            ac_types = np.array(["Unknown"] * agent_data.num_agents)
        ego_index = scenario.metadata.ego_agent_index

        agent_scores = np.ones(agent_data.num_agents, np.float32)
        if show_relevant:
            agents_to_predict = scenario.agents_to_predict
            for idx, difficulty in zip(agents_to_predict.agent_index, agents_to_predict.agent_difficulty, strict=False):
                agent_scores[idx] = difficulty
            agent_scores = BaseVisualizer.get_normalized_agent_scores(agent_scores, ego_index)

        # Zip information to plot
        agent_trajectories = AgentTrajectory(agent_data.agent_trajectories)
        agent_heading = agent_trajectories.heading
        agent_mask = agent_trajectories.valid.squeeze(-1).astype(bool)
        if self.use_latlon:
            agent_x_coords = agent_trajectories.longitude
            agent_y_coords = agent_trajectories.latitude
        else:
            agent_x_coords = agent_trajectories.xy_position[:, :, 1]
            agent_y_coords = agent_trajectories.xy_position[:, :, 0]

        return agent_x_coords, agent_y_coords, agent_heading, agent_mask, agent_types, ac_types, agent_ids, agent_scores

    def plot_sequences(
        self,
        ax: Axes,
        scenario: Scenario,
        *,
        show_relevant: bool = False,
        start_timestep: int = 0,
        end_timestep: int = -1,
        title: str = "",
    ) -> None:
        """Plots agent trajectories for a scenario, with optional highlighting and score-based transparency.

        Args:
            ax: Axes to plot on.
            scenario: Scenario to visualize.
            show_relevant: If True, highlights relevant and SDC agents. Defaults to False.
            start_timestep: Starting timestep to plot the sequences.
            end_timestep: Ending timestep to plot the sequences.
            title: Title for the plot. Defaults to "".
        """
        zipped = zip(*self._repack_agent_sequences(scenario, show_relevant=show_relevant), strict=True)

        for i, (agent_x, agent_y, agent_heading, agent_mask, agent_type, _, agent_id, agent_score) in enumerate(zipped):
            # Skip if there are less than 2 valid points
            atype = STRING_TO_AGENT_TYPE.get(agent_type, AgentType.UNKNOWN)

            mask = agent_mask[start_timestep:end_timestep]
            if not mask.any() or mask.sum() < self.min_valid_timesteps:
                continue

            # Plot the trajectory
            x, y = agent_x[start_timestep:end_timestep][mask], agent_y[start_timestep:end_timestep][mask]
            heading = np.rad2deg(agent_heading[start_timestep:end_timestep][mask][-1].item())
            color = AGENT_COLORS.get(atype, AGENT_COLORS[AgentType.UNKNOWN])
            ax.plot(x, y, color=color, linewidth=1, alpha=agent_score)

            # Plot the agent
            xf, yf = x[-1].item(), y[-1].item()
            aid = agent_id if self.display_agent_ids else None
            is_ego = i == scenario.metadata.ego_agent_index
            self.plot_agent(ax, xf, yf, heading, agent_type=atype, alpha=agent_score, agent_id=aid, add_halo=is_ego)

        if self.add_title:
            ax.set_title(title)

    def plot_sequences_with_critical_metadata(
        self,
        ax: Axes,
        scenario: Scenario,
        *,
        start_timestep: int = 0,
        end_timestep: int = -1,
    ) -> None:
        """Plots agent trajectories for a scenario with critical scenario highlighting.

        Args:
            ax: Axes to plot on.
            scenario: Scenario to visualize.
            start_timestep: Starting timestep to plot the sequences.
            end_timestep: Ending timestep to plot the sequences.
        """
        critical_metadata = scenario.critical_metadata
        assert critical_metadata is not None, (
            "Critical metadata must be present in the scenario to plot critical scenario information."
        )
        critical_agent_ids = set(critical_metadata.agent_ids or [])
        los_frame = critical_metadata.los_frame
        critical_aircraft_types = set(critical_metadata.aircraft_types)

        zipped = zip(*self._repack_agent_sequences(scenario), strict=True)

        for _, (agent_x, agent_y, agent_heading, agent_mask, agent_type, ac_type, agent_id, _) in enumerate(zipped):
            # Skip if there are less than 2 valid points
            atype = STRING_TO_AGENT_TYPE.get(agent_type, AgentType.UNKNOWN)

            mask = agent_mask[start_timestep:end_timestep]
            if not mask.any() or mask.sum() < self.min_valid_timesteps:
                continue

            if self.display_only_critical_actypes and ac_type not in critical_aircraft_types:
                # print(f"skipping non-critical actype {ac_type} not in {critical_aircraft_types}")
                continue

            # Plot the trajectory
            x, y = agent_x[start_timestep:end_timestep][mask], agent_y[start_timestep:end_timestep][mask]
            heading = np.rad2deg(agent_heading[start_timestep:end_timestep][mask][-1].item())
            color = AGENT_COLORS.get(atype, AGENT_COLORS[AgentType.UNKNOWN])
            is_critical_agent = agent_id in critical_agent_ids
            alpha = 1.0 if is_critical_agent else 0.8
            ax.plot(x, y, color=color, linewidth=1, alpha=alpha)

            # Display LoS frame if the agent is involved in the critical event
            if is_critical_agent:
                coll_x = agent_x[los_frame].item()
                coll_y = agent_y[los_frame].item()
                ax.scatter(coll_x, coll_y, color="black", marker="*", linewidths=1.5, alpha=1.0, s=10, zorder=180)
                label = f"Incident frame (t={los_frame})"
                los_handle = ax.scatter([], [], color="black", marker="*", linewidths=1.5, s=20, label=label)
                ax.legend(handles=[los_handle], loc="lower left", fontsize=6, framealpha=0.1)

            # Plot the agent
            self.plot_agent(
                ax=ax,
                x=x[-1].item(),
                y=y[-1].item(),
                heading=heading,
                agent_type=atype,
                alpha=alpha,
                agent_id=f"{agent_id} ({ac_type})" if self.display_agent_ids else None,
                add_halo=is_critical_agent,
                halo_color="#F7A362",
            )

        self.add_critical_metadata_annotation(ax, critical_metadata)

        if self.add_title:
            title = f"Event: {critical_metadata.event_id}"
            ax.set_title(title)

    def plot_sequences_with_critical_scenario_prediction(
        self,
        ax: Axes,
        scenario: Scenario,
        *,
        model_output: ModelOutput | None = None,
        start_timestep: int = 0,
        end_timestep: int = -1,
    ) -> None:
        """Plots agent trajectories for a scenario with critical scenario highlighting.

        Args:
            ax: Axes to plot on.
            scenario: Scenario to visualize.
            model_output: Model output for the scenario.
            start_timestep: Starting timestep to plot the sequences.
            end_timestep: Ending timestep to plot the sequences.
        """
        assert model_output is not None, "Model output must be provided to plot critical scenario predictions."
        assert model_output.critical_scenario_identification_output is not None, (
            "Critical scenario identification output must be present in the model output."
        )

        csi_output = model_output.critical_scenario_identification_output

        # Assumes binary classification with [non-critical prob, critical prob] for each agent
        agent_ids_pred = model_output.agent_ids.value.numpy()
        csi_agent_probs = csi_output.critical_agents_pred_probabilities.value.numpy()[:, 1]
        critical_agent_ids_pred = {
            int(agent_id): prob for agent_id, prob in zip(agent_ids_pred, csi_agent_probs, strict=False)
        }

        # Get LoS prediction information
        los_probabilities = csi_output.los_timestep_probabilities.value.numpy()
        los_frame = csi_output.los_timestep_index.value.item()
        los_frame_pred = los_probabilities.argmax()
        los_frame_prob = los_probabilities[los_frame_pred]
        _, hist_len, _ = model_output.history_ground_truth.value.numpy().shape
        los_frame_pred += hist_len  # Adjust for history length to get absolute frame index
        los_frame_error = los_frame - los_frame_pred

        agent_sequences = list(zip(*self._repack_agent_sequences(scenario), strict=True))

        # Find the ego's agent_id to exclude it when selecting the top critical agent
        ego_agent_id = agent_sequences[scenario.metadata.ego_agent_index][6]
        non_ego_probs = {aid: prob for aid, prob in critical_agent_ids_pred.items() if aid != ego_agent_id}
        top_critical_agent_id = max(non_ego_probs, key=lambda aid: non_ego_probs[aid]) if non_ego_probs else None

        for i, (agent_x, agent_y, agent_heading, agent_mask, agent_type, ac_type, agent_id, _) in enumerate(
            agent_sequences
        ):
            # Skip if there are less than 2 valid points
            atype = STRING_TO_AGENT_TYPE.get(agent_type, AgentType.UNKNOWN)

            mask = agent_mask[start_timestep:end_timestep]
            if not mask.any() or mask.sum() < self.min_valid_timesteps:
                continue

            # Plot the trajectory
            x, y = agent_x[start_timestep:end_timestep][mask], agent_y[start_timestep:end_timestep][mask]
            heading = np.rad2deg(agent_heading[start_timestep:end_timestep][mask][-1].item())
            color = AGENT_COLORS.get(atype, AGENT_COLORS[AgentType.UNKNOWN])

            alpha = critical_agent_ids_pred.get(agent_id, 0.0)  # Default to 0.0 if agent ID is not in predictions
            alpha = min(alpha + 0.4, 1.0)  # Add a base alpha to ensure visibility of non-critical agents, capped at 1.0
            ax.plot(x, y, color=color, linewidth=1, alpha=alpha)

            # Display LoS frame if the agent is involved in the critical event
            is_ego = i == scenario.metadata.ego_agent_index
            if is_ego:
                coll_x = agent_x[los_frame_pred].item()
                coll_y = agent_y[los_frame_pred].item()
                ax.scatter(coll_x, coll_y, color="black", marker="*", linewidths=1.5, alpha=1.0, s=10, zorder=180)
                timing = "late" if los_frame_error < 0 else "early"
                label = (
                    f"Incident (t={los_frame_pred}, p={los_frame_prob:.2f}, "
                    f"error={los_frame_error} timesteps ({timing}))"
                )
                los_handle = ax.scatter([], [], color="black", marker="*", linewidths=1.5, s=20, label=label)
                ax.legend(handles=[los_handle], loc="lower left", fontsize=6, framealpha=0.1)

            is_top_critical = agent_id == top_critical_agent_id
            add_halo = is_ego or is_top_critical
            halo_color = "#5D75EC" if is_ego else "#EC5D5D"

            # Plot the agent
            self.plot_agent(
                ax=ax,
                x=x[-1].item(),
                y=y[-1].item(),
                heading=heading,
                agent_type=atype,
                alpha=alpha,
                agent_id=f"{agent_id} ({ac_type})" if self.display_agent_ids else None,
                add_halo=add_halo,
                halo_color=halo_color,
            )

        if self.add_title:
            title = "Critical Scenario Identification"
            ax.set_title(title)

    def plot_sequences_with_scores(
        self,
        ax: Axes,
        scenario: Scenario,
        agent_scores: list[AgentScore],
        *,
        title: str = "",
    ) -> None:
        """Plots agent trajectories with score-colored halos indicating per-agent scores.

        Trajectories are drawn at full opacity with their standard agent-type color. Each agent receives a halo
        whose color is drawn from ``self.score_color_map``, mapped from the normalized score. A colorbar is
        added when ``self.show_score_colorbar`` is True.

        Args:
            ax: Axes to plot on.
            scenario: Scenario to visualize.
            agent_scores: Per-agent scores (e.g. ``ScenarioScores.individual_scores`` or
                ``ScenarioScores.interaction_scores``).
            title: Title for the plot.
        """
        agent_id_to_score = {s.agent_id: s.score for s in agent_scores}

        # Build a single ScalarMappable that drives both halo colors and the colorbar — ensures they are identical
        scored_values = [s.score for s in agent_scores]
        score_min = min(scored_values) if scored_values else 0.0
        score_max = max(scored_values) if scored_values else 1.0
        norm = mcolors.Normalize(vmin=score_min, vmax=score_max)
        sm = cm.ScalarMappable(cmap=self.score_color_map, norm=norm)
        sm.set_array([])

        ego_index = scenario.metadata.ego_agent_index

        zipped = zip(*self._repack_agent_sequences(scenario), strict=True)
        for i, (agent_x, agent_y, agent_heading, agent_mask, agent_type, _, agent_id, _score) in enumerate(zipped):
            atype = STRING_TO_AGENT_TYPE.get(agent_type, AgentType.UNKNOWN)
            mask = agent_mask.astype(bool)
            if not mask.any() or mask.sum() < self.min_valid_timesteps:
                continue

            x, y = agent_x[mask], agent_y[mask]
            heading = np.rad2deg(agent_heading[mask][-1].item())
            color = AGENT_COLORS.get(atype, AGENT_COLORS[AgentType.UNKNOWN])
            ax.plot(x, y, color=color, linewidth=1, alpha=1.0)

            aid = agent_id if self.display_agent_ids else None
            is_ego = i == ego_index
            has_score = int(agent_id) in agent_id_to_score

            if is_ego:
                # Keep ego identifiable with its standard blue halo rather than a score color
                self.plot_agent(
                    ax,
                    x[-1].item(),
                    y[-1].item(),
                    heading,
                    agent_type=atype,
                    alpha=1.0,
                    agent_id=aid,
                    add_halo=True,
                    halo_color="#59A1F3",
                    halo_size=self.score_halo_size,
                )
            elif has_score:
                raw_score = float(agent_id_to_score[int(agent_id)])
                halo_color: tuple[float, float, float, float] = self.score_color_map(float(norm(raw_score)))
                self.plot_agent(
                    ax,
                    x[-1].item(),
                    y[-1].item(),
                    heading,
                    agent_type=atype,
                    alpha=1.0,
                    agent_id=aid,
                    add_halo=True,
                    halo_color=halo_color,
                    halo_size=self.score_halo_size,
                )
            else:
                self.plot_agent(ax, x[-1].item(), y[-1].item(), heading, agent_type=atype, alpha=1.0, agent_id=aid)

        if self.show_score_colorbar:
            fig = ax.get_figure()
            if fig is not None:
                cb = fig.colorbar(
                    sm, ax=ax, orientation="horizontal", location="bottom", fraction=0.03, pad=0.02, aspect=30
                )
                for spine in cb.ax.spines.values():
                    spine.set_visible(False)
                cb.ax.tick_params(labelsize=6, length=2)
                cb.set_label("Score", fontsize=6)

        if self.add_title:
            ax.set_title(title)

    def plot_sequences_with_probe(
        self,
        ax: Axes,
        scenario: Scenario,
    ) -> None:
        """Plot agent trajectories with a counterfactual probe overlay.

        Draws ground-truth trajectories in standard colors, overlays the probed agent's original future
        as a dotted reference line and the counterfactual future as a dashed orange line, marks affected
        agents with halos, and annotates the criticality timestamp with a star marker.
        Raises ``ValueError`` if ``use_latlon`` is True (XYZ coordinates are required).
        Callers must ensure ``scenario.critical_probe`` is not None.

        Args:
            ax: Axes to plot on.
            scenario: Scenario to visualize. Must have ``critical_probe`` populated.
        """
        if self.use_latlon:
            error_message = "Probe visualization requires XYZ coordinates (use_latlon=False)."
            raise ValueError(error_message)

        assert scenario.critical_probe is not None
        probe: CriticalProbe = scenario.critical_probe
        affected_ids = set(probe.affected_agent_ids)
        current_time_index = scenario.metadata.current_time_index or 0

        # Draw ground-truth trajectories for all agents; track icon positions for highlighted agents
        legend_handles: list[object] = []
        agent_icon_positions: dict[int, tuple[float, float]] = {}
        zipped = zip(*self._repack_agent_sequences(scenario), strict=True)
        for _i, (agent_x, agent_y, agent_heading, agent_mask, agent_type, _, agent_id, _score) in enumerate(zipped):
            atype = STRING_TO_AGENT_TYPE.get(agent_type, AgentType.UNKNOWN)
            mask = agent_mask.astype(bool)
            if not mask.any() or mask.sum() < self.min_valid_timesteps:
                continue

            x, y = agent_x[mask], agent_y[mask]
            color = AGENT_COLORS.get(atype, AGENT_COLORS[AgentType.UNKNOWN])
            is_ego = _i == scenario.metadata.ego_agent_index
            is_probed = agent_id == probe.probed_agent_id
            is_affected = int(agent_id) in affected_ids
            alpha = 1.0 if (is_ego or is_probed or is_affected) else 0.1
            ax.plot(x, y, color=color, linewidth=1, alpha=alpha)

            # Show original future of the probed agent as a dotted reference line
            if is_probed:
                future_mask = mask.copy()
                future_mask[: current_time_index + 1] = False
                ox, oy = agent_x[future_mask], agent_y[future_mask]
                if len(ox) >= self.min_valid_timesteps:
                    (orig_handle,) = ax.plot(
                        ox, oy, color=color, linewidth=1.5, linestyle=":", zorder=140, label="Original future"
                    )
                    legend_handles.append(orig_handle)

            # Use the agent's state at current_time_index for icon placement (last valid up to that point)
            valid_up_to_current = mask.copy()
            valid_up_to_current[current_time_index + 1 :] = False
            if not valid_up_to_current.any():
                continue
            icon_idx = int(np.where(valid_up_to_current)[0][-1])
            icon_x = agent_x[icon_idx].item()
            icon_y = agent_y[icon_idx].item()
            icon_heading = np.rad2deg(agent_heading[icon_idx].item())

            if is_ego or is_probed or is_affected:
                agent_icon_positions[int(agent_id)] = (icon_x, icon_y)

            # Ego keeps its blue halo for consistency with other panes; all other highlighted agents get red
            halo_color = "#59A1F3" if is_ego else "#E05050"
            add_halo = is_ego or is_probed or is_affected
            aid = agent_id if self.display_agent_ids else None
            # zorder=170 ensures icons and halos render above the counterfactual trajectory (zorder=150)
            self.plot_agent(
                ax,
                icon_x,
                icon_y,
                icon_heading,
                agent_type=atype,
                alpha=alpha,
                agent_id=aid,
                add_halo=add_halo,
                halo_color=halo_color,
                halo_size=self.probe_halo_size,
                zorder=170,
            )

        # Draw a hollow ring around the probed agent's position at current_time_index to distinguish it
        probed_pos = agent_icon_positions.get(probe.probed_agent_id)
        if probed_pos is not None:
            ax.scatter(
                probed_pos[0],
                probed_pos[1],
                facecolors="none",
                edgecolors=self.probe_color,
                linewidths=1.5,
                s=self.probe_ring_size,
                zorder=160,
            )

        # Overlay the counterfactual future trajectory of the probed agent
        probe_traj = AgentTrajectory(probe.probed_agent_trajectory)
        future_slice = slice(current_time_index + 1, None)
        future_valid = probe_traj.valid.squeeze(-1)[future_slice].astype(bool)
        cf_x = probe_traj.xy_position[:, 1][future_slice][future_valid]
        cf_y = probe_traj.xy_position[:, 0][future_slice][future_valid]

        if len(cf_x) >= self.min_valid_timesteps:
            (cf_handle,) = ax.plot(
                cf_x,
                cf_y,
                color=self.probe_color,
                linewidth=1.0,
                linestyle="--",
                zorder=150,
                label=f"Counterfactual ({probe.probe_type})",
            )
            legend_handles.append(cf_handle)

        legend_handles.extend(h for h in [self._draw_criticality_marker(ax, probe, probe_traj)] if h is not None)

        if self.show_probe_metadata:
            self._annotate_probe_metadata(ax, probe, agent_icon_positions)

        if self.show_agent_distance_lines:
            self._draw_agent_distance_lines(ax, probe, agent_icon_positions, scenario.metadata.xyz_scale)

        # if legend_handles:
        #     ax.legend(
        #         handles=legend_handles,
        #         loc="lower left",
        #         bbox_to_anchor=(0, 1.02),
        #         fontsize=6,
        #         framealpha=0.1,
        #         borderaxespad=0,
        #     )

        if self.add_title:
            probe_label = "ego" if probe.is_ego_agent else "other"
            ax.set_title(f"Probe: agent {probe.probed_agent_id} ({probe_label}, {probe.probe_type})")

    @staticmethod
    def add_critical_metadata_annotation(ax: Axes, critical_metadata: CriticalScenarioMetadata) -> None:
        """Builds compact text lines from critical scenario metadata.

        Args:
            ax: Axes to add the annotation to.
            critical_metadata: Metadata associated with the critical scenario.
        """
        critical_agent_ids = critical_metadata.agent_ids or []
        critical_agents_str = ", ".join(str(agent_id) for agent_id in critical_agent_ids) or "N/A"
        aircraft_types = critical_metadata.aircraft_types
        aircraft_types_str = " vs ".join(aircraft_types) if aircraft_types else "N/A"
        metadata_list = [
            f"Event: {critical_metadata.event_id}",
            f"   Where: {critical_metadata.event_airport_id}, {critical_metadata.runway_description}",
            f"   When: {critical_metadata.event_local_date}, {critical_metadata.time}",
            f"   Type: {critical_metadata.incident_type_faa_code}",
            f"   Risk: {critical_metadata.runway_safety_risk_code}",
            "Aircraft:",
            f"   Type: {aircraft_types_str}",
            f"   ID: {critical_agents_str}",
        ]

        metadata_text = "\n".join(metadata_list)
        ax.annotate(
            metadata_text,
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=5,
            ha="left",
            va="top",
            linespacing=1.5,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cccccc", "alpha": 0.2},
            zorder=200,
        )

    def _draw_criticality_marker(
        self,
        ax: Axes,
        probe: CriticalProbe,
        probe_traj: AgentTrajectory,
    ) -> object | None:
        """Draw a star marker at each affected agent's criticality timestamp and optionally label it.

        Iterates ``probe.criticality_results`` (one entry per affected agent). For each entry,
        draws a star on the probed agent's counterfactual trajectory at the given frame index.

        Args:
            ax: Axes to draw on.
            probe: The critical probe result.
            probe_traj: Trajectory accessor for the probed agent's counterfactual trajectory.

        Returns:
            A dummy scatter handle for the first valid marker (for legend insertion), or ``None``
            if ``criticality_results`` is empty or no timestamps are valid.
        """
        if not probe.criticality_results:
            return None

        first_handle: object | None = None
        multiple = len(probe.criticality_results) > 1

        for aid_str, crit in probe.criticality_results.items():
            ts = crit.timestamp
            if not bool(probe_traj.valid.squeeze(-1)[ts]):
                continue
            cx = float(probe_traj.xy_position[ts, 1])
            cy = float(probe_traj.xy_position[ts, 0])
            ax.scatter(
                cx,
                cy,
                color=self.probe_color,
                marker="*",
                s=self.probe_criticality_marker_size,
                zorder=200,
                linewidths=1.0,
            )
            if self.show_criticality_label:
                label_text = (
                    f"id={aid_str}\nt={ts}\n{crit.metric.value}" if multiple else f"t={ts}\n{crit.metric.value}"
                )
                ax.annotate(
                    label_text,
                    (cx, cy),
                    fontsize=5,
                    ha="center",
                    va="bottom",
                    xytext=(0, 4),
                    textcoords="offset points",
                    zorder=201,
                )
            if first_handle is None:
                first_handle = ax.scatter(
                    [],
                    [],
                    color=self.probe_color,
                    marker="*",
                    s=self.probe_criticality_marker_size // 2,
                    label=f"Criticality (t={ts})",
                )

        return first_handle

    def _annotate_probe_metadata(
        self,
        ax: Axes,
        probe: CriticalProbe,
        agent_icon_positions: dict[int, tuple[float, float]],
    ) -> None:
        """Annotate highlighted agents with their ID and max pair-score delta.

        Args:
            ax: Axes to annotate.
            probe: The critical probe result containing pair scores.
            agent_icon_positions: Map from agent ID to ``(x, y)`` icon position at ``current_time_index``.
        """
        if not probe.affected_pair_scores_after:
            return
        agent_deltas: dict[int, float] = {}
        for key, score_after in probe.affected_pair_scores_after.items():
            id_a, id_b = (int(v) for v in key.split(":"))
            delta = score_after - probe.affected_pair_scores_before.get(key, 0.0)
            for aid in (id_a, id_b):
                agent_deltas[aid] = max(agent_deltas.get(aid, 0.0), delta)
        for aid, delta in agent_deltas.items():
            if aid not in agent_icon_positions:
                continue
            px, py = agent_icon_positions[aid]
            ax.annotate(
                f"id={aid}\nΔ{delta:.2f}",
                (px, py),
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=4,
                ha="center",
                va="bottom",
                zorder=175,
            )

    def _draw_agent_distance_lines(
        self,
        ax: Axes,
        probe: CriticalProbe,
        agent_icon_positions: dict[int, tuple[float, float]],
        xyz_scale: XYZScale,
    ) -> None:
        """Draw dashed lines from the probed agent to each affected agent, labeled with Euclidean distance in meters.

        Args:
            ax: Axes to draw on.
            probe: The critical probe result.
            agent_icon_positions: Map from agent ID to ``(x, y)`` icon position at ``current_time_index``.
            xyz_scale: Coordinate unit of the scenario's XY positions, used to convert raw distances to meters.
        """
        probed_pos = agent_icon_positions.get(probe.probed_agent_id)
        if probed_pos is None:
            return
        px, py = probed_pos
        scale_to_m = SCALE_FACTOR_TO_M[xyz_scale]
        for aid in probe.affected_agent_ids:
            aff_pos = agent_icon_positions.get(aid)
            if aff_pos is None:
                continue
            ax_pos, ay_pos = aff_pos
            ax.plot(
                [px, ax_pos], [py, ay_pos], color=self.probe_color, linewidth=0.8, linestyle="--", zorder=130, alpha=0.7
            )
            dist_m = float(np.sqrt((ax_pos - px) ** 2 + (ay_pos - py) ** 2)) * scale_to_m
            mx, my = (px + ax_pos) / 2, (py + ay_pos) / 2
            ax.annotate(
                f"{dist_m:.1f} m",
                (mx, my),
                fontsize=4,
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                zorder=175,
            )

    def plot_agent(
        self,
        ax: Axes,
        x: float,
        y: float,
        heading: float,
        agent_type: AgentType,
        alpha: float,
        *,
        agent_id: int | str | None = None,
        add_halo: bool = False,
        halo_color: str | tuple[float, float, float, float] = "#59A1F3",
        halo_size: int = 130,
        marker_size: int = 8,
        zorder: int = 100,
        marker: str = "o",
    ) -> None:
        """Plots an agent icon or marker on the given Axes.

        Args:
            ax: Axes to plot on.
            x: X coordinate of the agent.
            y: Y coordinate of the agent.
            heading: Heading of the agent in degrees.
            agent_type: Type of the agent.
            alpha: Transparency level for the agent icon/marker.
            agent_id: ID of the agent to display next to the marker. Defaults to None.
            add_halo: Whether to add a halo around the agent. Defaults to False.
            halo_color: Color for the halo. Defaults to "#59A1F3".
            halo_size: Size for the halo. Defaults to 130.
            marker_size: Size for non-icon markers. Defaults to 8.
            zorder: Z-order for layering the marker. Defaults to 100.
            marker: Marker style for non-icon markers. Defaults to "o"
        """
        if add_halo:
            ax.scatter(x, y, color=halo_color, alpha=alpha, s=halo_size, zorder=zorder - 1)

        if self.use_icons:
            icon = self.agent_icons.get(agent_type, self.agent_icons[AgentType.UNKNOWN])
            zoom = self.agent_zoom.get(agent_type, self.agent_zoom[AgentType.UNKNOWN])
            img = ndimage.rotate(icon, heading)
            img = np.fliplr(img)
            img = OffsetImage(img, zoom=zoom, alpha=alpha)
            ab = AnnotationBbox(img, (x, y), frameon=False, zorder=zorder)
            ax.add_artist(ab)
        else:
            color = AGENT_COLORS.get(agent_type, AGENT_COLORS[AgentType.UNKNOWN])
            ax.scatter(x, y, s=marker_size, zorder=zorder, c=color, marker=marker, alpha=alpha)

        if agent_id is not None:
            offset = 0.002 if self.use_latlon else 0.1
            ax.text(x, y - offset, str(agent_id), fontsize=4, ha="center", va="center", zorder=zorder + 1)

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
            tmp_dir_frames: Temporary directory where scenario image frames have been saved.
            output_filepath: Output filepath to save the GIF.
            fps: Frames per second for the GIF.
            disposal: How the previous frame should be treated before displaying the next frame.
                Defaults to 2 (restores background color, clears the previous frame).
            loop: Number of times the GIF should loop.
        """
        t_i = time.time()
        # Load all the temporary files
        files = natsorted(glob(f"{tmp_dir_frames}/temp_*.png"))
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
        _LOGGER.info("Saved GIF to %s (%.2fs)", output_filepath, t_f - t_i)

    @staticmethod
    def get_normalized_agent_scores(
        agent_scores: NDArray[np.floating], ego_index: int, amin: float = 0.5, amax: float = 1.0
    ) -> NDArray[np.float32]:
        """Gets the agent scores and returns a normalized score array.

        Args:
            agent_scores: Array containing the agent scores.
            ego_index: Index of the ego vehicle.
            amin: Minimum value to clip the array.
            amax: Maximum value to clip the array.

        Returns:
            Normalized agent scores.
        """
        min_score = np.nanmin(agent_scores)
        max_score = np.nanmax(agent_scores)
        if max_score > min_score:
            agent_scores = (agent_scores - min_score) / (max_score - min_score)
            # Clip scores to avoid zero alpha values
            agent_scores = np.clip(agent_scores, a_min=amin, a_max=amax)
        else:
            # All scores are identical — assign minimum alpha to all agents
            agent_scores = np.full_like(agent_scores, amin, dtype=np.float32)

        # Set ego-agent to maximum alpha
        agent_scores[ego_index] = amax
        return agent_scores.astype(np.float32)

    @staticmethod
    def get_first_and_last_ego_position(
        scenario: Scenario, min_valid_timesteps: int = 2
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[None, None]:
        """Gets the first and last valid positions of the ego vehicle.

        Args:
            scenario: Scenario to visualize.
            min_valid_timesteps: Minimum number of valid timesteps required.

        Returns:
            First and last valid ego positions, or (None, None) if insufficient valid timesteps.
        """
        ego_index = scenario.metadata.ego_agent_index
        agent_trajectories = AgentTrajectory(scenario.agent_data.agent_trajectories)
        agent_valid = agent_trajectories.valid.squeeze(-1).astype(bool)
        agent_positions = agent_trajectories.latlon

        # Get first valid ego position
        ego_traj = agent_positions[ego_index]

        valid_mask = agent_valid[ego_index] & np.all(np.isfinite(ego_traj), axis=-1)
        valid_ego_traj = ego_traj[valid_mask]
        if valid_ego_traj.shape[0] < min_valid_timesteps:
            return None, None

        # Return first and last valid positions
        return valid_ego_traj[0], valid_ego_traj[-1]

    def configure_axes(
        self,
        ax: Axes,
        scenario: Scenario,
        num_windows: int = 1,
        spine_linewidth: float = 0.3,
        spine_color: str = "#cccccc",
    ) -> None:
        """Configures axes for scenario visualization.

        Args:
            ax: Axes to configure.
            scenario: Scenario to visualize.
            num_windows: Number of subplot windows. Defaults to 1.
            spine_linewidth: Line width for the axes spines. Defaults to 0.3.
            spine_color: Color for the axes spines. Defaults to '#cccccc'.
        """
        assert self.ref_data.reference_system is not None, "Reference system must be set before configuring axes."
        first_ego_position, last_ego_position = BaseVisualizer.get_first_and_last_ego_position(
            scenario, self.min_valid_timesteps
        )
        if first_ego_position is None or last_ego_position is None:
            return

        if self.use_latlon:
            x_min = self.ref_data.reference_system.west - self.latlon_offset
            x_max = self.ref_data.reference_system.east + self.latlon_offset
            y_min = self.ref_data.reference_system.south - self.latlon_offset
            y_max = self.ref_data.reference_system.north + self.latlon_offset
        else:
            # NOTE: they are flipped
            x_min = self.ref_data.limits.y.min - self.xy_offset
            x_max = self.ref_data.limits.y.max + self.xy_offset
            y_min = self.ref_data.limits.x.min - self.xy_offset
            y_max = self.ref_data.limits.x.max + self.xy_offset

        # ego_displacement = np.linalg.norm(first_ego_position - last_ego_position, axis=-1)
        # distance = max(self.distance_to_ego_zoom_in, ego_displacement) + self.buffer_distance

        # Plot hold lines
        if num_windows == 1:
            ax = np.asarray([ax])  # pyright: ignore[reportAssignmentType]

        for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
            a.set_xticks([])
            a.set_yticks([])
            a.set_xlim(x_min, x_max)
            a.set_ylim(y_min, y_max)
            a.set_autoscale_on(False)
            # if self.update_limits:
            #     ax.set_xlim(first_ego_position[0] - distance, first_ego_position[0] + distance)
            #     ax.set_ylim(first_ego_position[1] - distance, first_ego_position[1] + distance)

            for spine in a.spines.values():
                spine.set_linewidth(spine_linewidth)
                spine.set_color(spine_color)

    @abstractmethod
    def visualize_scenario(
        self,
        scenario: Scenario,
        model_output: ModelOutput | None,
        scores: ScenarioScores | None = None,
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Visualizes a single scenario and saves the output to a file.

        This method should be implemented by subclasses to provide scenario-specific visualization,
        supporting flexible titles and output paths. It is designed to handle both static and dynamic map
        features, as well as agent trajectories and attributes.

        Args:
            scenario: Scenario to visualize.
            model_output: Model output for the scenario.
            scores: Optional scenario scores for score-based panes.
            output_dir: Directory where the visualization will be saved.

        Returns:
            Path to the saved visualization file.
        """
