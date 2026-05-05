"""Aviation-specific base visualizer."""

import time
from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
from pathlib import Path

import cv2
import imageio
import matplotlib as mpl
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

from characterization.domains.aviation.scenario_types import (
    AGENT_COLORS,
    MAP_COLORS,
    STRING_TO_AGENT_TYPE,
    AgentTrajectory,
    AgentType,
    MapType,
)
from characterization.domains.aviation.schemas.airport_metadata import ReferenceMetadata
from characterization.domains.aviation.schemas.critical_probe import CriticalProbe
from characterization.domains.aviation.schemas.scenario import MapData, Scenario
from characterization.domains.aviation.schemas.scenario_scores import AgentScore, ScenarioScores
from characterization.utils.common import XYZScale
from characterization.utils.constants import SCALE_FACTOR_TO_M
from characterization.utils.logging_utils import get_pylogger

_LOGGER = get_pylogger(__name__)


class SupportedPanes(Enum):
    """Panes supported for aviation scenario visualization.

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


class AviationBaseVisualizer(ABC):
    """Base visualizer for aviation scenarios.

    Provides aviation-specific rendering: raster/graph map display, aircraft icon overlays,
    lat/lon coordinate handling, counterfactual probe overlays, and score halo visualization.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes aviation-specific visualization state on top of the generic base.

        Args:
            config: Visualizer configuration.

        Raises:
            FileNotFoundError: If ``show_raster=True`` and the raster map PNG cannot be read.
        """
        super().__init__()

        panes_cfg = config.get("panes_to_plot", ["HIGHLIGHT_RELEVANT_AGENTS"])
        self.panes_to_plot: list[SupportedPanes] = [SupportedPanes[p] for p in panes_cfg]
        self.num_panes_to_plot: int = len(self.panes_to_plot)
        self.add_title: bool = config.get("add_title", False)
        self.fps: int = config.get("fps", 10)
        self.num_workers: int = config.get("num_workers", 1)
        self.time_scale_factor: float = config.get("time_scale_factor", 1.0)
        self.display_time: bool = config.get("display_time", True)
        self.min_valid_timesteps: int = config.get("min_valid_timesteps", 2)
        self.display_agent_ids: bool = config.get("display_agent_ids", True)
        self.score_color_map = cm.get_cmap(config.get("score_colormap", "Reds"))
        self.score_halo_size: int = config.get("score_halo_size", 80)
        self.show_score_colorbar: bool = config.get("show_score_colorbar", False)

        assets_path = Path(config.get("assets_path", "assets"))

        self.use_icons: bool = config.get("use_icons", False)
        if self.use_icons:
            self.agent_icons: dict[AgentType, NDArray[np.uint8]] = {
                AgentType.AIRCRAFT: imageio.imread(assets_path / "ac.png"),
                AgentType.VEHICLE: imageio.imread(assets_path / "vc.png"),
                AgentType.UNKNOWN: imageio.imread(assets_path / "uk_ac.png"),
            }
            agent_zoom_cfg = config.get("agent_zoom", {})
            self.agent_zoom: dict[AgentType, float] = {
                AgentType.AIRCRAFT: agent_zoom_cfg.get("AIRCRAFT", 0.010),
                AgentType.VEHICLE: agent_zoom_cfg.get("VEHICLE", 0.15),
                AgentType.UNKNOWN: agent_zoom_cfg.get("UNKNOWN", 0.010),
            }

        self.airport: str = config.airport
        self.use_latlon: bool = config.get("use_latlon", True)
        self.show_raster: bool = config.get("show_raster", False)
        self.xy_offset: float = config.get("xy_offset", 0.1)
        self.latlon_offset: float = config.get("latlon_offset", 0.02)

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
            graph_data_path = Path(config.get("graph_data_path", "graph_data"))
            map_filepath = graph_data_path / self.airport / f"{self.airport}.pkl"
            self.map_data = MapData.from_pickle(map_filepath)

        limits_filepath = assets_path / self.airport / "limits.json"
        self.ref_data = ReferenceMetadata.from_json_file(limits_filepath)
        self.ref_data.set_reference(config.espg)

        self.display_only_critical_actypes: bool = config.get("display_only_critical_actypes", False)

        # Probe visualization options
        self.probe_color: str = config.get("probe_color", "black")
        self.show_probe_metadata: bool = config.get("show_probe_metadata", False)
        self.show_criticality_label: bool = config.get("show_criticality_label", False)
        self.probe_halo_size: int = config.get("probe_halo_size", 80)
        self.probe_ring_size: int = config.get("probe_ring_size", 120)
        self.probe_criticality_marker_size: int = config.get("probe_criticality_marker_size", 50)
        self.show_agent_distance_lines: bool = config.get("show_agent_distance_lines", False)

    def plot_map_data(self, ax: Axes, num_windows: int = 1, alpha: float = 0.2) -> None:
        """Plots the aviation map (raster PNG or graph-based polylines).

        Args:
            ax: Axes to plot on.
            num_windows: Number of subplot windows.
            alpha: Transparency for map features.
        """
        if self.show_raster:
            self._plot_raster_map(ax, num_windows, alpha)
        else:
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
        self,
        scenario: Scenario,
        *,
        show_relevant: bool = False,
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
        """Repackages agent trajectory data into arrays suitable for plotting.

        Args:
            scenario: Scenario to visualize.
            show_relevant: If True, assigns difficulty-based scores to relevant agents.
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
            agent_scores = AviationBaseVisualizer.get_normalized_agent_scores(agent_scores, ego_index, amin=0.5)

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
        """Plots agent trajectories with optional relevance-based transparency.

        Args:
            ax: Axes to plot on.
            scenario: Scenario to visualize.
            show_relevant: Highlights relevant and SDC agents when True.
            start_timestep: First timestep to include.
            end_timestep: Last timestep to include (exclusive; -1 = all).
            title: Axes title.
        """
        zipped = zip(*self._repack_agent_sequences(scenario, show_relevant=show_relevant), strict=True)
        for i, (agent_x, agent_y, agent_heading, agent_mask, agent_type, _, agent_id, agent_score) in enumerate(zipped):
            atype = STRING_TO_AGENT_TYPE.get(agent_type, AgentType.UNKNOWN)
            mask = agent_mask[start_timestep:end_timestep]
            if not mask.any() or mask.sum() < self.min_valid_timesteps:
                continue

            x, y = agent_x[start_timestep:end_timestep][mask], agent_y[start_timestep:end_timestep][mask]
            heading = np.rad2deg(agent_heading[start_timestep:end_timestep][mask][-1].item())
            color = AGENT_COLORS.get(atype, AGENT_COLORS[AgentType.UNKNOWN])
            ax.plot(x, y, color=color, linewidth=1, alpha=agent_score)

            xf, yf = x[-1].item(), y[-1].item()
            aid = agent_id if self.display_agent_ids else None
            is_ego = i == scenario.metadata.ego_agent_index
            self.plot_agent(ax, xf, yf, heading, agent_type=atype, alpha=agent_score, agent_id=aid, add_halo=is_ego)

        if self.add_title:
            ax.set_title(title)

    def plot_sequences_with_scores(
        self,
        ax: Axes,
        scenario: Scenario,
        agent_scores: list[AgentScore],
        *,
        title: str = "",
    ) -> None:
        """Plots trajectories with score-colored halos and an optional colorbar.

        Args:
            ax: Axes to plot on.
            scenario: Scenario to visualize.
            agent_scores: Per-agent scores (e.g. individual or interaction scores).
            title: Axes title.
        """
        agent_id_to_score = {s.agent_id: s.score for s in agent_scores}
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
                    sm,
                    ax=ax,
                    orientation="horizontal",
                    location="bottom",
                    fraction=0.03,
                    pad=0.02,
                    aspect=30,
                )
                for spine in cb.ax.spines.values():
                    spine.set_visible(False)
                cb.ax.tick_params(labelsize=6, length=2)
                cb.set_label("Score", fontsize=6)

        if self.add_title:
            ax.set_title(title)

    def plot_sequences_with_probe(self, ax: Axes, scenario: Scenario) -> None:
        """Plots ground-truth trajectories with a counterfactual probe overlay.

        Raises:
            ValueError: If ``use_latlon=True`` (XYZ coordinates are required for probe rendering).
        """
        if self.use_latlon:
            error_message = "Probe visualization requires XYZ coordinates (use_latlon=False)."
            raise ValueError(error_message)

        assert scenario.critical_probe is not None
        probe: CriticalProbe = scenario.critical_probe
        affected_ids = set(probe.affected_agent_ids)
        current_time_index = scenario.metadata.current_time_index or 0

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

            if is_probed:
                future_mask = mask.copy()
                future_mask[: current_time_index + 1] = False
                ox, oy = agent_x[future_mask], agent_y[future_mask]
                if len(ox) >= self.min_valid_timesteps:
                    (orig_handle,) = ax.plot(
                        ox,
                        oy,
                        color=color,
                        linewidth=1.5,
                        linestyle=":",
                        zorder=140,
                        label="Original future",
                    )
                    legend_handles.append(orig_handle)

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

            halo_color = "#59A1F3" if is_ego else "#E05050"
            add_halo = is_ego or is_probed or is_affected
            aid = agent_id if self.display_agent_ids else None
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

        if self.add_title:
            probe_label = "ego" if probe.is_ego_agent else "other"
            ax.set_title(f"Probe: agent {probe.probed_agent_id} ({probe_label}, {probe.probe_type})")

    def _draw_criticality_marker(
        self,
        ax: Axes,
        probe: CriticalProbe,
        probe_traj: AgentTrajectory,
    ) -> object | None:
        """Draws a star marker on the counterfactual trajectory at each criticality timestamp."""
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
        """Annotates highlighted agents with their ID and max pair-score delta."""
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
        """Draws dashed distance lines from the probed agent to each affected agent."""
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
                [px, ax_pos],
                [py, ay_pos],
                color=self.probe_color,
                linewidth=0.8,
                linestyle="--",
                zorder=130,
                alpha=0.7,
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
        """Plots an agent icon (or fallback marker) with an optional halo.

        Args:
            ax: Axes to plot on.
            x: X coordinate.
            y: Y coordinate.
            heading: Heading in degrees.
            agent_type: Aviation agent type (determines icon / color).
            alpha: Transparency.
            agent_id: If set, renders the ID as a small text label.
            add_halo: Whether to add a background halo scatter marker.
            halo_color: Color for the halo.
            halo_size: Scatter size for the halo.
            marker_size: Scatter size for non-icon markers.
            zorder: Z-order for the marker.
            marker: Matplotlib marker string for non-icon rendering.
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
    def get_first_and_last_ego_position(
        scenario: Scenario,
        min_valid_timesteps: int = 2,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[None, None]:
        """Returns the first and last valid lat/lon positions of the ego aircraft.

        Args:
            scenario: Aviation scenario.
            min_valid_timesteps: Minimum number of valid timesteps required.

        Returns:
            ``(first_position, last_position)`` arrays, or ``(None, None)`` if the ego
            trajectory has fewer than ``min_valid_timesteps`` valid points.
        """
        ego_index = scenario.metadata.ego_agent_index
        agent_trajectories = AgentTrajectory(scenario.agent_data.agent_trajectories)
        agent_valid = agent_trajectories.valid.squeeze(-1).astype(bool)
        agent_positions = agent_trajectories.latlon

        ego_traj = agent_positions[ego_index]
        valid_mask = agent_valid[ego_index] & np.all(np.isfinite(ego_traj), axis=-1)
        valid_ego_traj = ego_traj[valid_mask]
        if valid_ego_traj.shape[0] < min_valid_timesteps:
            return None, None

        return valid_ego_traj[0], valid_ego_traj[-1]

    def configure_axes(
        self,
        ax: Axes,
        scenario: Scenario,
        num_windows: int = 1,
        spine_linewidth: float = 0.3,
        spine_color: str = "#cccccc",
    ) -> None:
        """Sets axis limits from the airport reference data and applies spine styling.

        Args:
            ax: Axes (or array of Axes) to configure.
            scenario: Scenario used to find ego position (for potential zoom).
            num_windows: Number of subplots sharing the configuration.
            spine_linewidth: Linewidth for axes spines.
            spine_color: Color for axes spines.
        """
        assert self.ref_data.reference_system is not None, "Reference system must be set before configuring axes."
        first_ego_position, last_ego_position = AviationBaseVisualizer.get_first_and_last_ego_position(
            scenario,
            self.min_valid_timesteps,
        )
        if first_ego_position is None or last_ego_position is None:
            return

        if self.use_latlon:
            x_min = self.ref_data.reference_system.west - self.latlon_offset
            x_max = self.ref_data.reference_system.east + self.latlon_offset
            y_min = self.ref_data.reference_system.south - self.latlon_offset
            y_max = self.ref_data.reference_system.north + self.latlon_offset
        else:
            x_min = self.ref_data.limits.y.min - self.xy_offset
            x_max = self.ref_data.limits.y.max + self.xy_offset
            y_min = self.ref_data.limits.x.min - self.xy_offset
            y_max = self.ref_data.limits.x.max + self.xy_offset

        if num_windows == 1:
            ax = np.asarray([ax])  # pyright: ignore[reportAssignmentType]

        for a in ax.reshape(-1):  # pyright: ignore[reportAttributeAccessIssue]
            a.set_xticks([])
            a.set_yticks([])
            a.set_xlim(x_min, x_max)
            a.set_ylim(y_min, y_max)
            a.set_autoscale_on(False)
            for spine in a.spines.values():
                spine.set_linewidth(spine_linewidth)
                spine.set_color(spine_color)

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
        _LOGGER.info("Saved GIF to %s [Time taken: %.2fs]", output_filepath, t_f - t_i)

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
        scores: ScenarioScores | None = None,
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Visualizes a scenario and saves the output to a file."""
