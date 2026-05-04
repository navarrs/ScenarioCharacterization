"""Helper utilities for scenario characterization — printing, plotting, and summary generation."""

from __future__ import annotations

import copy
import io
import math
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import seaborn as sns

from characterization.domains.aviation.schemas.scenario import MapData, Scenario
from characterization.domains.aviation.schemas.scenario_features import (
    IndividualAgentFeatures,
    InteractionPairFeatures,
    ScenarioFeatures,
)
from characterization.domains.aviation.schemas.scenario_scores import ScenarioScores
from characterization.utils.logging_utils import get_pylogger

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig

    from characterization.domains.aviation.utils.scenario_visualizer.scenario_visualizer import ScenarioVisualizer

_LOGGER = get_pylogger(__name__, use_rank_zero_only=False)

# ──────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ──────────────────────────────────────────────────────────────────────────────

_IND_COLS = ("agent_id", "agent_type", "speed", "accel", "decel", "waiting", "traj_type", "kalman_diff")
_INT_COLS = (
    "agent_id_a",
    "agent_id_b",
    "pair_type",
    "loss_of_separation",
    "mttcp",
    "thw",
    "ttc",
    "drac",
)
_SCORE_COLS = ("agent_id", "individual_score", "interaction_score")
_SCENE_SCORE_COLS = ("individual_scene", "interaction_scene", "combined")

_FLOAT_FMT = "{:.4f}"
_NONE_STR = "—"


def _fmt(value: object) -> str:
    if value is None:
        return _NONE_STR
    if isinstance(value, float):
        return _FLOAT_FMT.format(value)
    return str(value)


def _col_widths(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[int]:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths


def _print_table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = _col_widths(header, rows)
    sep = "  ".join("-" * w for w in widths)

    def _fmt_row(cols: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=False))

    print(_fmt_row(header))  # noqa: T201
    print(sep)  # noqa: T201
    for row in rows:
        print(_fmt_row(row))  # noqa: T201


def print_scenario_info(scenario: Scenario, pkl_path: Path) -> None:
    """Print a formatted header with scenario metadata."""
    m = scenario.metadata
    airport = scenario.static_map_data.airport_id if scenario.static_map_data else pkl_path.parent.name
    ts = m.timestamps_seconds
    duration = f"{float(ts[-1] - ts[0]):.1f}s" if len(ts) > 1 else "—"
    print(f"\n{'═' * 70}")  # noqa: T201
    print(f"  Scenario  : {m.scenario_id}")  # noqa: T201
    print(f"  Airport   : {airport}")  # noqa: T201
    print(f"  Dataset   : {m.dataset}")  # noqa: T201
    print(f"  Agents    : {scenario.agent_data.num_agents}")  # noqa: T201
    print(f"  Duration  : {duration}  ({len(m.timestamps_seconds)} frames @ {m.frequency_hz} Hz)")  # noqa: T201
    print(f"  Ego agent : id={m.ego_agent_id}  index={m.ego_agent_index}  strategy={m.ego_selection_strategy}")  # noqa: T201
    print(f"{'═' * 70}")  # noqa: T201


def _ind_row(f: IndividualAgentFeatures) -> tuple[str, ...]:
    return (
        str(f.agent_id),
        f.agent_type,
        _fmt(f.speed),
        _fmt(f.acceleration),
        _fmt(f.deceleration),
        _fmt(f.waiting_period),
        _fmt(f.trajectory_type),
        _fmt(f.kalman_difficulty),
    )


def _int_row(f: InteractionPairFeatures) -> tuple[str, ...]:
    return (
        str(f.agent_id_a),
        str(f.agent_id_b),
        f.pair_type,
        _fmt(f.loss_of_separation),
        _fmt(f.mttcp),
        _fmt(f.thw),
        _fmt(f.ttc),
        _fmt(f.drac),
    )


def print_features(features: ScenarioFeatures, max_agents: int, max_pairs: int) -> None:
    """Print individual and interaction feature tables for a scenario."""
    total_agents = len(features.individual_features)
    shown_agents = min(total_agents, max_agents)
    print(f"\n  Individual features ({total_agents} agents, showing {shown_agents})")  # noqa: T201
    print(f"  {'─' * 60}")  # noqa: T201
    _print_table(_IND_COLS, [_ind_row(f) for f in features.individual_features[:shown_agents]])
    if shown_agents < total_agents:
        print(f"  … {total_agents - shown_agents} more agent(s) not shown")  # noqa: T201

    total_pairs = len(features.interaction_features)
    shown_pairs = min(total_pairs, max_pairs)
    print(f"\n  Interaction features ({total_pairs} pairs, showing {shown_pairs})")  # noqa: T201
    print(f"  {'─' * 60}")  # noqa: T201
    if features.interaction_features:
        _print_table(_INT_COLS, [_int_row(f) for f in features.interaction_features[:shown_pairs]])
        if shown_pairs < total_pairs:
            print(f"  … {total_pairs - shown_pairs} more pair(s) not shown")  # noqa: T201
    else:
        print("  (no candidate pairs within distance threshold)")  # noqa: T201


def print_scores(scores: ScenarioScores) -> None:
    """Print per-agent and scene-level score tables for a scenario."""
    ind_by_id = {s.agent_id: s.score for s in scores.individual_scores}
    int_by_id = {s.agent_id: s.score for s in scores.interaction_scores}

    all_ids = sorted({s.agent_id for s in scores.individual_scores} | {s.agent_id for s in scores.interaction_scores})
    total_agents = len(all_ids)
    print(f"\n  Scores  ({total_agents} agents)")  # noqa: T201
    print(f"  {'─' * 60}")  # noqa: T201
    rows = [(str(agent_id), _fmt(ind_by_id.get(agent_id)), _fmt(int_by_id.get(agent_id))) for agent_id in all_ids]
    _print_table(_SCORE_COLS, rows)

    print("\n  Scene scores")  # noqa: T201
    print(f"  {'─' * 60}")  # noqa: T201
    _print_table(
        _SCENE_SCORE_COLS,
        [(_fmt(scores.individual_scene_score), _fmt(scores.interaction_scene_score), _fmt(scores.scene_score))],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Feature analysis plots and airport summaries
# ──────────────────────────────────────────────────────────────────────────────

_IND_CONT_COLS = ("speed", "acceleration", "deceleration", "waiting_period", "kalman_difficulty")
_IND_CONT_LABELS = ("Speed (m/s)", "Acceleration (m/s²)", "Deceleration (m/s²)", "Waiting (s)", "Kalman Diff.")
_INT_CONT_COLS = ("loss_of_separation", "mttcp", "thw", "ttc", "drac")
_INT_CONT_LABELS = ("Loss of Separation", "MTTCP (s)", "THW (s)", "TTC (s)", "DRAC (m/s²)")


# Abbreviation map for agent pair types: each token is abbreviated to its first letter.
# e.g. "AIRCRAFT_AIRCRAFT" → "AA", "AIRCRAFT_UNKNOWN" → "AU", "UNKNOWN_VEHICLE" → "UV"
def _abbrev_pair_type(pair_type: str) -> str:
    return "".join(part[0] for part in pair_type.split("_") if part)


def _is_valid_float(v: object) -> bool:
    """Return True if *v* is a finite, non-NaN number."""
    if v is None:
        return False
    try:
        f = float(v)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return False
    return not (math.isnan(f) or math.isinf(f))


def _stats(values: list[float]) -> dict[str, float]:
    """Compute basic descriptive statistics for a non-empty list of floats."""
    n = len(values)
    if n == 0:
        return {}
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0.0
    sorted_v = sorted(values)
    mid = n // 2
    median = sorted_v[mid] if n % 2 else (sorted_v[mid - 1] + sorted_v[mid]) / 2
    return {
        "count": float(n),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "median": median,
    }


def plot_features(features_list: list[ScenarioFeatures], out_path: Path, title: str) -> None:  # noqa: PLR0912
    """Generate a 2x6 subplot figure showing feature distributions.

    Row 0: individual features — 5 box plots grouped by agent_type, 1 trajectory_type bar chart.
    Row 1: interaction features — 5 box plots grouped by pair_type, 1 pair_type bar chart.
    Inf and NaN values are excluded from continuous plots.
    """
    ind_feats = [f for sf in features_list for f in sf.individual_features]
    int_feats = [f for sf in features_list for f in sf.interaction_features]

    # Group individual continuous values by agent_type
    ind_cont: dict[str, dict[str, list[float]]] = {col: defaultdict(list) for col in _IND_CONT_COLS}
    for f in ind_feats:
        atype = f.agent_type
        for col in _IND_CONT_COLS:
            v = getattr(f, col)
            if _is_valid_float(v):
                ind_cont[col][atype].append(float(v))  # pyright: ignore[reportArgumentType]

    traj_counts: Counter[str] = Counter(f.trajectory_type for f in ind_feats if f.trajectory_type is not None)

    # Group interaction continuous values by pair_type (inf/NaN already excluded by _is_valid_float)
    # Keys are stored as abbreviated labels (e.g. "AA", "AU") for compact axis display.
    int_cont: dict[str, dict[str, list[float]]] = {col: defaultdict(list) for col in _INT_CONT_COLS}
    for f in int_feats:
        ptype = _abbrev_pair_type(f.pair_type)
        for col in _INT_CONT_COLS:
            v = getattr(f, col)
            if _is_valid_float(v):
                int_cont[col][ptype].append(float(v))  # pyright: ignore[reportArgumentType]

    pair_counts: Counter[str] = Counter(_abbrev_pair_type(f.pair_type) for f in int_feats)

    palette = sns.color_palette("pastel")

    fig, axes = plt.subplots(2, 6, figsize=(26, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Remove top and right spines from all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Row 0: individual continuous features (box plots)
    for col_idx, (col, label) in enumerate(zip(_IND_CONT_COLS, _IND_CONT_LABELS, strict=False)):
        ax = axes[0, col_idx]
        groups = sorted(ind_cont[col])
        if groups:
            data = [ind_cont[col][g] for g in groups]
            colors = [palette[i % len(palette)] for i in range(len(groups))]
            bp = ax.boxplot(data, tick_labels=groups, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors, strict=False):
                patch.set_facecolor(color)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Agent type")

    # Row 0, col 5: trajectory_type bar chart
    ax = axes[0, 5]
    if traj_counts:
        labels_t = sorted(traj_counts)
        vals_t = [traj_counts[k] for k in labels_t]
        bars = ax.bar(labels_t, vals_t, color=palette[: len(labels_t)])
        for bar, val in zip(bars, vals_t, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(val), ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Trajectory type", fontsize=10)
    ax.set_xlabel("Type")
    ax.set_ylabel("Count")

    # Row 1: interaction continuous features (box plots, grouped by abbreviated pair type)
    for col_idx, (col, label) in enumerate(zip(_INT_CONT_COLS, _INT_CONT_LABELS, strict=False)):
        ax = axes[1, col_idx]
        groups = sorted(int_cont[col])
        if groups:
            data = [int_cont[col][g] for g in groups]
            colors = [palette[i % len(palette)] for i in range(len(groups))]
            bp = ax.boxplot(data, tick_labels=groups, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors, strict=False):
                patch.set_facecolor(color)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Pair type")

    # Row 1, col 5: pair_type bar chart (abbreviated labels)
    ax = axes[1, 5]
    if pair_counts:
        labels_p = sorted(pair_counts)
        vals_p = [pair_counts[k] for k in labels_p]
        bars = ax.bar(labels_p, vals_p, color=palette[: len(labels_p)])
        for bar, val in zip(bars, vals_p, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(val), ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.set_title("Pair type", fontsize=10)
    ax.set_xlabel("Type")
    ax.set_ylabel("Count")

    axes[0, 0].set_ylabel("Individual features")
    axes[1, 0].set_ylabel("Interaction features")

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _write_airport_text_summary(  # noqa: PLR0912
    airport_id: str,
    features_list: list[ScenarioFeatures],
    scores_list: list[ScenarioScores],
    out_path: Path,
) -> None:
    """Write an aggregate text summary for all scenarios of one airport."""
    ind_feats = [f for sf in features_list for f in sf.individual_features]
    int_feats = [f for sf in features_list for f in sf.interaction_features]

    lines: list[str] = []
    sep = "═" * 70
    lines.append(f"\n{sep}")
    lines.append(
        f"  Airport : {airport_id}"
        f"  |  {len(features_list)} scenarios"
        f"  |  {len(ind_feats)} agents"
        f"  |  {len(int_feats)} pairs",
    )
    lines.append(sep)

    # Individual feature statistics table
    stat_header: tuple[str, ...] = ("feature", "count", "mean", "std", "min", "max", "median")
    lines.append("\n  Individual feature statistics")
    lines.append(f"  {'─' * 60}")
    ind_stat_rows: list[tuple[str, ...]] = []
    for col in _IND_CONT_COLS:
        vals = [float(getattr(f, col)) for f in ind_feats if _is_valid_float(getattr(f, col))]  # pyright: ignore[reportArgumentType]
        s = _stats(vals)
        if s:
            ind_stat_rows.append(
                (
                    col,
                    str(int(s["count"])),
                    f"{s['mean']:.4f}",
                    f"{s['std']:.4f}",
                    f"{s['min']:.4f}",
                    f"{s['max']:.4f}",
                    f"{s['median']:.4f}",
                ),
            )
        else:
            ind_stat_rows.append((col, "0", _NONE_STR, _NONE_STR, _NONE_STR, _NONE_STR, _NONE_STR))

    widths = _col_widths(stat_header, ind_stat_rows)
    lines.append("  " + "  ".join(h.ljust(w) for h, w in zip(stat_header, widths, strict=False)))
    lines.append("  " + "  ".join("-" * w for w in widths))
    lines.extend("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=False)) for row in ind_stat_rows)

    # Agent type distribution
    atype_counts: Counter[str] = Counter(f.agent_type for f in ind_feats)
    total_agents = len(ind_feats)
    lines.append("\n  Agent type distribution")
    for atype, count in sorted(atype_counts.items()):
        pct = 100.0 * count / total_agents if total_agents else 0.0
        lines.append(f"    {atype:<12}: {count} ({pct:.1f}%)")

    # Trajectory type distribution
    traj_counts: Counter[str] = Counter(f.trajectory_type for f in ind_feats if f.trajectory_type is not None)
    if traj_counts:
        total_traj = sum(traj_counts.values())
        lines.append("\n  Trajectory type distribution")
        for ttype, count in sorted(traj_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total_traj
            lines.append(f"    {ttype:<14}: {count} ({pct:.1f}%)")

    # Interaction feature statistics table
    lines.append("\n  Interaction feature statistics")
    lines.append(f"  {'─' * 60}")
    int_stat_rows: list[tuple[str, ...]] = []
    for col in _INT_CONT_COLS:
        vals = [float(getattr(f, col)) for f in int_feats if _is_valid_float(getattr(f, col))]  # pyright: ignore[reportArgumentType]
        s = _stats(vals)
        if s:
            int_stat_rows.append(
                (
                    col,
                    str(int(s["count"])),
                    f"{s['mean']:.4f}",
                    f"{s['std']:.4f}",
                    f"{s['min']:.4f}",
                    f"{s['max']:.4f}",
                    f"{s['median']:.4f}",
                ),
            )
        else:
            int_stat_rows.append((col, "0", _NONE_STR, _NONE_STR, _NONE_STR, _NONE_STR, _NONE_STR))

    widths_int = _col_widths(stat_header, int_stat_rows)
    lines.append("  " + "  ".join(h.ljust(w) for h, w in zip(stat_header, widths_int, strict=False)))
    lines.append("  " + "  ".join("-" * w for w in widths_int))
    lines.extend("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths_int, strict=False)) for row in int_stat_rows)

    # Pair type distribution
    pair_counts: Counter[str] = Counter(f.pair_type for f in int_feats)
    if pair_counts:
        total_pairs = sum(pair_counts.values())
        lines.append("\n  Pair type distribution")
        for ptype, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total_pairs
            lines.append(f"    {ptype:<30}: {count} ({pct:.1f}%)")

    # Scene score statistics
    lines.append("\n  Scene score statistics")
    lines.append(f"  {'─' * 60}")
    for score_field in ("individual_scene_score", "interaction_scene_score", "scene_score"):
        vals = [float(getattr(sc, score_field)) for sc in scores_list if getattr(sc, score_field) is not None]
        s = _stats(vals)
        if s:
            lines.append(
                f"    {score_field:<26}: mean={s['mean']:.4f}  std={s['std']:.4f}"
                f"  min={s['min']:.4f}  max={s['max']:.4f}",
            )
        else:
            lines.append(f"    {score_field:<26}: no data")

    lines.append("")
    out_path.write_text("\n".join(lines))


def generate_airport_summaries(
    airport_features: dict[str, list[ScenarioFeatures]],
    airport_scores: dict[str, list[ScenarioScores]],
    summaries_dir: Path | None,
    plots_dir: Path | None,
) -> None:
    """Write per-airport text summaries and aggregate feature plots."""
    for airport_id, features_list in airport_features.items():
        scores_list = airport_scores.get(airport_id, [])
        n = len(features_list)
        title = f"Airport {airport_id} — {n} scenario(s)"

        if summaries_dir is not None:
            out_path = summaries_dir / airport_id / "airport_summary.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _write_airport_text_summary(airport_id, features_list, scores_list, out_path)
            _LOGGER.info("Airport summary → %s", out_path)

        if plots_dir is not None:
            plot_path = plots_dir / airport_id / "airport_summary.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                plot_features(features_list, plot_path, title=title)
                _LOGGER.info("Airport plot    → %s", plot_path)
            except Exception:
                _LOGGER.exception("Failed to plot airport summary for %s", airport_id)


def load_map(maps_dir: Path, airport_id: str) -> MapData | None:
    """Load a MapData object for *airport_id* from *maps_dir*, or return None on failure."""
    map_path = maps_dir / airport_id / f"{airport_id}.pkl"
    if not map_path.exists():
        _LOGGER.warning("Map not found for airport %s at %s", airport_id, map_path)
        return None
    try:
        return MapData.from_pickle(map_path)
    except Exception:
        _LOGGER.exception("Failed to load map for airport %s from %s", airport_id, map_path)
        return None


def visualize_scenario(
    scenario: Scenario,
    airport_id: str,
    viz_dir: Path,
    viz_cache: dict[str, ScenarioVisualizer | None],
    cfg: DictConfig,
    *,
    scores: ScenarioScores | None = None,
) -> None:
    """Render and save a visualization for *scenario* using the configured ScenarioVisualizer.

    The visualizer is instantiated once per airport and cached in *viz_cache*. If instantiation
    fails for an airport, ``viz_cache[airport_id]`` is set to ``None`` and subsequent calls for
    that airport are silently skipped.

    Args:
        scenario: Scenario to visualize.
        airport_id: Airport identifier (used for cache key and output subdirectory).
        viz_dir: Root directory for visualization output; files are saved under ``viz_dir/airport_id/``.
        viz_cache: Mutable dict mapping airport_id → ScenarioVisualizer (or None on failure).
        cfg: Hydra DictConfig; must contain a ``visualizer`` sub-config with ``_target_`` and ``airport``.
        scores: Optional scenario scores forwarded to ``visualizer.visualize_scenario``.
    """
    if airport_id not in viz_cache:
        visualizer_cfg = copy.deepcopy(cfg.visualizer)
        visualizer_cfg.airport = airport_id
        try:
            viz_cache[airport_id] = hydra.utils.get_class(visualizer_cfg._target_)(visualizer_cfg)
        except Exception:
            _LOGGER.exception("Failed to instantiate visualizer for airport %s", airport_id)
            viz_cache[airport_id] = None

    visualizer = viz_cache[airport_id]
    if visualizer is None:
        return

    viz_out_dir = viz_dir / airport_id
    viz_out_dir.mkdir(parents=True, exist_ok=True)
    try:
        visualizer.visualize_scenario(scenario, scores=scores, output_dir=viz_out_dir)
    except Exception:
        _LOGGER.exception("Failed to visualize scenario %s", scenario.metadata.scenario_id)


def build_scenario_summary(
    scenario: Scenario,
    pkl_path: Path,
    features: ScenarioFeatures,
    scores: ScenarioScores,
    max_agents: int,
    max_pairs: int,
) -> str:
    """Capture printed scenario info, features, and scores into a string."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_scenario_info(scenario, pkl_path)
        print_features(features, max_agents=max_agents, max_pairs=max_pairs)
        print_scores(scores)
    return buf.getvalue()
