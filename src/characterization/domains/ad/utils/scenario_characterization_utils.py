"""Helper utilities for AD scenario characterization — printing, plotting, and summary generation."""

from __future__ import annotations

import io
import math
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    from pathlib import Path

from characterization.domains.ad.schemas.scenario import Scenario
from characterization.domains.ad.schemas.scenario_features import (
    IndividualAgentFeatures,
    InteractionPairFeatures,
    ScenarioFeatures,
)
from characterization.schemas.scenario_scores import ScenarioScores
from characterization.utils.logging_utils import get_pylogger

_LOGGER = get_pylogger(__name__, use_rank_zero_only=False)

# ──────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ──────────────────────────────────────────────────────────────────────────────

_IND_COLS = (
    "agent_id",
    "agent_type",
    "speed",
    "spd_limit_diff",
    "accel",
    "decel",
    "jerk",
    "waiting",
    "traj_type",
    "kalman_diff",
)
_INT_COLS = ("agent_id_a", "agent_id_b", "pair_type", "separation", "collision", "mttcp", "thw", "ttc", "drac")
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
    ts = m.timestamps_seconds
    duration = f"{float(ts[-1] - ts[0]):.1f}s" if len(ts) > 1 else "—"
    group = pkl_path.parent.name
    print(f"\n{'═' * 70}")  # noqa: T201
    print(f"  Scenario  : {m.scenario_id}")  # noqa: T201
    print(f"  Group     : {group}")  # noqa: T201
    print(f"  Agents    : {scenario.agent_data.num_agents}")  # noqa: T201
    print(f"  Duration  : {duration}  ({len(ts)} frames @ {m.frequency_hz} Hz)")  # noqa: T201
    print(f"  Ego agent : id={m.ego_vehicle_id}")  # noqa: T201
    print(f"{'═' * 70}")  # noqa: T201


def _ind_row(f: IndividualAgentFeatures) -> tuple[str, ...]:
    return (
        str(f.agent_id),
        f.agent_type,
        _fmt(f.speed),
        _fmt(f.speed_limit_diff),
        _fmt(f.acceleration),
        _fmt(f.deceleration),
        _fmt(f.jerk),
        _fmt(f.waiting_period),
        _fmt(f.trajectory_type),
        _fmt(f.kalman_difficulty),
    )


def _int_row(f: InteractionPairFeatures) -> tuple[str, ...]:
    return (
        str(f.agent_id_a),
        str(f.agent_id_b),
        f.pair_type,
        _fmt(f.separation),
        _fmt(f.collision),
        _fmt(f.mttcp),
        _fmt(f.thw),
        _fmt(f.ttc),
        _fmt(f.drac),
    )


def print_features(features: ScenarioFeatures, max_agents: int, max_pairs: int) -> None:
    """Print individual and interaction feature tables for a scenario."""
    ind = features.individual_features or []
    int_ = features.interaction_features or []

    total_agents = len(ind)
    shown_agents = min(total_agents, max_agents)
    print(f"\n  Individual features ({total_agents} agents, showing {shown_agents})")  # noqa: T201
    print(f"  {'─' * 60}")  # noqa: T201
    if ind:
        _print_table(_IND_COLS, [_ind_row(f) for f in ind[:shown_agents]])
        if shown_agents < total_agents:
            print(f"  … {total_agents - shown_agents} more agent(s) not shown")  # noqa: T201
    else:
        print("  (no individual features)")  # noqa: T201

    total_pairs = len(int_)
    shown_pairs = min(total_pairs, max_pairs)
    print(f"\n  Interaction features ({total_pairs} pairs, showing {shown_pairs})")  # noqa: T201
    print(f"  {'─' * 60}")  # noqa: T201
    if int_:
        _print_table(_INT_COLS, [_int_row(f) for f in int_[:shown_pairs]])
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
    rows = [(str(aid), _fmt(ind_by_id.get(aid)), _fmt(int_by_id.get(aid))) for aid in all_ids]
    _print_table(_SCORE_COLS, rows)

    print("\n  Scene scores")  # noqa: T201
    print(f"  {'─' * 60}")  # noqa: T201
    _print_table(
        _SCENE_SCORE_COLS,
        [(_fmt(scores.individual_scene_score), _fmt(scores.interaction_scene_score), _fmt(scores.scene_score))],
    )


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


# ──────────────────────────────────────────────────────────────────────────────
# Feature distribution plots
# ──────────────────────────────────────────────────────────────────────────────

_IND_CONT_COLS = (
    "speed",
    "speed_limit_diff",
    "acceleration",
    "deceleration",
    "jerk",
    "waiting_period",
    "kalman_difficulty",
)
_IND_CONT_LABELS = (
    "Speed (m/s)",
    "Speed Limit Diff (m/s)",
    "Acceleration (m/s²)",
    "Deceleration (m/s²)",
    "Jerk (m/s³)",
    "Waiting (s)",
    "Kalman Diff.",
)
_INT_CONT_COLS = ("separation", "mttcp", "thw", "ttc", "drac")
_INT_CONT_LABELS = ("Separation (m)", "MTTCP (s)", "THW (s)", "TTC (s)", "DRAC (m/s²)")


def _is_valid_float(v: object) -> bool:
    if v is None:
        return False
    try:
        f = float(v)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return False
    return not (math.isnan(f) or math.isinf(f))


def plot_scenario_features(features_list: list[ScenarioFeatures], out_path: Path, title: str) -> None:  # noqa: PLR0912
    """Generate a 2x6 subplot figure showing AD feature distributions.

    Row 0: individual features — 5 box plots (first 5 continuous) grouped by agent_type, 1 trajectory_type bar chart.
    Row 1: interaction features — 5 box plots grouped by pair_type, 1 pair_type bar chart.
    """
    sns.set_theme(style="whitegrid")

    ind_feats = [f for sf in features_list for f in (sf.individual_features or [])]
    int_feats = [f for sf in features_list for f in (sf.interaction_features or [])]

    # Individual: first 5 continuous columns for boxplots, +1 trajectory_type bar chart
    ind_cont_cols = _IND_CONT_COLS[:5]
    ind_cont_labels = _IND_CONT_LABELS[:5]

    ind_cont: dict[str, dict[str, list[float]]] = {col: defaultdict(list) for col in ind_cont_cols}
    for f in ind_feats:
        atype = f.agent_type
        for col in ind_cont_cols:
            v = getattr(f, col, None)
            if _is_valid_float(v):
                ind_cont[col][atype].append(float(v))  # type: ignore[arg-type]

    traj_counts: Counter[str] = Counter(f.trajectory_type for f in ind_feats if f.trajectory_type)

    # Interaction: continuous columns for boxplots, pair_type bar chart
    int_cont: dict[str, dict[str, list[float]]] = {col: defaultdict(list) for col in _INT_CONT_COLS}
    for f in int_feats:
        ptype = f.pair_type
        for col in _INT_CONT_COLS:
            v = getattr(f, col, None)
            if _is_valid_float(v):
                int_cont[col][ptype].append(float(v))  # type: ignore[arg-type]

    pair_counts: Counter[str] = Counter(f.pair_type for f in int_feats)

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle(title, fontsize=12, y=1.01)

    # Row 0: individual continuous boxplots
    for col_idx, (col, label) in enumerate(zip(ind_cont_cols, ind_cont_labels, strict=False)):
        ax = axes[0, col_idx]
        data = ind_cont[col]
        if data:
            ax.boxplot(list(data.values()), labels=list(data.keys()), vert=True)
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label, fontsize=9)

    # Row 0, col 5: trajectory_type bar chart
    ax = axes[0, 5]
    if traj_counts:
        ax.bar(list(traj_counts.keys()), list(traj_counts.values()))
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Trajectory Type", fontsize=9)

    # Row 1: interaction continuous boxplots
    for col_idx, (col, label) in enumerate(zip(_INT_CONT_COLS, _INT_CONT_LABELS, strict=False)):
        ax = axes[1, col_idx]
        data = int_cont[col]
        if data:
            ax.boxplot(list(data.values()), labels=list(data.keys()), vert=True)
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label, fontsize=9)

    # Row 1, col 5: pair_type bar chart
    ax = axes[1, 5]
    if pair_counts:
        labels = [pt[:6] for pt in pair_counts]  # abbreviate long pair type names
        ax.bar(labels, list(pair_counts.values()))
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Pair Type", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Scenario loading and group summaries
# ──────────────────────────────────────────────────────────────────────────────


def load_scenario(pkl_path: Path) -> Scenario | None:
    """Load an AD scenario from *pkl_path*, returning ``None`` on failure."""
    try:
        return Scenario.from_pickle(pkl_path)
    except Exception:
        _LOGGER.exception("Failed to load scenario from %s", pkl_path)
        return None


def _stats(values: list[float]) -> dict[str, float]:
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


def _write_group_text_summary(
    group_id: str,
    features_list: list[ScenarioFeatures],
    scores_list: list[ScenarioScores],
    out_path: Path,
) -> None:
    n = len(features_list)
    ind_feats = [f for sf in features_list for f in (sf.individual_features or [])]
    int_feats = [f for sf in features_list for f in (sf.interaction_features or [])]

    lines: list[str] = [
        f"Group: {group_id}",
        f"Scenarios: {n}",
        f"Total agents: {len(ind_feats)}",
        f"Total interaction pairs: {len(int_feats)}",
        "",
    ]

    for col in _IND_CONT_COLS:
        values = [float(v) for f in ind_feats if _is_valid_float(v := getattr(f, col, None))]  # type: ignore[arg-type]
        stats = _stats(values)
        if stats:
            lines.append(
                f"  {col}: n={int(stats['count'])}  mean={stats['mean']:.4f}  "
                f"std={stats['std']:.4f}  min={stats['min']:.4f}  max={stats['max']:.4f}"
            )

    lines.append("")
    for col in _INT_CONT_COLS:
        values = [float(v) for f in int_feats if _is_valid_float(v := getattr(f, col, None))]  # type: ignore[arg-type]
        stats = _stats(values)
        if stats:
            lines.append(
                f"  {col}: n={int(stats['count'])}  mean={stats['mean']:.4f}  "
                f"std={stats['std']:.4f}  min={stats['min']:.4f}  max={stats['max']:.4f}"
            )

    if scores_list:
        scene_scores = [s.scene_score for s in scores_list if s.scene_score is not None]
        stats = _stats(scene_scores)
        if stats:
            lines.append("")
            lines.append(
                f"  scene_score: n={int(stats['count'])}  mean={stats['mean']:.4f}  "
                f"std={stats['std']:.4f}  min={stats['min']:.4f}  max={stats['max']:.4f}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def generate_group_summaries(
    group_id: str,
    features_list: list[ScenarioFeatures],
    scores_list: list[ScenarioScores],
    summaries_dir: Path | None,
    plots_dir: Path | None,
) -> None:
    """Write per-group text summary and aggregate feature distribution plot."""
    n = len(features_list)
    title = f"Group {group_id} — {n} scenario(s)"

    if summaries_dir is not None:
        out_path = summaries_dir / group_id / "group_summary.txt"
        _write_group_text_summary(group_id, features_list, scores_list, out_path)
        _LOGGER.info("Group summary → %s", out_path)

    if plots_dir is not None:
        plot_path = plots_dir / group_id / "group_summary.png"
        try:
            plot_scenario_features(features_list, plot_path, title=title)
            _LOGGER.info("Group plot    → %s", plot_path)
        except Exception:
            _LOGGER.exception("Failed to plot group summary for %s", group_id)
