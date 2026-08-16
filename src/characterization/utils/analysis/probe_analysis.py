"""Probe analysis utilities.

Helpers for loading, transforming, and visualizing the ``probe_summary.csv`` produced by
``run_scenario_probing``.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PathCollection

from characterization.processors.probe_processor import CSV_FIELDS
from characterization.utils.analysis.common_analysis import (
    CATEGORY_BOUNDARY_COLOR,
    get_dataset_colors,
    plot_histograms_from_dataframe,
)
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)

# Consistent colour palette across all plots
_PALETTE = {"Ego": "#5bc0de", "Non-ego": "#5cb85c"}
_PIE_COLORS = ["#d9534f", "#5bc0de", "#5cb85c"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_probe_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate the probe summary CSV.

    Args:
        csv_path: Path to ``probe_summary.csv``.

    Returns:
        Raw DataFrame with all CSV rows.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    if not csv_path.exists():
        msg = f"probe_csv not found: {csv_path}"
        raise FileNotFoundError(msg)
    df = pd.read_csv(csv_path)
    missing = [c for c in CSV_FIELDS if c not in df.columns]
    if missing:
        msg = f"Missing columns in {csv_path}: {missing}"
        raise ValueError(msg)
    return df


def build_probed_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to probed rows and derive helper columns.

    Args:
        df: Full probe summary DataFrame.

    Returns:
        Sub-DataFrame with ``probe_found == "yes"``, numeric score columns, and ``n_affected``.
    """
    df_probed: pd.DataFrame = df.loc[df["probe_found"] == "yes"].copy()  # pyright: ignore[reportAssignmentType]
    for col in ("score_before", "score_after", "score_delta"):
        df_probed[col] = pd.to_numeric(df_probed[col], errors="coerce")  # pyright: ignore[reportArgumentType]
    df_probed["n_affected"] = df_probed["affected_agent_ids"].apply(  # pyright: ignore[reportAttributeAccessIssue]
        lambda x: len(str(x).split(";")) if pd.notna(x) and str(x).strip() else 0  # pyright: ignore[reportUnknownLambdaType]
    )
    return df_probed


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_probe_outcome_pie(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Pie chart showing the three probe outcome categories across all scenarios.

    Args:
        df: Full probe summary DataFrame.
        output_dir: Directory to write ``probe_outcome_pie.png``.
        dpi: Output image resolution.
    """
    no_probe = int((df["probe_found"] == "no").sum())
    ego_probe = int(((df["probe_found"] == "yes") & (df["is_ego_agent"] == "yes")).sum())
    other_probe = int(((df["probe_found"] == "yes") & (df["is_ego_agent"] == "no")).sum())

    labels = [f"No probe\n(n={no_probe})", f"Ego\n(n={ego_probe})", f"Non-ego\n(n={other_probe})"]
    sizes = [no_probe, ego_probe, other_probe]

    _fig, ax = plt.subplots(figsize=(7, 7))
    _, _, autotexts = ax.pie(  # pyright: ignore[reportAssignmentType]
        sizes,
        labels=labels,
        colors=_PIE_COLORS,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(plt.rcParams["font.size"] * 0.8)
    ax.set_title("Probe Outcome\nDistribution", pad=15)
    plt.tight_layout()
    out = output_dir / "probe_outcome_pie.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved probe outcome pie chart -> %s", out)


def plot_score_distributions(df_probed: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Overlapping KDE/histogram density curves for score_before and score_after.

    Args:
        df_probed: Probed-only DataFrame with numeric score columns.
        output_dir: Directory to write ``score_distributions.png``.
        dpi: Output image resolution.
    """
    if df_probed.empty:
        logger.warning("No probed scenarios — skipping score distributions plot.")
        return
    scores_df: pd.DataFrame = df_probed[["score_after", "score_before"]].rename(  # pyright: ignore[reportAssignmentType, reportCallIssue]
        columns={"score_before": "Score Before", "score_after": "Score After"}
    )
    out = output_dir / "score_distributions.png"
    plot_histograms_from_dataframe(scores_df, out, dpi)
    logger.info("Saved score distributions -> %s", out)


def plot_score_delta_density(df_probed: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """KDE/histogram of score_delta with mean and median reference lines.

    Args:
        df_probed: Probed-only DataFrame with numeric ``score_delta`` column.
        output_dir: Directory to write ``score_delta_density.png``.
        dpi: Output image resolution.
    """
    if df_probed.empty:
        logger.warning("No probed scenarios — skipping score delta density plot.")
        return
    deltas: pd.Series = df_probed["score_delta"].dropna()  # pyright: ignore[reportAssignmentType]
    mean_val = float(deltas.mean())
    median_val = float(deltas.median())

    _fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(deltas, kde=True, stat="density", color=_PALETTE["Ego"], edgecolor="white", alpha=0.6, ax=ax)  # pyright: ignore[reportArgumentType]
    ax.axvline(mean_val, color="#d9534f", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.4f}")
    ax.axvline(median_val, color="#f0ad4e", linestyle="-.", linewidth=1.5, label=f"Median: {median_val:.4f}")
    ax.legend()
    ax.set_xlabel("Score Delta")
    ax.set_ylabel("Density")
    ax.set_title("Score Delta Distribution\n(Probed Scenarios)")
    sns.despine(top=True, right=True)
    plt.grid(visible=True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = output_dir / "score_delta_density.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved score delta density -> %s", out)


def plot_score_delta_by_agent_type(df_probed: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Violin plot of score_delta split by ego vs non-ego probed agent.

    Args:
        df_probed: Probed-only DataFrame with ``is_ego_agent`` and ``score_delta`` columns.
        output_dir: Directory to write ``score_delta_by_agent_type.png``.
        dpi: Output image resolution.
    """
    if df_probed.empty:
        logger.warning("No probed scenarios — skipping score delta by agent type plot.")
        return
    df_plot: pd.DataFrame = df_probed[["is_ego_agent", "score_delta"]].copy()  # pyright: ignore[reportAssignmentType]
    df_plot["Agent Type"] = df_plot["is_ego_agent"].map({"yes": "Ego", "no": "Non-ego"})  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    _fig, ax = plt.subplots(figsize=(7, 6))
    sns.violinplot(
        data=df_plot,  # pyright: ignore[reportArgumentType]
        x="Agent Type",
        y="score_delta",
        hue="Agent Type",
        palette=_PALETTE,
        inner="box",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Probed Agent Type")
    ax.set_ylabel("Score Delta")
    ax.set_title("Score Delta by\nProbed Agent Type")
    sns.despine(top=True, right=True)
    plt.grid(visible=True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = output_dir / "score_delta_by_agent_type.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved score delta by agent type -> %s", out)


def plot_score_scatter(
    df_probed: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    highlight_scenario_id: str | None = None,
) -> None:
    """Scatter plot of score_before vs score_after with a y=x no-change reference line.

    Args:
        df_probed: Probed-only DataFrame with numeric score columns and ``is_ego_agent``.
        output_dir: Directory to write ``score_scatter.png``, or ``score_scatter_<id>.png`` when a
            scenario is highlighted.
        dpi: Output image resolution.
        highlight_scenario_id: When set, that scenario's point is marked with a red X.
    """
    if df_probed.empty:
        logger.warning("No probed scenarios — skipping score scatter plot.")
        return
    df_plot: pd.DataFrame = df_probed[["scenario_id", "score_before", "score_after", "is_ego_agent"]].copy()  # pyright: ignore[reportAssignmentType]
    df_plot["Agent Type"] = df_plot["is_ego_agent"].map({"yes": "Ego", "no": "Non-ego"})  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    _fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(
        data=df_plot,  # pyright: ignore[reportArgumentType]
        x="score_before",
        y="score_after",
        hue="Agent Type",
        palette=_PALETTE,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        s=25,
        ax=ax,
    )
    lo = float(min(df_plot["score_before"].min(), df_plot["score_after"].min()))
    hi = float(max(df_plot["score_before"].max(), df_plot["score_after"].max()))
    margin = (hi - lo) * 0.05 if hi > lo else 0.05
    diag = np.linspace(lo - margin, hi + margin, 100)
    ax.plot(diag, diag, "k--", linewidth=1, label="y = x")

    out = output_dir / "score_scatter.png"
    if highlight_scenario_id is not None:
        row = df_plot.loc[df_plot["scenario_id"].astype(str) == str(highlight_scenario_id)]
        if row.empty:
            logger.warning("Scenario %s not in the probed set — skipping its scatter.", highlight_scenario_id)
            plt.close()
            return
        # Kept out of the legend: the ID is in the title, and a legend key at this marker size crowds the axes.
        ax.scatter(
            row["score_before"],
            row["score_after"],
            marker="X",
            s=plt.rcParams["lines.markersize"] ** 2 * 5,
            color=CATEGORY_BOUNDARY_COLOR,
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
            label="_nolegend_",
        )
        out = output_dir / f"score_scatter_{highlight_scenario_id}.png"

    ax.legend(loc="lower right", markerscale=1.5, framealpha=0.9)
    ax.set_xlabel("Score Before")
    ax.set_ylabel("Score After")
    ax.set_title("Score Before vs. After\n(Probed Scenarios)")
    sns.despine(top=True, right=True)
    plt.grid(visible=True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved score scatter -> %s", out)


def plot_top_score_delta_scatters(
    df_probed: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    top_n: int = 10,
) -> list[str]:
    """One scatter per top-``top_n`` scenario by ``score_delta``, each highlighting its own point.

    Args:
        df_probed: Probed-only DataFrame with numeric score columns and ``is_ego_agent``.
        output_dir: Directory to write ``score_scatter_<scenario_id>.png``.
        dpi: Output image resolution.
        top_n: How many of the largest score increases to plot.

    Returns:
        The highlighted scenario IDs, largest ``score_delta`` first.
    """
    if df_probed.empty:
        logger.warning("No probed scenarios — skipping top score delta scatters.")
        return []
    top = df_probed.nlargest(top_n, "score_delta")
    scenario_ids = [str(scenario_id) for scenario_id in top["scenario_id"]]
    for scenario_id in scenario_ids:
        plot_score_scatter(df_probed, output_dir, dpi, highlight_scenario_id=scenario_id)
    logger.info("Top %d score_delta scenarios: %s", len(scenario_ids), ",".join(scenario_ids))
    return scenario_ids


def plot_affected_agents_histogram(df_probed: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Stacked histogram of the number of affected agents per probe, split by agent type.

    Args:
        df_probed: Probed-only DataFrame with ``n_affected`` and ``is_ego_agent``.
        output_dir: Directory to write ``affected_agents_histogram.png``.
        dpi: Output image resolution.
    """
    if df_probed.empty:
        logger.warning("No probed scenarios — skipping affected agents histogram.")
        return
    df_plot: pd.DataFrame = df_probed[["n_affected", "is_ego_agent"]].copy()  # pyright: ignore[reportAssignmentType]
    df_plot["Agent Type"] = df_plot["is_ego_agent"].map({"yes": "Ego", "no": "Non-ego"})  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    _fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=df_plot,  # pyright: ignore[reportArgumentType]
        x="n_affected",
        hue="Agent Type",
        palette=_PALETTE,
        multiple="stack",
        discrete=True,
        shrink=0.8,
        edgecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Affected Agents")
    ax.set_ylabel("Count")
    ax.set_title("Affected Agents per Probe")
    sns.despine(top=True, right=True)
    plt.grid(visible=True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = output_dir / "affected_agents_histogram.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved affected agents histogram -> %s", out)


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def save_probe_summary_json(df: pd.DataFrame, df_probed: pd.DataFrame, output_dir: Path) -> None:
    """Write key aggregate statistics to ``probe_analysis_summary.json``.

    Args:
        df: Full probe summary DataFrame.
        df_probed: Probed-only sub-DataFrame.
        output_dir: Directory to write the JSON file.
    """
    total = len(df)
    n_probed = len(df_probed)
    n_ego = int((df_probed["is_ego_agent"] == "yes").sum()) if not df_probed.empty else 0
    n_non_ego = int((df_probed["is_ego_agent"] == "no").sum()) if not df_probed.empty else 0
    deltas: pd.Series = (  # pyright: ignore[reportAssignmentType]
        df_probed["score_delta"].dropna() if not df_probed.empty else pd.Series(dtype=float)  # pyright: ignore[reportAttributeAccessIssue]
    )

    summary: dict[str, object] = {
        "total_scenarios": total,
        "n_probed": n_probed,
        "n_no_probe": total - n_probed,
        "probe_rate": round(n_probed / total, 4) if total else 0.0,
        "n_ego_probe": n_ego,
        "n_non_ego_probe": n_non_ego,
        "score_delta": {
            "mean": round(float(deltas.mean()), 6) if not deltas.empty else None,
            "std": round(float(deltas.std()), 6) if not deltas.empty else None,
            "median": round(float(deltas.median()), 6) if not deltas.empty else None,
            "min": round(float(deltas.min()), 6) if not deltas.empty else None,
            "max": round(float(deltas.max()), 6) if not deltas.empty else None,
        },
    }
    out = output_dir / "probe_analysis_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info("Saved probe analysis summary -> %s", out)


# ---------------------------------------------------------------------------
# Multi-dataset combined overlay plots
# ---------------------------------------------------------------------------


def plot_multi_dataset_probe_outcomes(
    dfs_by_dataset: Mapping[str, pd.DataFrame],
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Grouped bar chart comparing probe outcome percentages across multiple datasets.

    Args:
        dfs_by_dataset: Mapping from dataset label to the full probe summary DataFrame.
        output_dir: Directory to write ``probe_outcomes_combined.png``.
        dpi: Output image resolution.
    """
    dataset_labels = list(dfs_by_dataset.keys())
    colors = get_dataset_colors(dataset_labels)

    categories = ["No probe", "Ego", "Non-ego"]
    x = np.arange(len(categories))
    bar_width = 0.8 / max(len(dataset_labels), 1)

    _fig, ax = plt.subplots(figsize=(9, 5))
    max_pct = 0.0
    for i, label in enumerate(dataset_labels):
        df = dfs_by_dataset[label]
        total = max(len(df), 1)
        no_probe = int((df["probe_found"] == "no").sum())
        ego_probe = int(((df["probe_found"] == "yes") & (df["is_ego_agent"] == "yes")).sum())
        non_ego_probe = int(((df["probe_found"] == "yes") & (df["is_ego_agent"] == "no")).sum())
        counts = [no_probe, ego_probe, non_ego_probe]
        pcts = [100.0 * v / total for v in counts]
        offset = (i - len(dataset_labels) / 2 + 0.5) * bar_width
        max_pct = max(max_pct, *pcts)
        bars = ax.bar(x + offset, pcts, width=bar_width, label=f"{label} (n={total})", color=colors[label], alpha=0.8)
        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                str(count),
                ha="center",
                va="bottom",
                fontsize=plt.rcParams["font.size"] * 0.5,
            )

    # Just enough headroom for the count labels above the tallest bar.
    ax.set_ylim(0, max(max_pct * 1.1, 1.0))
    ax.tick_params(axis="y", labelsize=plt.rcParams["ytick.labelsize"] * 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Probe Outcome Distribution\nby Dataset")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncols=len(dataset_labels),
        frameon=False,
        fontsize=plt.rcParams["legend.fontsize"] * 0.7,
    )
    sns.despine(top=True, right=True)
    plt.grid(visible=True, axis="y", linestyle="--", alpha=0.4)
    out = output_dir / "probe_outcomes_combined.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("Saved combined probe outcomes chart -> %s", out)


def plot_multi_dataset_probe_score_distributions(
    dfs_probed_by_dataset: Mapping[str, pd.DataFrame],
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Overlapping KDE histograms of score_before and score_after across multiple datasets.

    Args:
        dfs_probed_by_dataset: Mapping from dataset label to the probed-only DataFrame.
        output_dir: Directory to write ``score_distributions_combined.png``.
        dpi: Output image resolution.
    """
    dataset_labels = list(dfs_probed_by_dataset.keys())
    colors = get_dataset_colors(dataset_labels)

    _fig, ax = plt.subplots(figsize=(10, 6))
    for label, df_probed in dfs_probed_by_dataset.items():
        if df_probed.empty:
            continue
        color = colors[label]
        for col, linestyle, col_label, alpha in [
            ("score_before", "--", "Before", 0.20),
            ("score_after", "-", "After", 0.75),
        ]:
            numeric_col: pd.Series = pd.to_numeric(df_probed[col], errors="coerce")  # pyright: ignore[reportAssignmentType]
            values = numeric_col.dropna()
            sns.histplot(
                x=values,
                color=color,
                bins=20,
                kde=True,
                stat="density",
                alpha=alpha,
                edgecolor="white",
                linestyle=linestyle,
                label=f"{label} — {col_label} (n={len(values)})",
                ax=ax,
            )

    sns.despine(top=True, right=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Before / After Distributions\n(Probed Scenarios)")
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    ax.legend(title="Dataset — Metric", fontsize=plt.rcParams["legend.fontsize"] * 0.6)
    plt.tight_layout()
    out = output_dir / "score_distributions_combined.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved combined score distributions -> %s", out)


def _score_values(df_probed: pd.DataFrame, col: str) -> pd.Series:
    """Returns the numeric, NaN-free values of *col*, empty when the DataFrame holds no probed scenarios."""
    if df_probed.empty:
        return pd.Series(dtype=float)
    numeric: pd.Series = pd.to_numeric(df_probed[col], errors="coerce")  # pyright: ignore[reportAssignmentType]
    return numeric.dropna()


def _score_bounds(dfs_probed_by_dataset: Mapping[str, pd.DataFrame], col: str) -> float:
    """Pooled maximum of *col* across every dataset, with a small margin so edge points stay inside the axes.

    The scatter panels take their shared axis limits from here, so a cloud's distance above the diagonal means the
    same thing in every panel.

    Args:
        dfs_probed_by_dataset: Mapping from dataset label to the probed-only DataFrame.
        col: Score column to bound, e.g. ``score_before``.

    Returns:
        The upper bound, or 1.0 when no dataset holds any probed scenario.
    """
    pooled = pd.concat(
        [_score_values(df, col) for df in dfs_probed_by_dataset.values()],
        ignore_index=True,
    )
    return 1.0 if pooled.empty else float(pooled.max()) * 1.02


def plot_multi_dataset_probe_score_distributions_grid(
    dfs_probed_by_dataset: Mapping[str, pd.DataFrame],
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Grid of histogram panels: one per dataset (before vs after) plus two combined panels.

    Layout — N+2 columns sharing a y-axis:
      - Columns 1..N: per-dataset panel showing score_before (dashed) and score_after (solid).
      - Column N+1: all datasets' score_before overlaid.
      - Column N+2: all datasets' score_after overlaid.

    Args:
        dfs_probed_by_dataset: Mapping from dataset label to the probed-only DataFrame.
        output_dir: Directory to write ``score_distributions_grid_combined.png``.
        dpi: Output image resolution.
    """
    dataset_labels = list(dfs_probed_by_dataset.keys())
    colors = get_dataset_colors(dataset_labels)
    n = len(dataset_labels)

    x_max = 0.0
    for _df in dfs_probed_by_dataset.values():
        if _df.empty:
            continue
        for _col in ("score_before", "score_after"):
            _s: pd.Series = pd.to_numeric(_df[_col], errors="coerce")  # pyright: ignore[reportAssignmentType]
            x_max = max(x_max, float(_s.max()))

    fig, axes = plt.subplots(1, n + 2, figsize=(6 * (n + 2), 7), sharey=True, sharex=True)

    # Per-dataset panels: before in the dataset's color, after in red (matching individual plots)
    for i, label in enumerate(dataset_labels):
        ax = axes[i]
        df_probed = dfs_probed_by_dataset[label]
        for col, color, col_label in [
            ("score_before", colors[label], "Before"),
            ("score_after", _PIE_COLORS[0], "After"),
        ]:
            if df_probed.empty:
                continue
            numeric_col: pd.Series = pd.to_numeric(df_probed[col], errors="coerce")  # pyright: ignore[reportAssignmentType]
            values = numeric_col.dropna()
            sns.histplot(
                x=values,
                color=color,
                kde=True,
                stat="density",
                alpha=0.5,
                edgecolor="white",
                label=f"{col_label} (n={len(values)})",
                ax=ax,
            )
        ax.set_title(label)
        ax.set_xlabel("Score")
        ax.legend(fontsize=plt.rcParams["legend.fontsize"] * 0.7)
        ax.grid(visible=True, linestyle="--", alpha=0.4)
        sns.despine(ax=ax, top=True, right=True)

    # Combined "All Before" and "All After" panels
    for panel_idx, (col, col_label) in enumerate([("score_before", "All — Before"), ("score_after", "All — After")]):
        ax = axes[n + panel_idx]
        for label, df_probed in dfs_probed_by_dataset.items():
            if df_probed.empty:
                continue
            numeric_col2: pd.Series = pd.to_numeric(df_probed[col], errors="coerce")  # pyright: ignore[reportAssignmentType]
            values = numeric_col2.dropna()
            sns.histplot(
                x=values,
                color=colors[label],
                bins=20,
                kde=True,
                stat="density",
                alpha=0.5,
                edgecolor="white",
                label=f"{label} (n={len(values)})",
                ax=ax,
            )
        ax.set_title(col_label)
        ax.set_xlabel("Score")
        ax.legend(fontsize=plt.rcParams["legend.fontsize"] * 0.7)
        ax.grid(visible=True, linestyle="--", alpha=0.4)
        sns.despine(ax=ax, top=True, right=True)

    axes[0].set_xlim(0, x_max)
    axes[0].set_ylabel("Density")
    fig.suptitle("Score Distributions per Dataset and Combined\n(Probed Scenarios)")
    plt.tight_layout(rect=(0, 0, 1, 0.86))
    out = output_dir / "score_distributions_grid_combined.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved score distributions grid -> %s", out)


def plot_multi_dataset_probe_score_scatter(
    dfs_probed_by_dataset: Mapping[str, pd.DataFrame],
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Score before vs. after scatters: one panel per dataset plus an all-datasets panel.

    Every panel shares the same axes and ``y = x`` reference, so the vertical distance of a cloud above the diagonal
    is comparable across panels. The axes are bounded per variable — x by the largest ``score_before``, y by the
    largest ``score_after`` — so the before-score spread fills the panel width instead of being squeezed into the
    left third by the much longer after-score tail. Points are colored by dataset; the per-agent-type split lives in
    the single-dataset ``score_scatter.png``.

    Args:
        dfs_probed_by_dataset: Mapping from dataset label to the probed-only DataFrame.
        output_dir: Directory to write ``score_scatter_combined.png``.
        dpi: Output image resolution.
    """
    dataset_labels = list(dfs_probed_by_dataset.keys())
    colors = get_dataset_colors(dataset_labels)
    x_hi = _score_bounds(dfs_probed_by_dataset, "score_before")
    y_hi = _score_bounds(dfs_probed_by_dataset, "score_after")
    panels: list[tuple[str, list[str]]] = [(label, [label]) for label in dataset_labels]
    panels.append(("All datasets", dataset_labels))

    panel_h = 3.2
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(2.6 * len(panels) + 1.0, panel_h + 1.2),
        sharex=True,
        sharey=True,
        squeeze=False,
        gridspec_kw={"wspace": 0.05},
    )
    for ax, (title, labels_in_panel) in zip(axes[0], panels, strict=True):
        for label in labels_in_panel:
            before = _score_values(dfs_probed_by_dataset[label], "score_before")
            after = _score_values(dfs_probed_by_dataset[label], "score_after")
            if before.empty or after.empty:
                continue
            ax.scatter(
                before,
                after,
                color=colors[label],
                alpha=0.45,
                edgecolor="none",
                s=10,
                label=f"{label} (n={len(before)})",
            )
        ax.plot([0, x_hi], [0, x_hi], "k--", linewidth=1, label="y = x")
        ax.set_xlim(0, x_hi)
        ax.set_ylim(0, y_hi)
        # Counts live in the shared legend: spelled out per panel the titles collide at this panel width.
        ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"] * 0.7)
        # Same tick label size as the probe outcomes chart.
        ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"] * 0.7)
        ax.grid(visible=True, linestyle="--", alpha=0.4)
        sns.despine(ax=ax, top=True, right=True)

    # Figure-level axis labels: the panels share both scales, so per-panel labels would just repeat four times.
    label_size = plt.rcParams["axes.labelsize"] * 0.75
    fig.supxlabel("Score Before", y=0.10, fontsize=label_size)
    fig.supylabel("Score After", fontsize=label_size)
    # One legend under the figure: in-axes keys at this point density sit on top of the cloud.
    handles, legend_labels = axes[0][-1].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncols=len(legend_labels),
        frameon=False,
        fontsize=plt.rcParams["legend.fontsize"] * 0.6,
    )
    # The keys inherit the scatter's alpha and point size, which leaves them nearly invisible at legend scale.
    for handle in legend.legend_handles:
        if handle is None:
            continue
        handle.set_alpha(1.0)
        if isinstance(handle, PathCollection):
            handle.set_sizes([60])
    # Sized off axes.titlesize: the theme leaves figure.titlesize as a keyword ("large"), which is not scalable.
    fig.suptitle("Score Before vs. After Probing", fontsize=plt.rcParams["axes.titlesize"] * 0.9)
    # Margins set explicitly: tight_layout would re-expand the gaps between the panels.
    fig.subplots_adjust(left=0.10, right=0.99, top=0.84, bottom=0.24, wspace=0.05)
    out = output_dir / "score_scatter_combined.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved combined score scatter -> %s", out)


def plot_multi_dataset_probe_score_delta_density(
    dfs_probed_by_dataset: Mapping[str, pd.DataFrame],
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Overlapping KDE histograms of score_delta across multiple datasets.

    Args:
        dfs_probed_by_dataset: Mapping from dataset label to the probed-only DataFrame.
        output_dir: Directory to write ``score_delta_density_combined.png``.
        dpi: Output image resolution.
    """
    dataset_labels = list(dfs_probed_by_dataset.keys())
    colors = get_dataset_colors(dataset_labels)

    _fig, ax = plt.subplots(figsize=(10, 6))
    for label, df_probed in dfs_probed_by_dataset.items():
        if df_probed.empty:
            continue
        numeric_delta: pd.Series = pd.to_numeric(df_probed["score_delta"], errors="coerce")  # pyright: ignore[reportAssignmentType]
        deltas = numeric_delta.dropna()
        if deltas.empty:
            continue
        sns.histplot(
            x=deltas,
            color=colors[label],
            bins=20,
            kde=True,
            stat="density",
            alpha=0.5,
            edgecolor=None,
            label=f"{label} (n={len(deltas)})",
            ax=ax,
        )

    sns.despine(top=True, right=True)
    ax.set_xlabel("Score Delta")
    ax.set_ylabel("Density")
    ax.set_title("Score Delta Distribution\n(Probed Scenarios)")
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    ax.legend(title="Dataset", fontsize=plt.rcParams["legend.fontsize"] * 0.7)
    plt.tight_layout()
    out = output_dir / "score_delta_density_combined.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved combined score delta density -> %s", out)
