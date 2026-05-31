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

from characterization.processors.probe_processor import CSV_FIELDS
from characterization.utils.analysis.common_analysis import get_dataset_colors, plot_histograms_from_dataframe
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

    labels = [f"No probe\n(n={no_probe})", f"Ego probe\n(n={ego_probe})", f"Non-ego probe\n(n={other_probe})"]
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
        t.set_fontsize(11)
    ax.set_title("Probe Outcome Distribution", fontsize=14, fontweight="bold", pad=15)
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
    ax.legend(fontsize=11)
    ax.set_xlabel("Score Delta (score_after - score_before)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Delta Distribution (Probed Scenarios)", fontsize=14, fontweight="bold")
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
    ax.set_xlabel("Probed Agent Type", fontsize=12)
    ax.set_ylabel("Score Delta", fontsize=12)
    ax.set_title("Score Delta by Probed Agent Type", fontsize=14, fontweight="bold")
    sns.despine(top=True, right=True)
    plt.grid(visible=True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = output_dir / "score_delta_by_agent_type.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved score delta by agent type -> %s", out)


def plot_score_scatter(df_probed: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Scatter plot of score_before vs score_after with a y=x no-change reference line.

    Args:
        df_probed: Probed-only DataFrame with numeric score columns and ``is_ego_agent``.
        output_dir: Directory to write ``score_scatter.png``.
        dpi: Output image resolution.
    """
    if df_probed.empty:
        logger.warning("No probed scenarios — skipping score scatter plot.")
        return
    df_plot: pd.DataFrame = df_probed[["score_before", "score_after", "is_ego_agent"]].copy()  # pyright: ignore[reportAssignmentType]
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
        s=50,
        ax=ax,
    )
    lo = float(min(df_plot["score_before"].min(), df_plot["score_after"].min()))
    hi = float(max(df_plot["score_before"].max(), df_plot["score_after"].max()))
    margin = (hi - lo) * 0.05 if hi > lo else 0.05
    diag = np.linspace(lo - margin, hi + margin, 100)
    ax.plot(diag, diag, "k--", linewidth=1, label="No change (y = x)")
    ax.legend(fontsize=11)
    ax.set_xlabel("Score Before", fontsize=12)
    ax.set_ylabel("Score After", fontsize=12)
    ax.set_title("Score Before vs. Score After (Probed Scenarios)", fontsize=14, fontweight="bold")
    sns.despine(top=True, right=True)
    plt.grid(visible=True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = output_dir / "score_scatter.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved score scatter -> %s", out)


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
    ax.set_xlabel("Number of Affected Agents per Probe", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Affected Agents per Probe", fontsize=14, fontweight="bold")
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

    categories = ["No probe", "Ego probe", "Non-ego probe"]
    x = np.arange(len(categories))
    bar_width = 0.8 / max(len(dataset_labels), 1)

    _fig, ax = plt.subplots(figsize=(9, 5))
    for i, label in enumerate(dataset_labels):
        df = dfs_by_dataset[label]
        total = max(len(df), 1)
        no_probe = int((df["probe_found"] == "no").sum())
        ego_probe = int(((df["probe_found"] == "yes") & (df["is_ego_agent"] == "yes")).sum())
        non_ego_probe = int(((df["probe_found"] == "yes") & (df["is_ego_agent"] == "no")).sum())
        counts = [no_probe, ego_probe, non_ego_probe]
        pcts = [100.0 * v / total for v in counts]
        offset = (i - len(dataset_labels) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, pcts, width=bar_width, label=f"{label} (n={total})", color=colors[label], alpha=0.8)
        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                str(count),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Probe Outcome Distribution by Dataset", fontsize=14, fontweight="bold")
    ax.legend(title="Dataset", fontsize=9)
    sns.despine(top=True, right=True)
    plt.grid(visible=True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = output_dir / "probe_outcomes_combined.png"
    plt.savefig(out, dpi=dpi)
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
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Before / After Distributions (Probed Scenarios)", fontsize=14, fontweight="bold")
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    ax.legend(title="Dataset — Metric", fontsize=8)
    plt.tight_layout()
    out = output_dir / "score_distributions_combined.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved combined score distributions -> %s", out)


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

    fig, axes = plt.subplots(1, n + 2, figsize=(6 * (n + 2), 4), sharey=True, sharex=True)

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
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Score", fontsize=10)
        ax.legend(fontsize=9)
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
        ax.set_title(col_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Score", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(visible=True, linestyle="--", alpha=0.4)
        sns.despine(ax=ax, top=True, right=True)

    axes[0].set_xlim(0, x_max)
    axes[0].set_ylabel("Density", fontsize=11)
    fig.suptitle("Score Distributions per Dataset and Combined (Probed Scenarios)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "score_distributions_grid_combined.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved score distributions grid -> %s", out)


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
    ax.set_xlabel("Score Delta (score_after - score_before)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Delta Distribution (Probed Scenarios)", fontsize=14, fontweight="bold")
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    ax.legend(title="Dataset", fontsize=9)
    plt.tight_layout()
    out = output_dir / "score_delta_density_combined.png"
    plt.savefig(out, dpi=dpi)
    plt.close()
    logger.info("Saved combined score delta density -> %s", out)
