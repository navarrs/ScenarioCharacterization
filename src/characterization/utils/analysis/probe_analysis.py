"""Probe analysis utilities.

Helpers for loading, transforming, and visualizing the ``probe_summary.csv`` produced by
``run_scenario_probing``.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from characterization.processors.probe_processor import CSV_FIELDS
from characterization.utils.analysis.common_analysis import plot_histograms_from_dataframe
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
        palette=_PALETTE,
        inner="box",
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
