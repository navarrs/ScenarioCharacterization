"""Counts individual trajectories and interaction pairs per dataset from cached feature artifacts.

Both the "considered" and the "valid" population are recovered from a single pass over the feature
pickles: ``Individual.agent_types`` is the full per-scenario agent list while ``valid_idxs`` selects
the agents that survived feature computation, and ``Interaction.interaction_agent_types`` covers every
enumerated pair while ``interaction_status`` records why a pair was discarded.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer
from numpy.typing import NDArray
from tqdm import tqdm

from characterization.schemas import ScenarioFeatures
from characterization.utils.analysis.common_analysis import AGENT_COLORS, get_dataset_colors
from characterization.utils.common import InteractionStatus
from characterization.utils.io_utils import from_pickle, get_logger
from characterization.utils.scenario_types import AgentPairType, AgentType, get_agent_pair_type

logger = get_logger(__name__)

# A pair is "valid" when its features were computed. PARTIAL_INVALID_HEADING pairs have valid
# separation/intersection/collision/mTTCP values, so they are counted here exactly as they are counted
# by regroup_interaction_features.
VALID_INTERACTION_STATUSES = frozenset({InteractionStatus.COMPUTED_OK, InteractionStatus.PARTIAL_INVALID_HEADING})

COUNTED_AGENT_TYPES: tuple[AgentType, ...] = tuple(AgentType)
COUNTED_PAIR_TYPES: tuple[AgentPairType, ...] = tuple(AgentPairType)

# Short column headers used by the LaTeX table.
_LATEX_SHORT_NAMES: dict[str, str] = {
    "TYPE_VEHICLE": "V",
    "TYPE_PEDESTRIAN": "P",
    "TYPE_CYCLIST": "C",
    "TYPE_EGO_AGENT": "E",
    "TYPE_VEHICLE_VEHICLE": "V-V",
    "TYPE_VEHICLE_PEDESTRIAN": "V-P",
    "TYPE_VEHICLE_CYCLIST": "V-C",
    "TYPE_PEDESTRIAN_PEDESTRIAN": "P-P",
    "TYPE_PEDESTRIAN_CYCLIST": "P-C",
    "TYPE_CYCLIST_CYCLIST": "C-C",
}

# Under the ego pair scope every pair contains the ego agent, which get_agent_pair_type folds into the
# vehicle side. The pair-type columns are then ego-relative, so the headers name the partner agent.
_LATEX_EGO_SHORT_NAMES: dict[str, str] = {
    "TYPE_VEHICLE_VEHICLE": "E-V",
    "TYPE_VEHICLE_PEDESTRIAN": "E-P",
    "TYPE_VEHICLE_CYCLIST": "E-C",
}


def _set_theme() -> None:
    """Applies the shared plot theme used by the dataset figures."""
    sns.set_theme(
        style="whitegrid",
        font_scale=0.9,
        rc={
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
        },
    )


def _empty_counts() -> dict[str, int]:
    """Returns a counts dict with every agent-type and pair-type column initialized to zero."""
    counts = {"pairs_with_ego_total": 0, "pairs_with_ego_valid": 0}
    for agent_type in COUNTED_AGENT_TYPES:
        counts[f"agents_total_{agent_type.name}"] = 0
        counts[f"agents_valid_{agent_type.name}"] = 0
    for pair_type in COUNTED_PAIR_TYPES:
        counts[f"pairs_total_{pair_type.name}"] = 0
        counts[f"pairs_valid_{pair_type.name}"] = 0
    return counts


def count_scenario_features(features: ScenarioFeatures) -> dict[str, int]:
    """Counts trajectories and interaction pairs in a single scenario, by type and by validity.

    Args:
        features (ScenarioFeatures): Feature artifact for one scenario. Either half may be ``None`` if
            only individual or only interaction features were computed; missing halves count as zero.

    Returns:
        dict[str, int]: Flat counts keyed by ``agents_{total,valid}_<AGENT_TYPE>``,
            ``pairs_{total,valid}_<AGENT_PAIR_TYPE>``, and ``pairs_with_ego_{total,valid}``.
    """
    counts = _empty_counts()

    individual = features.individual_features
    if individual is not None and individual.agent_types is not None:
        agent_types = individual.agent_types
        for agent_type, count in Counter(agent_types).items():
            counts[f"agents_total_{agent_type.name}"] += count
        if individual.valid_idxs is not None:
            valid_types = Counter(agent_types[i] for i in individual.valid_idxs)
            for agent_type, count in valid_types.items():
                counts[f"agents_valid_{agent_type.name}"] += count

    interaction = features.interaction_features
    if interaction is not None and interaction.interaction_agent_types is not None:
        # interaction_agent_types and interaction_status are parallel arrays over every enumerated pair.
        statuses = interaction.interaction_status or [InteractionStatus.UNKNOWN] * len(
            interaction.interaction_agent_types
        )
        for (type_a, type_b), status in zip(interaction.interaction_agent_types, statuses, strict=True):
            pair_type = get_agent_pair_type(type_a, type_b)
            is_valid = status in VALID_INTERACTION_STATUSES
            # AgentPairType has no ego member -- get_agent_pair_type folds ego into vehicle -- so ego
            # involvement is tracked separately to keep it visible.
            has_ego = int(AgentType.TYPE_EGO_AGENT in (type_a, type_b))
            counts[f"pairs_total_{pair_type.name}"] += 1
            counts["pairs_with_ego_total"] += has_ego
            if is_valid:
                counts[f"pairs_valid_{pair_type.name}"] += 1
                counts["pairs_with_ego_valid"] += has_ego

    return counts


def count_dataset_features(
    scenario_ids: list[str],
    scenario_type: str,
    criterion: str,
    features_path: Path,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Accumulates per-scenario counts across a dataset, loading one feature artifact at a time.

    Streams rather than reusing ``load_scenario_features``, which retains every scenario's full feature
    set in memory. Only one ``(scenario_type, criterion)`` branch is counted; counting several would
    multiply every scenario.

    Args:
        scenario_ids (list[str]): Scenario file names to count, as returned by ``get_valid_scenario_ids``.
        scenario_type (str): Scenario type branch (e.g. ``gt``).
        criterion (str): Criterion branch (e.g. ``critical_continuous``).
        features_path (Path): Path to the directory containing the per-branch feature directories.

    Returns:
        tuple[dict[str, int], pd.DataFrame]: Accumulated counts plus ``num_scenarios_counted``, and one
            row of raw counts per scenario. The per-scenario rows are kept because these distributions
            are heavily right-skewed, so a dataset mean does not describe a typical scenario.
    """
    key = f"{scenario_type}_{criterion}"
    branch_path = features_path / key

    totals = _empty_counts()
    per_scenario: list[dict[str, int]] = []
    for scenario_id in tqdm(scenario_ids, desc=f"Counting {key} features"):
        features = from_pickle(str(branch_path / scenario_id))  # nosec B301
        counts = count_scenario_features(ScenarioFeatures.model_validate(features))
        per_scenario.append(counts)
        for column, value in counts.items():
            totals[column] += value

    totals["num_scenarios_counted"] = len(scenario_ids)
    return totals, pd.DataFrame(per_scenario)


def count_scenarios_on_disk(scenario_base_path: Path) -> int | None:
    """Counts scenario pickles present on disk, to reveal scenarios that feature computation skipped.

    Args:
        scenario_base_path (Path): Directory holding the raw scenario pickles.

    Returns:
        int | None: Number of ``.pkl`` files found, or ``None`` if the directory does not exist.
    """
    if not scenario_base_path.is_dir():
        logger.warning("Scenario path %s does not exist; skipping on-disk scenario count", scenario_base_path)
        return None
    return sum(1 for _ in scenario_base_path.rglob("*.pkl"))


def save_dataset_counts_json(counts_df: pd.DataFrame, output_dir: Path) -> None:
    """Writes the per-dataset counts to ``dataset_counts.json``, nested by dataset label.

    Args:
        counts_df (pd.DataFrame): One row per dataset, with a ``dataset`` column.
        output_dir (Path): Directory to write the JSON file.
    """
    summary = {row["dataset"]: {k: v for k, v in row.items() if k != "dataset"} for row in counts_df.to_dict("records")}
    output_filepath = output_dir / "dataset_counts.json"
    output_filepath.write_text(json.dumps(summary, indent=2))
    logger.info("Saved dataset counts -> %s", output_filepath)


THOUSAND = 1_000
TEN_THOUSAND = 10_000
MILLION = 1_000_000


def _format_count(value: int, *, compact: bool) -> str:
    """Renders a count as ``3.6M`` / ``300k`` / ``2.4k`` when *compact*, else with thousands separators.

    Values below ten thousand keep one decimal, so that e.g. 1,500 and 2,419 stay distinguishable.
    """
    if not compact or value < THOUSAND:
        return f"{value:,}"
    if value < TEN_THOUSAND:
        return f"{value / 1e3:.1f}k"
    if value < MILLION:
        return f"{value / 1e3:.0f}k"
    return f"{value / 1e6:.1f}M"


def _is_ego_scoped(counts_df: pd.DataFrame) -> bool:
    """Reports whether every row was computed under the ego pair scope.

    Under ``pair_scope="ego"`` every enumerated pair contains the ego agent, so the ego pair count equals
    the total across pair types. A mismatch means the feature cache holds all-pairs artifacts, and a
    mixture across rows means the caches disagree with each other.
    """
    pair_columns = [f"pairs_valid_{pair_type.name}" for pair_type in COUNTED_PAIR_TYPES]
    scoped = [
        int(row["pairs_with_ego_valid"]) == sum(int(row[column]) for column in pair_columns)
        for row in counts_df.to_dict("records")
    ]
    if any(scoped) and not all(scoped):
        all_pairs = [
            str(row["dataset"])
            for row, is_scoped in zip(counts_df.to_dict("records"), scoped, strict=True)
            if not is_scoped
        ]
        logger.warning("Datasets %s are not ego-scoped; labelling the whole table as all-pairs", all_pairs)
    return all(scoped)


def _format_cell(total: int, values: NDArray[np.int64] | None, *, compact: bool) -> str:
    """Renders one table cell as ``IQM (total)`` when per-scenario counts are given, else the total.

    The interquartile mean rather than the mean: these counts are right-skewed to different degrees per
    dataset, so a mean reports a heavier tail as if it were denser typical traffic. It is preferred over
    the median because averaging the middle half keeps resolution where a median collapses to an integer.
    Note that ``IQM * num_scenarios`` does not recover the total, which is why the total is printed too.
    """
    formatted_total = _format_count(total, compact=compact)
    if values is None or values.size == 0:
        return formatted_total
    low, high = np.percentile(values, [25, 75])
    middle = values[(values >= low) & (values <= high)]
    interquartile_mean = middle.mean() if middle.size else 0.0
    return f"{interquartile_mean:.1f} ({formatted_total})"


def write_latex_table(
    counts_df: pd.DataFrame,
    output_path: Path,
    agent_types: list[str],
    pair_types: list[str],
    labels: dict[str, str] | None = None,
    per_scenario_counts: dict[str, pd.DataFrame] | None = None,
    *,
    compact_numbers: bool = True,
) -> None:
    r"""Writes a booktabs table of valid trajectory and interaction counts, one row per dataset.

    Hand-rolled rather than ``DataFrame.to_latex`` because the layout needs ``\multicolumn`` group
    headers and ``\cmidrule`` spans. Requires ``booktabs``, ``graphicx`` and, when per-scenario counts
    are given, ``multirow`` in the document preamble.

    Args:
        counts_df (pd.DataFrame): One row per dataset, as written to ``dataset_counts.csv``.
        output_path (Path): File to write the LaTeX fragment to.
        agent_types (list[str]): ``AgentType`` names to render as trajectory columns.
        pair_types (list[str]): ``AgentPairType`` names to render as interaction columns.
        labels (dict[str, str] | None): Optional dataset label -> LaTeX row label (e.g. a ``\cite``
            macro). Datasets absent from the mapping fall back to their plain label.
        per_scenario_counts (dict[str, pd.DataFrame] | None): Dataset label -> per-scenario counts, as
            returned by ``count_dataset_features``. When given, cells read ``IQM (total)``: the
            interquartile mean keeps pools of different sizes comparable without letting the right skew
            in these distributions inflate one dataset more than another, and the total preserves scale.
            When omitted, cells carry the total alone.
        compact_numbers (bool): Render counts as ``300k`` / ``3.6M`` instead of ``300,000``.
    """
    labels = labels or {}
    per_scenario_counts = per_scenario_counts or {}
    ego_scoped = _is_ego_scoped(counts_df)
    num_agent_cols, num_pair_cols = len(agent_types), len(pair_types)
    # The scenario count is stated in the surrounding text rather than given a column, which leaves each
    # cell enough width to carry a central estimate and the dataset total at full size.
    agent_span = (2, 1 + num_agent_cols)
    pair_span = (2 + num_agent_cols, 1 + num_agent_cols + num_pair_cols)

    pair_names = {**_LATEX_SHORT_NAMES, **_LATEX_EGO_SHORT_NAMES} if ego_scoped else _LATEX_SHORT_NAMES
    pair_group = "Ego interactions" if ego_scoped else "Interactions"
    headers = [_LATEX_SHORT_NAMES.get(name, name) for name in agent_types]
    headers += [pair_names.get(name, name) for name in pair_types]
    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        r"    \small",
        # Shrink to fit the column but never scale up, so a narrow table is not blown up into the margins.
        r"    \resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{%",
        rf"    \begin{{tabular}}{{l{'c' * (num_agent_cols + num_pair_cols)}}}",
        r"        \toprule",
        rf"        & \multicolumn{{{num_agent_cols}}}{{c}}{{\textbf{{Trajectories}}}} "
        rf"& \multicolumn{{{num_pair_cols}}}{{c}}{{\textbf{{{pair_group}}}}} \\",
        rf"        \cmidrule(lr){{{agent_span[0]}-{agent_span[1]}}} "
        rf"\cmidrule(lr){{{pair_span[0]}-{pair_span[1]}}}",
        r"        \textbf{Dataset} & " + " & ".join(headers) + r" \\",
        r"        \midrule",
    ]

    typed_columns = [
        f"{prefix}{name}"
        for prefix, names in (("agents_valid_", agent_types), ("pairs_valid_", pair_types))
        for name in names
    ]
    for row in counts_df.to_dict("records"):
        dataset = str(row["dataset"])
        label = labels.get(dataset, dataset)
        scenario_counts = per_scenario_counts.get(dataset)
        cells = [
            _format_cell(
                int(row[column]),
                None if scenario_counts is None else np.asarray(scenario_counts[column]),
                compact=compact_numbers,
            )
            for column in typed_columns
        ]
        lines.append(f"        {label} & " + " & ".join(cells) + r" \\")

    scope_sentence = (
        "Every pair contains the ego agent, so interaction columns name the partner agent type. "
        if ego_scoped
        else "The ego agent is folded into the vehicle side of each pair type. "
    )
    normalization_sentence = (
        "Cells report the per-scenario interquartile mean, the mean over the middle half of scenarios, "
        "with the dataset total in parentheses. "
    )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    }",
        r"    \caption{Valid individual trajectories and interaction counts per dataset. "
        + (normalization_sentence if per_scenario_counts else "")
        + scope_sentence
        + r"The ego agent is counted in its own trajectory column where present.}",
        r"    \label{tab:experimental_setup}",
        r"\end{table}",
        "",
    ]
    output_path.write_text("\n".join(lines))
    logger.info("Saved LaTeX counts table -> %s", output_path)


def _palette_by_label(counts_df: pd.DataFrame) -> dict[str, str]:
    """Maps each dataset's display label to the color of its canonical ``dataset_name``.

    Colors are resolved from ``dataset_name`` rather than from ``dataset`` so that a dataset keeps the
    same color as in the feature and score analyses no matter what display label it is given. Falls back
    to the label when ``dataset_name`` is absent (e.g. a CSV written before that column existed).
    """
    labels = [str(label) for label in counts_df["dataset"]]
    names = [str(name) for name in counts_df["dataset_name"]] if "dataset_name" in counts_df else labels
    colors_by_name = get_dataset_colors(names)
    return {label: colors_by_name[name] for label, name in zip(labels, names, strict=True)}


def _long_form(counts_df: pd.DataFrame, prefix: str, type_names: list[str]) -> pd.DataFrame:
    """Reshapes the wide counts table into ``dataset`` / ``type`` / ``count`` rows for seaborn."""
    records = [
        {
            "dataset": str(row["dataset"]),
            "type": _LATEX_SHORT_NAMES.get(name, name),
            "count": int(row[f"{prefix}{name}"]),
        }
        for row in counts_df.to_dict("records")
        for name in type_names
    ]
    return pd.DataFrame(records)


def plot_dataset_counts(
    counts_df: pd.DataFrame,
    output_dir: Path,
    agent_types: list[str],
    pair_types: list[str],
    dpi: int = 300,
) -> None:
    """Plots valid trajectory and interaction counts as grouped bars, one hue per dataset.

    Args:
        counts_df (pd.DataFrame): One row per dataset, as written to ``dataset_counts.csv``.
        output_dir (Path): Directory to save the output plots.
        agent_types (list[str]): ``AgentType`` names to plot on the trajectory chart.
        pair_types (list[str]): ``AgentPairType`` names to plot on the interaction chart.
        dpi (int): Dots per inch for the saved figures.
    """
    _set_theme()
    palette = _palette_by_label(counts_df)

    for prefix, type_names, title, filename in [
        ("agents_valid_", agent_types, "Valid Trajectories per Dataset", "trajectory_counts.png"),
        ("pairs_valid_", pair_types, "Valid Interactions per Dataset", "interaction_counts.png"),
    ]:
        df = _long_form(counts_df, prefix, type_names)
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.barplot(data=df, x="type", y="count", hue="dataset", palette=palette, ax=ax)
        # Counts span several orders of magnitude across agent types.
        ax.set_yscale("log")
        for container in ax.containers:
            if isinstance(container, BarContainer):
                ax.bar_label(container, fmt=lambda v: _format_count(int(v), compact=True), fontsize=7, padding=2)

        sns.despine(top=True, right=True)
        ax.set_xlabel("")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(title)
        ax.grid(visible=True, linestyle="--", alpha=0.4)
        ax.legend(title="Dataset", fontsize=8)

        plt.tight_layout()
        output_filepath = output_dir / filename
        plt.savefig(output_filepath, dpi=dpi)
        plt.close()
        logger.info("Saved %s -> %s", title.lower(), output_filepath)


def plot_dataset_distributions(
    per_scenario_counts: dict[str, pd.DataFrame],
    output_dir: Path,
    agent_types: list[str],
    pair_types: list[str],
    dpi: int = 300,
    dataset_names: dict[str, str] | None = None,
) -> None:
    """Plots the per-scenario count distributions as boxes, showing the spread the table summarizes.

    The counts are right-skewed and skewed to different degrees across datasets, so a mean or even a
    median alone understates how concentrated the population is. Log y-axis, since the tails run one to
    two orders of magnitude above the median.

    Args:
        per_scenario_counts (dict[str, pd.DataFrame]): Dataset label -> per-scenario counts.
        output_dir (Path): Directory to save the output plots.
        agent_types (list[str]): ``AgentType`` names to plot on the trajectory chart.
        pair_types (list[str]): ``AgentPairType`` names to plot on the interaction chart.
        dpi (int): Dots per inch for the saved figures.
        dataset_names (dict[str, str] | None): Dataset label -> canonical name, so colors match the
            other analyses. Labels fall back to themselves when absent.
    """
    if not per_scenario_counts:
        logger.warning("No per-scenario counts given; skipping distribution plots")
        return

    _set_theme()
    labels = list(per_scenario_counts)
    names = {label: (dataset_names or {}).get(label, label) for label in labels}
    colors_by_name = get_dataset_colors(list(names.values()))
    palette = {label: colors_by_name[name] for label, name in names.items()}
    # Same invariant the table uses: under the ego scope every pair contains the ego agent.
    ego_scoped = all(
        int(frame["pairs_with_ego_valid"].sum())
        == sum(int(frame[f"pairs_valid_{pair_type.name}"].sum()) for pair_type in COUNTED_PAIR_TYPES)
        for frame in per_scenario_counts.values()
    )
    pair_names = {**_LATEX_SHORT_NAMES, **_LATEX_EGO_SHORT_NAMES} if ego_scoped else _LATEX_SHORT_NAMES

    for prefix, type_names, short_names, title, filename in [
        ("agents_valid_", agent_types, _LATEX_SHORT_NAMES, "Trajectories per Scenario", "trajectory_distribution.png"),
        ("pairs_valid_", pair_types, pair_names, "Interactions per Scenario", "interaction_distribution.png"),
    ]:
        records = [
            {"dataset": label, "type": short_names.get(name, name), "count": int(value)}
            for label, frame in per_scenario_counts.items()
            for name in type_names
            for value in frame[f"{prefix}{name}"]
        ]
        df = pd.DataFrame(records)

        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.boxplot(data=df, x="type", y="count", hue="dataset", palette=palette, ax=ax, showfliers=False)
        # Counts span two orders of magnitude but bottom out at zero, so symlog keeps the sparse types
        # visible where a log axis would drop them. Clamp the bottom, or symlog reserves its negative half.
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0, top=float(df["count"].max()) * 1.2)

        sns.despine(top=True, right=True)
        ax.set_xlabel("")
        ax.set_ylabel("Count per scenario (symlog scale)")
        ax.set_title(title)
        ax.grid(visible=True, axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="Dataset", fontsize=8)

        plt.tight_layout()
        output_filepath = output_dir / filename
        plt.savefig(output_filepath, dpi=dpi)
        plt.close()
        logger.info("Saved %s -> %s", title.lower(), output_filepath)


def plot_dataset_composition(
    counts_df: pd.DataFrame,
    output_dir: Path,
    agent_types: list[str],
    pair_types: list[str],
    dpi: int = 300,
) -> None:
    """Plots each dataset's class mix as 100%-stacked bars, making imbalance comparable across datasets.

    Shares are taken over the selected types only, so excluding a type also removes it from the
    denominator.

    Args:
        counts_df (pd.DataFrame): One row per dataset, as written to ``dataset_counts.csv``.
        output_dir (Path): Directory to save the output plots.
        agent_types (list[str]): ``AgentType`` names to stack on the trajectory chart.
        pair_types (list[str]): ``AgentPairType`` names to stack on the interaction chart.
        dpi (int): Dots per inch for the saved figures.
    """
    _set_theme()
    dataset_labels = [str(label) for label in counts_df["dataset"]]

    for prefix, type_names, enum_cls, title, filename in [
        ("agents_valid_", agent_types, AgentType, "Trajectory Composition", "trajectory_composition.png"),
        ("pairs_valid_", pair_types, AgentPairType, "Interaction Composition", "interaction_composition.png"),
    ]:
        counts = np.asarray(
            [[int(row[f"{prefix}{name}"]) for name in type_names] for row in counts_df.to_dict("records")],
            dtype=np.float64,
        )
        totals = counts.sum(axis=1, keepdims=True)
        shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) * 100.0

        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        bottom = np.zeros(len(dataset_labels))
        for column, name in enumerate(type_names):
            color = AGENT_COLORS.get(enum_cls[name], "gray")
            bars = ax.bar(
                dataset_labels,
                shares[:, column],
                bottom=bottom,
                color=color,
                label=_LATEX_SHORT_NAMES.get(name, name),
                edgecolor="white",
            )
            ax.bar_label(bars, fmt=lambda v: f"{v:.1f}%" if v >= 1.0 else "", label_type="center", fontsize=7)
            bottom += shares[:, column]

        sns.despine(top=True, right=True)
        ax.set_xlabel("")
        ax.set_ylabel("Share of selected types (%)")
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.grid(visible=True, axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="Type", fontsize=8)

        plt.tight_layout()
        output_filepath = output_dir / filename
        plt.savefig(output_filepath, dpi=dpi)
        plt.close()
        logger.info("Saved %s -> %s", title.lower(), output_filepath)
