from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from characterization.utils.io_utils import get_logger
from characterization.utils.scenario_types import AgentPairType, AgentType

logger = get_logger(__name__)

SUPPORTED_FEATURES = ["individual", "interaction"]
FEATURE_COLOR_MAP = {
    "speed": "blue",
    "speed_limit_diff": "green",
    "acceleration": "orange",
    "deceleration": "red",
    "jerk": "purple",
    "waiting_period": "brown",
    "kalman_difficulty": "cyan",
    "collision": "olive",
    "mttcp": "magenta",
    "thw": "teal",
    "ttc": "navy",
    "drac": "coral",
}

AGENT_COLORS = {
    AgentType.TYPE_UNSET: "gray",
    AgentType.TYPE_VEHICLE: "slategray",
    AgentType.TYPE_PEDESTRIAN: "plum",
    AgentType.TYPE_CYCLIST: "forestgreen",
    AgentType.TYPE_OTHER: "gray",
    AgentType.TYPE_EGO_AGENT: "dodgerblue",
    AgentType.TYPE_RELEVANT: "coral",
    AgentPairType.TYPE_VEHICLE_VEHICLE: "slategray",
    AgentPairType.TYPE_VEHICLE_PEDESTRIAN: "plum",
    AgentPairType.TYPE_VEHICLE_CYCLIST: "forestgreen",
    AgentPairType.TYPE_PEDESTRIAN_PEDESTRIAN: "lightpink",
    AgentPairType.TYPE_PEDESTRIAN_CYCLIST: "mediumseagreen",
    AgentPairType.TYPE_CYCLIST_CYCLIST: "darkgreen",
    AgentPairType.TYPE_OTHER: "gray",
}

# Colors chosen to be visually distinct from all AGENT_COLORS hues (gray, slategray, plum, forestgreen,
# dodgerblue, coral, lightpink, mediumseagreen, darkgreen).
DATASET_COLORS: dict[str, str] = {
    "waymo": "#1B82E2",  # vivid red
    "nuscenes": "#52E2D1",  # vivid orange
    "argoverse2": "#EEBC47",  # medium purple
}

_DATASET_FALLBACK_PALETTE: tuple[str, ...] = (
    "#8C564B",  # brown
    "#BCBD22",  # olive/yellow-green
    "#17BECF",  # cyan
    "#E377C2",  # magenta-pink
)


def get_dataset_colors(dataset_labels: list[str]) -> dict[str, str]:
    """Returns a stable label → color mapping for multi-dataset plots.

    Known datasets (waymo, nuscenes, argoverse2) get fixed colors that do not overlap with
    AGENT_COLORS. Unknown labels cycle through _DATASET_FALLBACK_PALETTE.
    """
    result: dict[str, str] = {}
    fallback_iter = iter(_DATASET_FALLBACK_PALETTE)
    for label in dataset_labels:
        label_low = label.lower()
        result[label] = DATASET_COLORS[label_low] if label_low in DATASET_COLORS else next(fallback_iter, "#AAAAAA")
    return result


DEFAULT_FEATURE_CATEGORIES: list[dict[str, Any]] = [
    {"name": "LOW", "percentile_range": [0, 25]},
    {"name": "MEDIUM", "percentile_range": [25, 75]},
    {"name": "HIGH", "percentile_range": [75, 90]},
    {"name": "CRITICAL", "percentile_range": [90, 100]},
]


def get_sample_to_plot(
    df: pd.DataFrame,
    key: str,
    min_value: float,
    max_value: float,
    seed: int,
    sample_size: int,
) -> pd.DataFrame:
    """Selects a random sample of rows from a DataFrame within a specified value range for a given column.

    Args:
        df (pd.DataFrame): The DataFrame to sample from.
        key (str): The column name to filter by value range.
        min_value (float): The minimum value (inclusive) for filtering.
        max_value (float): The maximum value (exclusive) for filtering.
        seed (int): Random seed for reproducibility.
        sample_size (int): Number of samples to return.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled rows within the specified range.
    """
    df_subset = df.loc[(df[key] >= min_value) & (df[key] < max_value)]
    subset_size = len(df_subset)
    logger.info("Found %d rows between [%.2f to %.2f] for %s", subset_size, min_value, max_value, key)
    sample_size = min(sample_size, subset_size)
    return df_subset.sample(n=sample_size, random_state=seed)


def get_valid_scenario_ids(scenario_types: list[str], criteria: list[str], base_path: Path) -> list[str]:
    """Finds scenario IDs that are common across all specified scenario types and criteria.

    Args:
        scenario_types (list[str]): List of scenario types.
        criteria (list[str]): List of criteria.
        base_path (str): Base path where scenario score files are stored.

    Returns:
        scenario_ids (list[Path]): List of scenario IDs that are present in all specified scenario types and criteria.
    """
    scenario_lists = []
    for scenario_type, criterion in product(scenario_types, criteria):
        scenarios_path = base_path / f"{scenario_type}_{criterion}"
        scenario_files = [f.name for f in scenarios_path.iterdir()]
        scenario_lists.append(scenario_files)
    return list(set.intersection(*[set(scenario_list) for scenario_list in scenario_lists]))


def get_scored_scenario_ids(scenario_types: list[str], criteria: list[str], base_path: Path) -> dict[str, list[str]]:
    """Retrieves the list of scored scenario IDs for each combination of scenario type and criterion.

    Args:
        scenario_types (list[str]): List of scenario types to consider.
        criteria (list[str]): List of criteria to consider.
        base_path (Path): Base path where the scored scenario files are located.

    Returns:
        dict[str, list[str]]: A dictionary mapping each scenario type and criterion combination to a list of scored
            scenario IDs (file names).
    """
    scenario_lists = {}
    for scenario_type, criterion in product(scenario_types, criteria):
        key = f"{scenario_type}_{criterion}"
        scenario_type_criterion_path = base_path / key
        scenario_type_scores_files = [file.name for file in scenario_type_criterion_path.glob("*.pkl")]
        scenario_lists[key] = scenario_type_scores_files
    return scenario_lists


def plot_histograms_from_dataframe(
    df: pd.DataFrame,
    output_filepath: Path = Path("temp.png"),
    dpi: int = 30,
    alpha: float = 0.5,
) -> None:
    """Plots overlapping histograms and density curves for each numeric column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing numeric data to plot.
        output_filepath (Path): Path to save the output plot image.
        dpi (int): Dots per inch for the saved figure.
        alpha (float): Transparency level for the histograms (0 = transparent, 1 = solid).

    Raises:
        ValueError: If no numeric columns are found in the DataFrame.
    """
    # Select numeric columns, excluding the specified one
    columns_to_plot = df.select_dtypes(include="number").columns
    num_columns_to_plot = len(columns_to_plot)

    if num_columns_to_plot == 0:
        error_message = "No numeric columns found in the DataFrame to plot."
        raise ValueError(error_message)

    palette = sns.color_palette("husl", num_columns_to_plot)

    plt.figure(figsize=(10, 6))

    for i, col in enumerate(columns_to_plot):
        sns.histplot(
            data=df,
            x=col,
            color=palette[i],
            label=col,
            kde=True,
            stat="density",
            alpha=alpha,
            edgecolor="white",
        )

    sns.despine(top=True, right=True)

    plt.legend()
    plt.xlabel("Scores")
    plt.ylabel("Density")
    plt.title("Score Density Function over Scenarios")
    plt.grid(visible=True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=dpi)
    plt.close()


def compute_jaccard_index(set1: set[Any], set2: set[Any]) -> float:
    """Calculates the Jaccard index between two sets.

    Args:
        set1 (set): First set.
        set2 (set): Second set.

    Returns:
        float: Jaccard index between the two sets.
    """
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0


def compute_category_thresholds(data: NDArray[np.float64], categories: list[dict[str, Any]]) -> dict[str, float]:
    """Computes unique boundary thresholds between consecutive semantic categories.

    For each boundary, the nominal threshold is the upper-end percentile of the lower category. If that value equals the
    previous boundary, the range is scanned at 0.5-percentile steps to find the first strictly greater value.

    Returns:
        dict mapping "CAT_A/CAT_B" boundary labels to threshold values.
    """
    # Work on a local copy so pushing hi doesn't mutate the caller's category list.
    cats = [{"name": c["name"], "percentile_range": list(c["percentile_range"])} for c in categories]
    thresholds: dict[str, float] = {}
    prev: float | None = None
    for i in range(len(cats) - 1):
        cat, next_cat = cats[i], cats[i + 1]
        hi = cat["percentile_range"][1]
        next_hi = next_cat["percentile_range"][1]
        midpoint = (hi + next_hi) / 2
        label = f"{cat['name']}/{next_cat['name']}"
        value = float(np.round(np.percentile(data, hi), decimals=2))

        if prev is not None and value == prev:
            for p in np.arange(hi + 0.5, midpoint + 0.5, 0.5):
                candidate = float(np.round(np.percentile(data, p), decimals=2))
                if candidate > prev:
                    value = candidate
                    next_cat["percentile_range"][0] = p  # keep ranges contiguous
                    break
            else:
                logger.warning("No unique threshold found for boundary %s; using duplicate value %.4f", label, value)

        thresholds[label] = value
        prev = value
    return thresholds
