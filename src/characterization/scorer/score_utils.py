import os
from itertools import combinations, product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.progress import track

from characterization.schemas import ScenarioScores
from characterization.utils.io_utils import from_pickle, get_logger

logger = get_logger(__name__)

SUPPORTED_SCORERS = ["individual", "interaction", "safeshift"]


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
    df_subset = df[(df[key] >= min_value) & (df[key] < max_value)]
    subset_size = len(df_subset)
    logger.info("Found %d rows between [%.2f to %.2f] for %s", subset_size, min_value, max_value, key)
    sample_size = min(sample_size, subset_size)
    return df_subset.sample(n=sample_size, random_state=seed)


def get_valid_scenario_ids(scenario_types: list[str], criteria: list[str], base_path: str) -> list[str]:
    """Finds scenario IDs that are common across all specified scenario types and criteria.

    Args:
        scenario_types (list[str]): List of scenario types.
        criteria (list[str]): List of criteria.
        base_path (str): Base path where scenario score files are stored.

    Returns:
        scenario_ids (list[str]): List of scenario IDs that are present in all specified scenario types and criteria.
    """
    scenario_lists = []
    for scenario_type, criterion in product(scenario_types, criteria):
        scenario_type_scores_path = os.path.join(base_path, f"{scenario_type}_{criterion}")
        scenario_type_scores_files = os.listdir(scenario_type_scores_path)
        scenario_lists.append(scenario_type_scores_files)
    return list(set.intersection(*[set(scenario_list) for scenario_list in scenario_lists]))


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
            df[col],
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


def plot_score_vs_critical_time_heatmap_from_dataframe(
    df_scores: pd.DataFrame,
    df_critical_times: pd.DataFrame,
    output_filepath: Path = Path("temp.png"),
    dpi: int = 30,
    alpha: float = 0.8,
) -> None:
    """Plots overlapping histograms and density curves for each numeric column in a DataFrame.

    Args:
        df_scores (pd.DataFrame): DataFrame containing numeric score data to plot.
        df_critical_times (pd.DataFrame): DataFrame containing numeric critical time data to plot
        output_filepath (Path): Path to save the output plot image.
        dpi (int): Dots per inch for the saved figure.
        alpha (float): Transparency level for the histograms (0 = transparent, 1 = solid).

    Raises:
        ValueError: If no numeric columns are found in the DataFrame.
    """
    # Select numeric columns, excluding the specified one
    assert set(df_scores.columns[1:]) == set(df_critical_times.columns[1:]), (
        "df_scores and df_critical_times must have the same numeric columns to plot"
    )
    columns_to_plot = df_scores.columns[1:]
    num_columns_to_plot = len(columns_to_plot)

    if num_columns_to_plot == 0:
        error_message = "No numeric columns found in the DataFrame to plot."
        raise ValueError(error_message)

    plt.figure(figsize=(10, 6))

    _, axes = plt.subplots(1, num_columns_to_plot, figsize=(6 * num_columns_to_plot, 6), squeeze=False)
    x_bins = np.arange(0, 11, 2)  # Temporal binning
    for i, col in enumerate(columns_to_plot):
        # 2D histogram for heatmap data

        scores, critical_times = df_scores[col], df_critical_times[col]
        # Only concatenate if the values in the dataframes are numpy arrays
        if isinstance(scores.iloc[0], np.ndarray):
            # If each row is a numpy array, flatten all elements into a single column
            scores = np.concatenate(df_scores[col].values)
            critical_times = np.concatenate(df_critical_times[col].values)
            # Now each element is a new row, and scores/critical_times are 1D arrays

        critical_times = np.clip(critical_times, 0, 10)  # Clip to max 10s
        max_score = scores.max().round() + 1
        y_bins = np.arange(0, max_score, max_score / 10)  # Score binning
        heatmap_data, xedges, yedges = np.histogram2d(critical_times, scores, bins=[x_bins, y_bins])

        ax = axes[0, i]
        sns.heatmap(
            heatmap_data.T,
            ax=ax,
            cmap="rocket",
            cbar=True,
            xticklabels=[str(x) for x in np.round(xedges[:-1], 2)],
            yticklabels=[str(y) for y in np.round(yedges[:-1], 2)],
            alpha=alpha,
            annot=heatmap_data.T.astype(int),  # Show integer counts
            fmt="d",
        )
        ax.set_xlabel("Critical Times")
        ax.set_ylabel("Scores")
        ax.set_title(f"Scores vs Critical Times ({col})")
        # Invert y-axis so 0 is at the bottom
        ax.invert_yaxis()

    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=dpi)
    plt.close()


def load_scores(
    scenario_ids: list[str],
    scores_path: Path,
    prefix: str,
) -> dict[str, ScenarioScores]:
    """Loads scenario scores from the specified path and updates the scores DataFrame.

    Args:
        scenario_ids (list[str]): List of scenario IDs to load scores for.
        scores_path (str): Path to the directory containing score files.
        prefix (str): Prefix for the score files.

    Returns:
        dict[str, ScenarioScores]: Dictionary mapping scenario IDs to their corresponding ScenarioScores.
    """
    scores_dict = {}
    for scenario_id in track(scenario_ids, description=f"Loading {prefix} scores"):
        filepath = str(scores_path / scenario_id)
        scores = from_pickle(filepath)  # nosec B301
        scores = ScenarioScores.model_validate(scores)
        scores_dict[scenario_id] = scores
    return scores_dict


def load_scenario_scores(
    scenario_ids: list[str],
    scenario_types: list[str],
    scenario_scorers: list[str],
    criteria: list[str],
    scores_path: Path,
) -> tuple[dict[str, Any], ...]:
    """Loads scenario scores for given scenario types, scorers, and criteria.

    Args:
        scenario_ids (list[str]): List of scenario IDs to load scores for.
        scenario_types (list[str]): List of scenario types.
        scenario_scorers (list[str]): List of scenario scorers.
        criteria (list[str]): List of criteria.
        scores_path (Path): Path to the directory containing score files.

    Returns:
        Tuple containing three dictionaries:
            - scene_scores: Dictionary mapping score keys to lists of scene scores.
            - agent_scores: Dictionary mapping score keys to lists of agent scores.
            - scene_critical_times: Dictionary mapping score keys to lists of scene critical times.
            - agent_critical_times: Dictionary mapping score keys to lists of agent critical times.
            - scenario_scores: Dictionary mapping scenario IDs to their corresponding ScenarioScores.
    """
    scenario_scores = {}
    scene_scores = {"scenario_ids": scenario_ids}
    agent_scores = {"scenario_ids": scenario_ids}
    agent_critical_times = {"scenario_ids": scenario_ids}
    scene_critical_times = {"scenario_ids": scenario_ids}

    for scenario_type, scorer, criterion in product(scenario_types, scenario_scorers, criteria):
        key = f"{scenario_type}_{criterion}_{scorer}"
        scene_scores[key] = []
        scene_critical_times[key] = []
        agent_scores[key] = []
        agent_critical_times[key] = []

    for scenario_type, criterion in product(scenario_types, criteria):
        key = f"{scenario_type}_{criterion}"
        scenario_scores_path = scores_path / key
        scenario_scores[key] = load_scores(scenario_ids, scenario_scores_path, key)
        for scores in scenario_scores[key].values():
            for scorer in scenario_scorers:
                key = f"{scenario_type}_{criterion}_{scorer}"
                scores_key = f"{scorer}_scores"
                scene_scores[key].append(scores[scores_key].scene_score)
                scene_critical_times[key].append(scores[scores_key].scene_critical_time)
                agent_scores[key].append(scores[scores_key].agent_scores)
                agent_critical_times[key].append(scores[scores_key].agent_critical_times)
    return scene_scores, agent_scores, scene_critical_times, agent_critical_times, scenario_scores


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


def get_scenario_splits(
    scene_scores_df: pd.DataFrame, test_percentile: float, *, add_jaccard_index: bool = True
) -> dict[str, Any]:
    """Splits scenarios into in-distribution and out-of-distribution sets based on score percentiles.

    Args:
        scene_scores_df (pd.DataFrame): DataFrame containing scenario scores.
        test_percentile (float): Percentile threshold to define out-of-distribution scenarios.
        add_jaccard_index (bool): Whether to compute and include Jaccard indices between OOD sets of different scores.

    Returns:
        dict[str, Any]: Dictionary containing scenario splits for each score type.
    """
    scenario_splits = {}
    for key in scene_scores_df:
        if key == "scenario_ids":
            continue
        score_threshold = np.percentile(scene_scores_df[key], test_percentile)
        logger.info("Score value in the %s percentile for %s: %s", test_percentile, key, score_threshold)

        # Get scenario IDs below and above the score threshold for each score type
        in_distribution = scene_scores_df[scene_scores_df[key] < score_threshold]["scenario_ids"].tolist()
        out_of_distribution = scene_scores_df[scene_scores_df[key] >= score_threshold]["scenario_ids"].tolist()
        scenario_splits[key] = {
            "in_distribution": in_distribution,
            "out_of_distribution": out_of_distribution,
            "num_in_distribution": len(in_distribution),
            "num_out_of_distribution": len(out_of_distribution),
        }

    if add_jaccard_index:
        jaccard_indices = {}
        keys = list(scenario_splits.keys())
        pairwise_keys = list(combinations(keys, 2))
        for key1, key2 in pairwise_keys:
            logger.info("Calculating Jaccard index between %s and %s", key1, key2)
            key1_ood = set(scenario_splits[key1]["out_of_distribution"])
            key2_ood = set(scenario_splits[key2]["out_of_distribution"])
            jaccard_index = compute_jaccard_index(key1_ood, key2_ood)
            key = f"{key1}_{key2}_ood"
            jaccard_indices[key] = jaccard_index
            logger.info("Jaccard index for %s: %.4f", key, jaccard_index)
        scenario_splits["jaccard_indices"] = jaccard_indices

    return scenario_splits
