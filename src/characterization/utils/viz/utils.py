import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.progress import track

from characterization.schemas import ScenarioScores
from characterization.utils.io_utils import from_pickle, get_logger

logger = get_logger(__name__)


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
    logger.info(f"Found {subset_size} rows between [{round(min_value, 2)} to {round(max_value, 2)}] for {key}")
    sample_size = min(sample_size, subset_size)
    return df_subset.sample(n=sample_size, random_state=seed)


def get_valid_scenario_ids(scenario_types: str, criteria: str, base_path: str) -> list[str]:
    scenario_lists = []
    for scenario_type, criteria in product(scenario_types, criteria):
        scenario_type_scores_path = os.path.join(base_path, f"{scenario_type}_{criteria}")
        scenario_type_scores_files = os.listdir(scenario_type_scores_path)
        scenario_lists.append(scenario_type_scores_files)
    scenario_ids = list(set.intersection(*[set(scenario_list) for scenario_list in scenario_lists]))
    return scenario_ids


def plot_histograms_from_dataframe(
    df: pd.DataFrame,
    output_filepath: str = "temp.png",
    dpi: int = 30,
    alpha: float = 0.5,
) -> None:
    """Plots overlapping histograms and density curves for each numeric column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing numeric data to plot.
        output_filepath (str): Path to save the output plot image.
        dpi (int): Dots per inch for the saved figure.
        alpha (float): Transparency level for the histograms (0 = transparent, 1 = solid).

    Raises:
        ValueError: If no numeric columns are found in the DataFrame.
    """
    # Select numeric columns, excluding the specified one
    columns_to_plot = df.select_dtypes(include="number").columns
    N = len(columns_to_plot)

    if N == 0:
        raise ValueError("No numeric columns to plot.")

    palette = sns.color_palette("husl", N)

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
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=dpi)
    plt.close()


def load_scores(
    scenario_ids: list[str],
    scores_path: str,
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
        filepath = os.path.join(scores_path, scenario_id)
        scores = from_pickle(filepath)  # nosec B301
        scores = ScenarioScores.model_validate(scores)
        scores_dict[scenario_id] = scores
    return scores_dict
