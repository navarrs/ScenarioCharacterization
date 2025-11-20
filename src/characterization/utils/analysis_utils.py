import json
import os
from itertools import combinations, product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from rich.progress import track

from characterization.schemas import Individual, ScenarioFeatures, ScenarioScores
from characterization.utils.ad_types import AgentType
from characterization.utils.io_utils import from_pickle, get_logger

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
            - scenario_scores: Dictionary mapping scenario IDs to their corresponding ScenarioScores.
    """
    scenario_scores = {}
    scene_scores = {"scenario_ids": scenario_ids}
    agent_scores = {"scenario_ids": scenario_ids}

    for scenario_type, scorer, criterion in product(scenario_types, scenario_scorers, criteria):
        key = f"{scenario_type}_{criterion}_{scorer}"
        scene_scores[key] = []
        agent_scores[key] = []

    for scenario_type, criterion in product(scenario_types, criteria):
        key = f"{scenario_type}_{criterion}"
        scenario_scores_path = scores_path / key
        scenario_scores[key] = load_scores(scenario_ids, scenario_scores_path, key)
        for scores in scenario_scores[key].values():
            for scorer in scenario_scorers:
                key = f"{scenario_type}_{criterion}_{scorer}"
                scores_key = f"{scorer}_scores"
                scene_scores[key].append(scores[scores_key].scene_score)
                agent_scores[key].append(scores[scores_key].agent_scores)
    return scene_scores, agent_scores, scenario_scores


def load_features(
    scenario_ids: list[str],
    features_path: Path,
    prefix: str,
) -> dict[str, ScenarioFeatures]:
    """Loads scenario features from the specified path and updates the features DataFrame.

    Args:
        scenario_ids (list[str]): List of scenario IDs to load features for.
        features_path (Path): Path to the directory containing feature files.
        prefix (str): Prefix for the feature files.

    Returns:
        dict[str, ScenarioFeatures]: Dictionary mapping scenario IDs to their corresponding ScenarioFeatures.
    """
    features_dict = {}
    for scenario_id in track(scenario_ids, description=f"Loading {prefix} features"):
        filepath = str(features_path / scenario_id)
        features = from_pickle(filepath)  # nosec B301
        features = ScenarioFeatures.model_validate(features)
        features_dict[scenario_id] = features
    return features_dict


def load_scenario_features(
    scenario_ids: list[str],
    scenario_types: list[str],
    criteria: list[str],
    features_path: Path,
) -> tuple[dict[str, Any], ...]:
    """Loads scenario features for given scenario types and criteria.

    Args:
        scenario_ids (list[str]): List of scenario IDs to load scores for.
        scenario_types (list[str]): List of scenario types.
        criteria (list[str]): List of criteria.
        features_path (Path): Path to the directory containing feature files.

    Returns:
        Tuple of dictionaries containing scenario features for each scenario type and criterion.
    """
    individual_features = {"scenario_ids": [], "features": []}
    interaction_features = {"scenario_ids": [], "features": []}
    for scenario_type, criterion in product(scenario_types, criteria):
        key = f"{scenario_type}_{criterion}"
        scenario_features_path = features_path / key
        features = load_features(scenario_ids, scenario_features_path, key)
        for scenario_id, feat in features.items():
            if feat.individual_features is not None:
                individual_features["scenario_ids"].append(scenario_id)
                individual_features["features"].append(feat.individual_features)  # pyright: ignore[reportArgumentType]
            if feat.interaction_features is not None:
                interaction_features["scenario_ids"].append(scenario_id)
                interaction_features["features"].append(feat.interaction_features)  # pyright: ignore[reportArgumentType]
    return individual_features, interaction_features


def regroup_individual_features(individual_features: dict[str, Any]) -> dict[AgentType, Any]:
    """Regroups individual features by agent type.

    Args:
        individual_features (dict[str, Any]): Dictionary containing individual features with scenario IDs.

    Returns:
        dict[AgentType, Any]: Dictionary mapping each AgentType to its corresponding features.
    """

    def _init_empty() -> dict[str, list[float]]:
        """Initializes an empty feature dictionary for each agent type."""
        return {
            "speed": [],
            "speed_limit_diff": [],
            "acceleration": [],
            "deceleration": [],
            "jerk": [],
            "waiting_period": [],
            "kalman_difficulty": [],
        }

    def _extend_features(
        feature_dict: dict[AgentType, Any], feature: Individual, mask: NDArray[np.bool_], agent_type: AgentType
    ) -> None:
        """Extends features to the corresponding agent type in the feature dictionary."""
        if feature.speed is not None:
            feature_dict[agent_type]["speed"].extend(feature.speed[mask].tolist())
        if feature.speed_limit_diff is not None:
            feature_dict[agent_type]["speed_limit_diff"].extend(feature.speed_limit_diff[mask].tolist())
        if feature.acceleration is not None:
            feature_dict[agent_type]["acceleration"].extend(feature.acceleration[mask].tolist())
        if feature.deceleration is not None:
            feature_dict[agent_type]["deceleration"].extend(feature.deceleration[mask].tolist())
        if feature.jerk is not None:
            feature_dict[agent_type]["jerk"].extend(feature.jerk[mask].tolist())
        if feature.waiting_period is not None:
            feature_dict[agent_type]["waiting_period"].extend(feature.waiting_period[mask].tolist())
        if feature.kalman_difficulty is not None:
            feature_dict[agent_type]["kalman_difficulty"].extend(feature.kalman_difficulty[mask].tolist())

    regrouped_features = {
        AgentType.TYPE_VEHICLE: _init_empty(),
        AgentType.TYPE_CYCLIST: _init_empty(),
        AgentType.TYPE_PEDESTRIAN: _init_empty(),
    }

    for _, features in zip(individual_features["scenario_ids"], individual_features["features"], strict=False):
        agent_types = np.asarray([features.agent_types[i] for i in features.valid_idxs])

        # Regroup vehicle features
        vehicle_mask = agent_types == AgentType.TYPE_VEHICLE
        _extend_features(regrouped_features, features, vehicle_mask, AgentType.TYPE_VEHICLE)

        # Regroup cyclist features
        cyclist_mask = agent_types == AgentType.TYPE_CYCLIST
        _extend_features(regrouped_features, features, cyclist_mask, AgentType.TYPE_CYCLIST)

        # Regroup pedestrian features
        pedestrian_mask = agent_types == AgentType.TYPE_PEDESTRIAN
        _extend_features(regrouped_features, features, pedestrian_mask, AgentType.TYPE_PEDESTRIAN)

    for key in regrouped_features:  # noqa: PLC0206
        for feature_name in regrouped_features[key]:
            regrouped_features[key][feature_name] = np.array(  # pyright: ignore[reportArgumentType]
                regrouped_features[key][feature_name], dtype=np.float32
            )
    return regrouped_features


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


def plot_feature_distributions(
    feature_data: dict[AgentType, Any],
    output_dir: Path,
    dpi: int = 100,
    percentile_values: list[int] = [1, 10, 25, 50, 75, 90, 95, 99],  # noqa: B006
) -> None:
    """Plots the distribution of a feature using a histogram and density curve.

    Args:
        feature_data (NDArray[np.float32]): Array of feature values to plot.
        output_dir (Path): Directory to save the output plots.
        dpi (int): Dots per inch for the saved figure.
        percentile_values (list[int]): List of percentiles to compute and display on the plot.
    """
    feature_percentiles = {}
    for agent_type, features in feature_data.items():
        num_features = len(features)
        _, axs = plt.subplots(1, num_features, figsize=(8 * num_features, 6))

        for feature_name, feature_values in features.items():
            ax = axs if num_features == 1 else axs[list(features.keys()).index(feature_name)]
            sns.histplot(
                feature_values,
                color=FEATURE_COLOR_MAP.get(feature_name, "gray"),
                kde=True,
                stat="density",
                alpha=0.6,
                edgecolor="white",
                ax=ax,
            )

            sns.despine(top=True, right=True)

            ax.set_xlabel(f"{feature_name} values")
            ax.set_ylabel("Density")
            ax.set_title(f"{feature_name} Distribution ({feature_values.shape[0]} samples)")
            ax.grid(visible=True, linestyle="--", alpha=0.4)

            percentiles = np.percentile(feature_values, percentile_values)
            feature_percentiles[feature_name] = dict(zip(percentile_values, percentiles, strict=False))
            for p, v in zip(percentile_values, percentiles, strict=False):
                ax.axvline(v, color="black", linestyle="--", alpha=0.6)
                ax.text(v, ax.get_ylim()[1] * 0.9, f"{p}th: {v:.2f}", rotation=90, verticalalignment="center")

        plt.tight_layout()
        output_filepath = output_dir / f"{agent_type.name.lower()}_distributions.png"
        plt.savefig(output_filepath, dpi=dpi)
        plt.close()

        output_filepath = output_dir / f"{agent_type.name.lower()}_feature_percentiles.json"
        with open(output_filepath, "w") as f:
            json.dump(feature_percentiles, f, indent=4)


def plot_agent_scores_distributions(
    agent_scores: dict[str, Any],
    output_dir: Path,
    dpi: int = 100,
    percentile_values: list[int] = [10, 25, 50, 75, 90, 95, 99],  # noqa: B006
) -> None:
    """Plots the distribution of agent scores using histograms and density curves.

    Args:
        agent_scores (dict[str, Any]): Dictionary containing agent scores with scenario IDs.
        output_dir (Path): Directory to save the output plots.
        dpi (int): Dots per inch for the saved figure.
        percentile_values (list[int]): List of percentiles to compute and display on the plot.
    """
    for key, values in agent_scores.items():
        if key == "scenario_ids":
            continue

        agent_scores_flattened = []
        for scores in values:
            agent_scores_flattened.extend(scores.tolist())

        _, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            agent_scores_flattened,
            color="blue",
            kde=True,
            stat="density",
            alpha=0.6,
            edgecolor="white",
            ax=ax,
        )

        sns.despine(top=True, right=True)

        ax.set_xlabel("Scores values")
        ax.set_ylabel("Density")
        ax.set_title(f"Scores Distribution ({len(agent_scores_flattened)} agents)")
        ax.grid(visible=True, linestyle="--", alpha=0.4)

        percentiles = np.percentile(agent_scores_flattened, percentile_values).tolist()
        score_percentiles = dict(zip(percentile_values, percentiles, strict=False))
        for p, v in score_percentiles.items():
            ax.axvline(float(v), color="black", linestyle="--", alpha=0.6)
            ax.text(float(v), ax.get_ylim()[1] * 0.9, f"{p}th: {v:.2f}", rotation=90, verticalalignment="center")

        plt.tight_layout()
        output_filepath = output_dir / f"agent_score_distribution_{key}.png"
        plt.savefig(output_filepath, dpi=dpi)
        plt.close()

        output_filepath = output_dir / f"{key}.json"
        with open(output_filepath, "w") as f:
            json.dump(score_percentiles, f, indent=4)
