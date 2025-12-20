import json
from itertools import combinations, product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from numpy.typing import NDArray
from tqdm import tqdm

from characterization.schemas import Individual, Interaction, ScenarioFeatures, ScenarioScores
from characterization.utils.common import BIG_EPS, SMALL_EPS, InteractionStatus
from characterization.utils.io_utils import from_pickle, get_logger
from characterization.utils.scenario_types import AgentPairType, AgentType, get_agent_pair_type

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


def load_scores(scenario_ids: list[str], scores_path: Path, prefix: str) -> dict[str, ScenarioScores]:
    """Loads scenario scores from the specified path and updates the scores DataFrame.

    Args:
        scenario_ids (list[str]): List of scenario IDs to load scores for.
        scores_path (str): Path to the directory containing score files.
        prefix (str): Prefix for the score files.

    Returns:
        dict[str, ScenarioScores]: Dictionary mapping scenario IDs to their corresponding ScenarioScores.
    """
    scores_dict = {}
    for scenario_id in tqdm(scenario_ids, f"Loading {prefix} scores"):
        filename = str(scores_path / scenario_id)
        scores = from_pickle(filename)  # nosec B301
        scores = ScenarioScores.model_validate(scores)
        scores_dict[scenario_id] = scores
    return scores_dict


def load_scenario_scores(
    scenario_ids: list[str],
    scenario_types: list[str],
    criteria: list[str],
    scores_path: Path,
) -> dict[str, dict[str, ScenarioScores]]:
    """Loads scenario scores for given scenario types, scorers, and criteria.

    Args:
        scenario_ids (list[str]): List of scenario IDs to load scores for.
        scenario_types (list[str]): List of scenario types.
        criteria (list[str]): List of criteria.
        scores_path (Path): Path to the directory containing score files.

    Returns:
        dict[str, dict[str, ScenarioScores]]: Dictionary mapping scenario type and criterion keys to their corresponding
            ScenarioScores.
    """
    scenario_scores = {}
    for scenario_type, criterion in product(scenario_types, criteria):
        key = f"{scenario_type}_{criterion}"
        scenario_scores_path = scores_path / key
        scenario_scores[key] = load_scores(scenario_ids, scenario_scores_path, key)
    return scenario_scores


def regroup_scenario_scores(
    scenario_scores: dict[str, dict[str, ScenarioScores]],
    scenario_ids: list[str],
    scenario_types: list[str],
    scenario_scorers: list[str],
    criteria: list[str],
) -> tuple[dict[str, Any], ...]:
    """Loads scenario scores for given scenario types, scorers, and criteria.

    Args:
        scenario_scores (dict[str, dict[str, ScenarioScores]]): Dictionary containing scenario scores.
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
    scene_scores = {"scenario_ids": scenario_ids}
    agent_scores = {"scenario_ids": scenario_ids}
    agent_scores_valid = {"scenario_ids": scenario_ids}

    for scenario_type, scorer, criterion in product(scenario_types, scenario_scorers, criteria):
        key = f"{scenario_type}_{criterion}_{scorer}"
        scene_scores[key] = []
        agent_scores[key] = []
        agent_scores_valid[key] = []

    for scenario_type, criterion in product(scenario_types, criteria):
        key = f"{scenario_type}_{criterion}"
        for scores in scenario_scores[key].values():
            for scorer in scenario_scorers:
                key = f"{scenario_type}_{criterion}_{scorer}"
                scores_key = f"{scorer}_scores"
                scene_score = scores[scores_key].scene_score
                if scene_score is not None:
                    scene_scores[key].append(scene_score)
                    agent_scores[key].append(scores[scores_key].agent_scores)
                    agent_scores_valid[key].append(scores[scores_key].agent_scores_valid)
    return scene_scores, agent_scores, agent_scores_valid


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
    for scenario_id in tqdm(scenario_ids, desc=f"Loading {prefix} features"):
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
            kalman_difficulty = feature.kalman_difficulty[mask]
            kalman_difficulty = kalman_difficulty[kalman_difficulty >= 0]  # Filter out negative values
            feature_dict[agent_type]["kalman_difficulty"].extend(kalman_difficulty.tolist())

    regrouped_features = {
        AgentType.TYPE_VEHICLE: _init_empty(),
        AgentType.TYPE_CYCLIST: _init_empty(),
        AgentType.TYPE_PEDESTRIAN: _init_empty(),
    }

    for _, features in zip(individual_features["scenario_ids"], individual_features["features"], strict=False):
        # Only consider the agent types for valid indeces, since we only computed features for valid agents.
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


def regroup_interaction_features(interaction_features: dict[str, Any]) -> dict[AgentPairType, Any]:
    """Regroups interaction features by agent type.

    Args:
        interaction_features (dict[str, Any]): Dictionary containing interaction features with scenario IDs.

    Returns:
        dict[AgentType, Any]: Dictionary mapping each AgentType to its corresponding features.
    """

    def _init_empty() -> dict[str, list[float]]:
        """Initializes an empty feature dictionary for each agent type."""
        return {
            "inv_separation": [],
            "separation": [],
            "intersection": [],
            "collision": [],
            "inv_mttcp": [],
            "mttcp": [],
            "inv_thw": [],
            "thw": [],
            "inv_ttc": [],
            "ttc": [],
            "drac": [],
        }

    def _append_feature(
        feature_dict: dict[AgentPairType, Any], feature: Interaction, index: int, agent_pair: AgentPairType
    ) -> None:
        """Extends features to the corresponding agent type in the feature dictionary."""
        if feature.separation is not None:
            feature_dict[agent_pair]["inv_separation"].append(1 / (feature.separation[index] + SMALL_EPS))
            feature_dict[agent_pair]["separation"].append(feature.separation[index])
        if feature.intersection is not None:
            feature_dict[agent_pair]["intersection"].append(feature.intersection[index])
        if feature.collision is not None:
            feature_dict[agent_pair]["collision"].append(feature.collision[index])
        if feature.mttcp is not None:
            feature_dict[agent_pair]["mttcp"].append(feature.mttcp[index])
        if feature.inv_mttcp is not None:
            feature_dict[agent_pair]["inv_mttcp"].append(feature.inv_mttcp[index])
        if feature.thw is not None:
            feature_dict[agent_pair]["thw"].append(feature.thw[index])
        if feature.inv_thw is not None:
            feature_dict[agent_pair]["inv_thw"].append(feature.inv_thw[index])
        if feature.ttc is not None:
            feature_dict[agent_pair]["ttc"].append(feature.ttc[index])
        if feature.inv_ttc is not None:
            feature_dict[agent_pair]["inv_ttc"].append(feature.inv_ttc[index])
        if feature.drac is not None:
            feature_dict[agent_pair]["drac"].append(feature.drac[index])

    regrouped_features = {
        agent_pair: _init_empty() for agent_pair in AgentPairType if agent_pair != AgentPairType.TYPE_UNSET
    }

    for _, features in zip(interaction_features["scenario_ids"], interaction_features["features"], strict=False):
        interaction_agent_types = features.interaction_agent_types
        interaction_status = features.interaction_status
        if interaction_agent_types is None or interaction_status is None:
            continue

        for i, (agent_types, status) in enumerate(zip(interaction_agent_types, interaction_status, strict=False)):
            if status not in [InteractionStatus.COMPUTED_OK, InteractionStatus.PARTIAL_INVALID_HEADING]:
                continue
            agent_pair_type = get_agent_pair_type(agent_types[0], agent_types[1])
            _append_feature(regrouped_features, features, i, agent_pair_type)

    for key in regrouped_features:  # noqa: PLC0206
        for feature_name in regrouped_features[key]:
            regrouped_features[key][feature_name] = np.array(regrouped_features[key][feature_name], dtype=np.float32)  # pyright: ignore[reportArgumentType]
            regrouped_features[key][feature_name][np.isinf(regrouped_features[key][feature_name])] = BIG_EPS

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
    scene_scores_df: pd.DataFrame, test_percentile: float, output_filepath: Path, *, add_jaccard_index: bool = True
) -> dict[str, Any]:
    """Splits scenarios into in-distribution and out-of-distribution sets based on score percentiles.

    Args:
        scene_scores_df (pd.DataFrame): DataFrame containing scenario scores.
        test_percentile (float): Percentile threshold to define out-of-distribution scenarios.
        output_filepath (Path): Path to save the scenario splits JSON file.
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

    with output_filepath.open("w") as f:
        json.dump(scenario_splits, f, indent=4)
    return scenario_splits


def _filter_percentiles(percentile_values: list[int], percentiles: list[float]) -> tuple[list[int], list[float]]:
    """Filters percentiles to keep only those where the value changes from the previous one."""
    filtered_percentiles = []
    filtered_percentile_values = []
    prev_value = None
    for p, v in zip(percentile_values, percentiles, strict=False):
        if prev_value is None or v != prev_value:
            filtered_percentile_values.append(p)
            filtered_percentiles.append(v)
            prev_value = v
    return filtered_percentile_values, filtered_percentiles


def plot_feature_distributions(
    feature_data: dict[AgentType, Any] | dict[AgentPairType, Any],
    output_dir: Path,
    dpi: int = 300,
    tag: str = "",
    percentile_values: list[int] = [1, 10, 25, 50, 75, 90, 95, 99],  # noqa: B006
    *,
    show_kde: bool = True,
    show_percentiles: bool = True,
) -> None:
    """Plots the distribution of a feature using a histogram and density curve.

    Args:
        feature_data (NDArray[np.float32]): Array of feature values to plot.
        output_dir (Path): Directory to save the output plots.
        dpi (int): Dots per inch for the saved figure.
        tag (str): Optional tag to prepend to the output filenames.
        percentile_values (list[int]): List of percentiles to compute and display on the plot.
        show_kde (bool): Whether to show the kernel density estimate on the plot.
        show_percentiles (bool): Whether to display percentile lines on the plot.
    """
    prefix = f"{tag}_" if tag else ""
    feature_percentiles = {}
    for agent_type, features in feature_data.items():
        if agent_type == AgentPairType.TYPE_OTHER:
            continue

        for feature_name, feature_values in features.items():
            logger.info("Plotting %s for %s with %d samples", feature_name, agent_type.name, feature_values.shape[0])

            _, ax = plt.subplots(1, 1, figsize=(10, 6))
            sns.histplot(
                feature_values,
                color=FEATURE_COLOR_MAP.get(feature_name, "gray"),
                kde=show_kde,
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

            percentiles = np.round(np.percentile(feature_values, percentile_values), decimals=2).tolist()
            # Only keep percentiles where the value changes from the previous one
            filtered_percentile_values, filtered_percentiles = _filter_percentiles(percentile_values, percentiles)

            feature_percentiles[feature_name] = dict(
                zip(filtered_percentile_values, filtered_percentiles, strict=False)
            )
            if show_percentiles:
                for p, v in zip(filtered_percentile_values, filtered_percentiles, strict=False):
                    ax.axvline(v, color="black", linestyle="--", alpha=0.6)
                    y = ax.get_ylim()[1] * 0.9
                    ax.text(v, y, f"{p}th: {v:.2f}", rotation=90, verticalalignment="center", fontsize=8)

            plt.tight_layout()
            output_filepath = output_dir / f"{prefix}{feature_name}_{agent_type.name.lower()}_distributions.png"
            plt.savefig(output_filepath, dpi=dpi)
            plt.close()

        output_filepath = output_dir / f"{agent_type.name.lower()}_feature_percentiles.json"
        with open(output_filepath, "w") as f:
            json.dump(feature_percentiles, f, indent=4)


def plot_agent_scores_distributions(
    agent_scores: dict[str, Any],
    agent_scores_valid: dict[str, Any],
    output_dir: Path,
    dpi: int = 100,
    percentile_values: list[int] = [10, 25, 50, 75, 90, 95, 99],  # noqa: B006
) -> None:
    """Plots the distribution of agent scores using histograms and density curves.

    Args:
        agent_scores (dict[str, Any]): Dictionary containing agent scores with scenario IDs.
        agent_scores_valid (dict[str, Any]): Dictionary containing validity masks for agent scores.
        output_dir (Path): Directory to save the output plots.
        dpi (int): Dots per inch for the saved figure.
        percentile_values (list[int]): List of percentiles to compute and display on the plot.
    """
    for key, values in agent_scores.items():
        if key == "scenario_ids":
            continue

        agent_scores_flattened = []
        valid = agent_scores_valid.get(key)
        if valid is None:
            for scores in values:
                agent_scores_flattened.extend(scores.tolist())
        else:
            for scores, valid_mask in zip(values, valid, strict=True):
                agent_scores_flattened.extend(scores[valid_mask].tolist())
        agent_scores_flattened = [score for score in agent_scores_flattened if score >= 0.0]

        _, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            agent_scores_flattened,
            color="blue",
            kde=True,
            stat="probability",
            alpha=0.6,
            edgecolor="white",
            ax=ax,
        )

        sns.despine(top=True, right=True)

        ax.set_xlabel("Scores values")
        ax.set_ylabel("Density")
        ax.set_title(f"Scores Distribution ({len(agent_scores_flattened)} agents)")
        ax.grid(visible=True, linestyle="--", alpha=0.4)

        percentiles = np.round(np.percentile(agent_scores_flattened, percentile_values), decimals=2).tolist()
        # Only keep percentiles where the value changes from the previous one
        filtered_percentile_values, filtered_percentiles = _filter_percentiles(percentile_values, percentiles)

        score_percentiles = dict(zip(filtered_percentile_values, filtered_percentiles, strict=False))
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


def plot_agent_scores_heatmap(
    agent_scores: dict[str, Any],
    agent_scores_valid: dict[str, Any],
    scenario_type: str,
    criterion: str,
    output_dir: Path,
    dpi: int = 100,
) -> None:
    """Plots heatmaps of agent scores.

    Args:
        agent_scores (dict[str, Any]): Dictionary containing agent scores with scenario IDs.
        agent_scores_valid (dict[str, Any]): Dictionary containing validity masks for agent scores.
        criterion (str): Criterion used for filtering or labeling the heatmap.
        scenario_type (str): Type of scenario being analyzed.
        output_dir (Path): Directory to save the output plots.
        dpi (int): Dots per inch for the saved figure.
    """
    individual_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_individual", None)
    interaction_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_interaction", None)
    if individual_agent_scores is None or interaction_agent_scores is None:
        logger.error("Individual or interaction agent scores not found for criterion %s", criterion)
        return
    individual_agent_scores = np.concatenate(individual_agent_scores).astype(int)
    interaction_agent_scores = np.concatenate(interaction_agent_scores).astype(int)

    individual_agent_scores_valid = agent_scores_valid.get(f"{scenario_type}_{criterion}_individual", None)
    interaction_agent_scores_valid = agent_scores_valid.get(f"{scenario_type}_{criterion}_interaction", None)
    if individual_agent_scores_valid is not None and interaction_agent_scores_valid is not None:
        individual_agent_scores_valid = np.concatenate(individual_agent_scores_valid)
        interaction_agent_scores_valid = np.concatenate(interaction_agent_scores_valid)
        mask = individual_agent_scores_valid & interaction_agent_scores_valid
        individual_agent_scores = individual_agent_scores[mask]
        interaction_agent_scores = interaction_agent_scores[mask]

    assert individual_agent_scores.shape == interaction_agent_scores.shape, (
        f"Agent scores shapes do not match. {individual_agent_scores.shape}, {interaction_agent_scores.shape}"
    )

    heatmap = np.zeros(shape=(individual_agent_scores.max() + 1, interaction_agent_scores.max() + 1), dtype=int)
    for individual, interaction in tqdm(
        zip(individual_agent_scores, interaction_agent_scores, strict=True),
        desc=f"Plotting agent scores heatmap for {criterion}",
        total=len(individual_agent_scores),
    ):
        if individual < 0 or interaction < 0:
            logger.warning("Skipping invalid scores: %d, %d", individual, interaction)
            continue
        heatmap[individual, interaction] += 1

    sns.heatmap(
        heatmap,
        annot=True,
        fmt="d",
        cmap="rocket_r",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"label": "Number of Agents"},
        annot_kws={"fontsize": 6},
    )
    plt.xlabel("Interaction Agent Scores")
    plt.ylabel("Individual Agent Scores")
    plt.title(f"Agent Scores Heatmap for Criterion: {criterion}")
    plt.tight_layout()

    output_filepath = output_dir / f"agent_score_heatmap_{criterion}.png"
    plt.savefig(output_filepath, dpi=dpi)
    plt.close()


def plot_agent_scores_voxel(
    agent_scores: dict[str, Any],
    agent_scores_valid: dict[str, Any],
    scenario_type: str,
    criterion: str,
    output_dir: Path,
    dpi: int = 100,
) -> None:
    """Plots a 3D voxel plot of agent scores.

    Args:
        agent_scores (dict[str, Any]): Dictionary containing agent scores with scenario IDs.
        agent_scores_valid (dict[str, Any]): Dictionary containing validity masks for agent scores.
        scenario_type (str): Type of scenario being analyzed.
        criterion (str): Criterion used for filtering or labeling the plot.
        output_dir (Path): Directory to save the output plots.
        dpi (int): Dots per inch for the saved figure.
    """
    individual_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_individual", None)
    interaction_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_interaction", None)
    safeshift_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_safeshift", None)
    if individual_agent_scores is None or interaction_agent_scores is None or safeshift_agent_scores is None:
        logger.error("Individual or interaction or safeshift agent scores not found for criterion %s", criterion)
        return
    individual_agent_scores = np.concatenate(individual_agent_scores).astype(int)
    interaction_agent_scores = np.concatenate(interaction_agent_scores).astype(int)
    safeshift_agent_scores = np.concatenate(safeshift_agent_scores).astype(int)

    individual_agent_scores_valid = agent_scores_valid.get(f"{scenario_type}_{criterion}_individual", None)
    interaction_agent_scores_valid = agent_scores_valid.get(f"{scenario_type}_{criterion}_interaction", None)
    if individual_agent_scores_valid is not None and interaction_agent_scores_valid is not None:
        individual_agent_scores_valid = np.concatenate(individual_agent_scores_valid)
        interaction_agent_scores_valid = np.concatenate(interaction_agent_scores_valid)
        mask = individual_agent_scores_valid & interaction_agent_scores_valid
        individual_agent_scores = individual_agent_scores[mask]
        interaction_agent_scores = interaction_agent_scores[mask]
        safeshift_agent_scores = safeshift_agent_scores[mask]

    assert individual_agent_scores.shape == interaction_agent_scores.shape == safeshift_agent_scores.shape, (
        f"Agent scores shapes do not match. "
        f"{individual_agent_scores.shape}, {interaction_agent_scores.shape}, {safeshift_agent_scores.shape}"
    )

    # Create voxel grid
    voxels = np.zeros(
        shape=(individual_agent_scores.max() + 1, interaction_agent_scores.max() + 1, safeshift_agent_scores.max() + 1),
        dtype=int,
    )
    for individual, interaction, safeshift in tqdm(
        zip(individual_agent_scores, interaction_agent_scores, safeshift_agent_scores, strict=True),
        desc=f"Plotting agent scores voxel for {criterion}",
        total=len(individual_agent_scores),
    ):
        if individual < 0 or interaction < 0 or safeshift < 0:
            logger.warning("Skipping invalid scores: %d, %d, %d", individual, interaction, safeshift)
            continue
        voxels[individual, interaction, safeshift] += 1

    # Create 3D voxel plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("magma_r")
    norm = Normalize(vmin=0, vmax=voxels.max(initial=1))
    filled = voxels > 0
    ax.voxels(filled, facecolors=cmap(norm(voxels)), edgecolor="k", alpha=0.85, shade=False)  # pyright: ignore[reportAttributeAccessIssue]

    ax.set_xlabel("Individual Agent Score", labelpad=10)
    ax.set_ylabel("Interaction Agent Score", labelpad=10)
    ax.set_zlabel("Safeshift Agent Score", labelpad=10)  # pyright: ignore[reportAttributeAccessIssue]
    ax.set_title(f"Agent Scores for Criterion: {criterion}", pad=16, fontsize=14)

    ax.view_init(elev=25, azim=250)  # pyright: ignore[reportAttributeAccessIssue]
    ax.set_box_aspect((1, 1.3, 1.1))  # pyright: ignore[reportArgumentType]

    # Make panes and grid subtle
    ax.grid(visible=True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):  # pyright: ignore[reportAttributeAccessIssue]
        axis.pane.set_facecolor("white")  # pyright: ignore[reportAttributeAccessIssue]
        axis.pane.set_edgecolor("white")  # pyright: ignore[reportAttributeAccessIssue]
        axis._axinfo["grid"].update({"color": (0.5, 0.5, 0.5, 0.10), "linewidth": 0.8})  # noqa: SLF001 # pyright: ignore[reportAttributeAccessIssue]

    ax.set_xticks(np.arange(0, voxels.shape[0] + 1, 1))
    ax.set_yticks(np.arange(0, voxels.shape[1] + 1, 1))
    ax.set_zticks(np.arange(0, voxels.shape[2] + 1, 1))  # pyright: ignore[reportAttributeAccessIssue]

    # Add a colorbar mapping voxel to agent counts
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label("Voxel Count", fontsize=11)

    plt.tight_layout()
    output_filepath = output_dir / f"agent_score_voxel_{criterion}.png"
    plt.savefig(output_filepath, dpi=dpi)
    plt.close()
