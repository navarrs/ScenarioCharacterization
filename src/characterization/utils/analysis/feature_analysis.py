import json
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from tqdm import tqdm

from characterization.schemas import Individual, Interaction, ScenarioFeatures
from characterization.utils.analysis.common_analysis import (
    AGENT_COLORS,
    DEFAULT_FEATURE_CATEGORIES,
    FEATURE_COLOR_MAP,
    compute_category_thresholds,
    get_dataset_colors,
)
from characterization.utils.common import EPSILON, LARGE_VALUE, InteractionStatus
from characterization.utils.io_utils import from_pickle, get_logger
from characterization.utils.scenario_types import AgentPairType, AgentType, get_agent_pair_type

logger = get_logger(__name__)

_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "drac": "Deceleration Rate to Avoid Collision (DRAC)",
    "ttc": "Time to Collision (TTC)",
    "inv_ttc": "Inverse Time to Collision (Inv. TTC)",
    "thw": "Time Headway (THW)",
    "inv_thw": "Inverse Time Headway (Inv. THW)",
    "mttcp": "Minimum Time to Collision Point (mTTCP)",
    "inv_mttcp": "Inverse Minimum Time to Collision Point (Inv. mTTCP)",
}


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
    agent_types = {"scenario_ids": [], "agent_types": []}
    for scenario_type, criterion in product(scenario_types, criteria):
        key = f"{scenario_type}_{criterion}"
        scenario_features_path = features_path / key
        features = load_features(scenario_ids, scenario_features_path, key)
        for scenario_id, feat in features.items():
            if feat.individual_features is not None:
                individual_features["scenario_ids"].append(scenario_id)
                individual_features["features"].append(feat.individual_features)  # pyright: ignore[reportArgumentType]
                agent_types["scenario_ids"].append(scenario_id)
                agent_types["agent_types"].append(feat.individual_features.agent_types)  # pyright: ignore[reportArgumentType]
            if feat.interaction_features is not None:
                interaction_features["scenario_ids"].append(scenario_id)
                interaction_features["features"].append(feat.interaction_features)  # pyright: ignore[reportArgumentType]
    return individual_features, interaction_features, agent_types


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
            sld = feature.speed_limit_diff[mask]
            feature_dict[agent_type]["speed_limit_diff"].extend(sld[~np.isnan(sld)].tolist())
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
            feature_dict[agent_pair]["inv_separation"].append(1 / (feature.separation[index] + EPSILON))
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
            regrouped_features[key][feature_name][np.isinf(regrouped_features[key][feature_name])] = LARGE_VALUE

    return regrouped_features


def plot_feature_distributions(
    feature_data: dict[AgentType, Any] | dict[AgentPairType, Any],
    output_dir: Path,
    dpi: int = 300,
    tag: str = "",
    categories: list[dict[str, Any]] | None = None,
    *,
    show_kde: bool = True,
    show_percentiles: bool = True,
    show_colored_by_agent_type: bool = True,
    include_pairs_with_no_vehicles: bool = False,
) -> None:
    """Plots the distribution of a feature using a histogram and density curve.

    Args:
        feature_data (NDArray[np.float32]): Array of feature values to plot.
        output_dir (Path): Directory to save the output plots.
        dpi (int): Dots per inch for the saved figure.
        tag (str): Optional tag to prepend to the output filenames.
        categories: Semantic category definitions, each a dict with "name" and "percentile_range" keys. Boundaries
            between consecutive categories are used as thresholds. If a boundary value duplicates the previous one, the
            range is scanned at finer granularity to find a unique value. Defaults to LOW/MEDIUM/HIGH/CRITICAL split at
            the 25th, 75th, and 90th percentiles.
        show_kde (bool): Whether to show the kernel density estimate on the plot.
        show_percentiles (bool): Whether to display percentile lines on the plot.
        show_colored_by_agent_type (bool): Whether to color the histograms by agent type.
        include_pairs_with_no_vehicles (bool): Whether to include interaction pairs that do not involve any vehicles.
    """
    if categories is None:
        categories = DEFAULT_FEATURE_CATEGORIES
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
    prefix = f"{tag}_" if tag else ""
    feature_percentiles = {}
    for agent_type, features in feature_data.items():
        if agent_type == AgentPairType.TYPE_OTHER:
            continue

        if not include_pairs_with_no_vehicles and agent_type in [
            AgentPairType.TYPE_CYCLIST_CYCLIST,
            AgentPairType.TYPE_PEDESTRIAN_PEDESTRIAN,
            AgentPairType.TYPE_PEDESTRIAN_CYCLIST,
        ]:
            continue

        for feature_name, feature_values in features.items():
            logger.info("Plotting %s for %s with %d samples", feature_name, agent_type.name, feature_values.shape[0])

            if feature_values.shape[0] == 0:
                logger.warning("No data for feature %s / %s; skipping plot", feature_name, agent_type.name)
                continue

            color = (
                AGENT_COLORS.get(agent_type, "gray")
                if show_colored_by_agent_type
                else FEATURE_COLOR_MAP.get(feature_name, "gray")
            )
            _, ax = plt.subplots(1, 1, figsize=(10, 6))
            sns.histplot(
                feature_values,
                color=color,
                bins=20,
                kde=show_kde,
                stat="density",
                alpha=0.7,
                # edgecolor="white",
                edgecolor=None,
                ax=ax,
            )
            ax.set_yscale("log")
            sns.despine(top=True, right=True)
            feature_title = feature_name.replace("_", " ").title()
            # feature_title = feature_name.replace("_", " ").upper()
            ax.set_xlabel(f"{feature_title} values")
            ax.set_ylabel("Density")
            ax.set_title(f"{feature_title} Distribution ({feature_values.shape[0]} samples)")
            ax.grid(visible=True, linestyle="--", alpha=0.4)

            thresholds = compute_category_thresholds(feature_values, categories)
            feature_percentiles[feature_name] = thresholds

            if show_percentiles:
                for label, v in thresholds.items():
                    ax.axvline(v, color="black", linestyle="--", alpha=0.9)
                    y = ax.get_ylim()[1] * 0.9
                    x = v + 0.08
                    ax.text(
                        x, y, f"{label}: {v:.2f}", rotation=90, verticalalignment="center", fontsize=8, color="dimgray"
                    )

            # Only for speed_lim_diff aesthetics
            # ax.set_xlim(left=0, right=30)

            # Only for drac aesthetics
            # ax.set_xlim(left=0, right=14)

            plt.tight_layout()
            output_filepath = output_dir / f"{prefix}{feature_name}_{agent_type.name.lower()}_distributions.png"
            plt.savefig(output_filepath, dpi=dpi)
            plt.close()

        output_filepath = output_dir / f"{agent_type.name.lower()}_feature_percentiles.json"
        with open(output_filepath, "w") as f:
            json.dump(feature_percentiles, f, indent=4)


def plot_multi_dataset_feature_distributions(
    feature_data_by_dataset: Mapping[str, dict[AgentType, Any] | dict[AgentPairType, Any]],
    output_dir: Path,
    dpi: int = 300,
    tag: str = "",
    *,
    show_kde: bool = True,
    include_pairs_with_no_vehicles: bool = False,
) -> None:
    """Plots overlapping feature distributions for multiple datasets on shared axes.

    Each dataset is rendered as a semi-transparent histogram with a distinct color. One PNG is
    saved per (agent_type, feature_name) pair. Feature names in ``_FEATURE_DISPLAY_NAMES`` are
    rendered with their full spelled-out name and abbreviation.

    Args:
        feature_data_by_dataset (dict[str, dict]): Mapping from dataset label to regrouped feature
            data (as returned by ``regroup_individual_features`` or ``regroup_interaction_features``).
        output_dir (Path): Directory to save the output plots.
        dpi (int): Dots per inch for the saved figure.
        tag (str): Optional tag to prepend to output filenames.
        show_kde (bool): Whether to show the kernel density estimate on each histogram.
        include_pairs_with_no_vehicles (bool): Whether to include interaction pairs that do not
            involve any vehicles.
    """
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
    prefix = f"{tag}_" if tag else ""
    dataset_labels = list(feature_data_by_dataset.keys())
    dataset_colors = get_dataset_colors(dataset_labels)

    # Use the first dataset's structure as the canonical schema; all datasets share it.
    first_dataset_features = next(iter(feature_data_by_dataset.values()))

    for agent_type, features_for_first in first_dataset_features.items():
        if agent_type == AgentPairType.TYPE_OTHER:
            continue
        if not include_pairs_with_no_vehicles and agent_type in [
            AgentPairType.TYPE_CYCLIST_CYCLIST,
            AgentPairType.TYPE_PEDESTRIAN_PEDESTRIAN,
            AgentPairType.TYPE_PEDESTRIAN_CYCLIST,
        ]:
            continue

        for feature_name in features_for_first:
            _, ax = plt.subplots(1, 1, figsize=(10, 6))
            all_values: list[Any] = []

            for label, feature_data in feature_data_by_dataset.items():
                if agent_type not in feature_data:
                    continue
                feature_values = feature_data[agent_type].get(feature_name)  # pyright: ignore[reportArgumentType]
                if feature_values is None or feature_values.shape[0] == 0:
                    continue
                all_values.append(feature_values)
                logger.info(
                    "Plotting %s for %s / %s with %d samples",
                    feature_name,
                    label,
                    agent_type.name,
                    feature_values.shape[0],
                )
                sns.histplot(
                    feature_values,
                    color=dataset_colors[label],
                    bins=20,
                    kde=show_kde,
                    stat="density",
                    alpha=0.5,
                    edgecolor=None,
                    label=f"{label} (n={feature_values.shape[0]})",
                    ax=ax,
                )

            ax.set_yscale("log")
            sns.despine(top=True, right=True)
            feature_title = _FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace("_", " ").title())
            ax.set_xlabel(f"{feature_title} values")
            ax.set_ylabel("Density")
            ax.set_title(f"{feature_title} Distribution")
            ax.grid(visible=True, linestyle="--", alpha=0.4)
            ax.legend(title="Dataset", fontsize=8)

            plt.tight_layout()
            output_filepath = (
                output_dir / f"{prefix}{feature_name}_{agent_type.name.lower()}_distributions_combined.png"
            )
            plt.savefig(output_filepath, dpi=dpi)
            plt.close()
