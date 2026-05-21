import json
from itertools import combinations, product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from tqdm import tqdm

from characterization.schemas import ScenarioScores
from characterization.utils.analysis.common_analysis import (
    DEFAULT_FEATURE_CATEGORIES,
    compute_category_thresholds,
    compute_jaccard_index,
)
from characterization.utils.io_utils import from_pickle, get_logger
from characterization.utils.scenario_types import AgentType

logger = get_logger(__name__)


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


def plot_agent_scores_distributions(
    agent_scores: dict[str, Any],
    agent_scores_valid: dict[str, Any],
    output_dir: Path,
    dpi: int = 100,
    categories: list[dict[str, Any]] | None = None,
) -> None:
    """Plots the distribution of agent scores using histograms and density curves.

    Args:
        agent_scores (dict[str, Any]): Dictionary containing agent scores with scenario IDs.
        agent_scores_valid (dict[str, Any]): Dictionary containing validity masks for agent scores.
        output_dir (Path): Directory to save the output plots.
        dpi (int): Dots per inch for the saved figure.
        categories: Semantic category definitions used to compute boundary thresholds.
            Defaults to LOW/MEDIUM/HIGH/CRITICAL split at the 25th, 75th, and 90th percentiles.
    """
    if categories is None:
        categories = DEFAULT_FEATURE_CATEGORIES
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

        score_percentiles = compute_category_thresholds(np.array(agent_scores_flattened), categories)
        for label, v in score_percentiles.items():
            ax.axvline(v, color="black", linestyle="--", alpha=0.6)
            ax.text(v, ax.get_ylim()[1] * 0.9, f"{label}: {v:.2f}", rotation=90, verticalalignment="center")

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
    individual_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_individual")
    interaction_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_interaction")
    if individual_agent_scores is None or interaction_agent_scores is None:
        logger.error("Individual or interaction agent scores not found for criterion %s", criterion)
        return
    individual_agent_scores = np.concatenate(individual_agent_scores).astype(int)
    interaction_agent_scores = np.concatenate(interaction_agent_scores).astype(int)

    individual_agent_scores_valid = agent_scores_valid.get(f"{scenario_type}_{criterion}_individual")
    interaction_agent_scores_valid = agent_scores_valid.get(f"{scenario_type}_{criterion}_interaction")
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

    ax = sns.heatmap(
        heatmap,
        annot=True,
        fmt="d",
        cmap="rocket_r",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"label": "Number of Agents"},
        annot_kws={"fontsize": 6},
    )
    ax.invert_yaxis()
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
    individual_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_individual")
    interaction_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_interaction")
    safeshift_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_safeshift")
    if individual_agent_scores is None or interaction_agent_scores is None or safeshift_agent_scores is None:
        logger.error("Individual or interaction or safeshift agent scores not found for criterion %s", criterion)
        return
    individual_agent_scores = np.concatenate(individual_agent_scores).astype(int)
    interaction_agent_scores = np.concatenate(interaction_agent_scores).astype(int)
    safeshift_agent_scores = np.concatenate(safeshift_agent_scores).astype(int)

    individual_agent_scores_valid = agent_scores_valid.get(f"{scenario_type}_{criterion}_individual")
    interaction_agent_scores_valid = agent_scores_valid.get(f"{scenario_type}_{criterion}_interaction")
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


def plot_agent_scores_voxel_by_agent_type(
    agent_scores: dict[str, Any],
    agent_scores_valid: dict[str, Any],
    scenario_type: str,
    criterion: str,
    agent_types: dict[str, Any],
    output_dir: Path,
    dpi: int = 100,
) -> None:
    """Plots a 3D voxel plot of agent scores.

    Args:
        agent_scores (dict[str, Any]): Dictionary containing agent scores with scenario IDs.
        agent_scores_valid (dict[str, Any]): Dictionary containing validity masks for agent scores.
        scenario_type (str): Type of scenario being analyzed.
        criterion (str): Criterion used for filtering or labeling the plot.
        agent_types (dict[str, Any]): Dictionary containing agent types with scenario IDs.
        output_dir (Path): Directory to save the output plots.
        dpi (int): Dots per inch for the saved figure.
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
    scenario_agent_types = agent_types["agent_types"]

    individual_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_individual")
    interaction_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_interaction")
    safeshift_agent_scores = agent_scores.get(f"{scenario_type}_{criterion}_safeshift")
    if individual_agent_scores is None or interaction_agent_scores is None or safeshift_agent_scores is None:
        logger.error("Individual or interaction or safeshift agent scores not found for criterion %s", criterion)
        return

    for types, scores in zip(scenario_agent_types, individual_agent_scores, strict=True):
        if len(types) != len(scores):
            logger.error(
                "Length of agent type list does not match length of individual agent scores for criterion %s", criterion
            )
            return

    scenario_agent_types = np.concatenate(scenario_agent_types)
    individual_agent_scores = np.concatenate(individual_agent_scores).astype(int)
    interaction_agent_scores = np.concatenate(interaction_agent_scores).astype(int)
    safeshift_agent_scores = np.concatenate(safeshift_agent_scores).astype(int)

    individual_agent_scores_valid = agent_scores_valid.get(f"{scenario_type}_{criterion}_individual")
    interaction_agent_scores_valid = agent_scores_valid.get(f"{scenario_type}_{criterion}_interaction")
    if individual_agent_scores_valid is not None and interaction_agent_scores_valid is not None:
        individual_agent_scores_valid = np.concatenate(individual_agent_scores_valid)
        interaction_agent_scores_valid = np.concatenate(interaction_agent_scores_valid)
        mask = individual_agent_scores_valid & interaction_agent_scores_valid
        individual_agent_scores = individual_agent_scores[mask]
        interaction_agent_scores = interaction_agent_scores[mask]
        safeshift_agent_scores = safeshift_agent_scores[mask]
        scenario_agent_types = scenario_agent_types[mask]

    assert individual_agent_scores.shape == interaction_agent_scores.shape == safeshift_agent_scores.shape, (
        f"Agent scores shapes do not match. "
        f"{individual_agent_scores.shape}, {interaction_agent_scores.shape}, {safeshift_agent_scores.shape}"
    )

    colormap = {
        AgentType.TYPE_VEHICLE: "bone_r",
        AgentType.TYPE_CYCLIST: "Greens",
        AgentType.TYPE_PEDESTRIAN: "RdPu",
    }
    for agent_type in [AgentType.TYPE_VEHICLE, AgentType.TYPE_CYCLIST, AgentType.TYPE_PEDESTRIAN]:
        agent_type_mask = scenario_agent_types == agent_type
        if not np.any(agent_type_mask):
            logger.warning("No agents of type %s found for criterion %s", agent_type.name, criterion)
            continue
        individual_agent_scores_agent_type = individual_agent_scores[agent_type_mask]
        interaction_agent_scores_agent_type = interaction_agent_scores[agent_type_mask]
        safeshift_agent_scores_agent_type = safeshift_agent_scores[agent_type_mask]

        # Create voxel grid
        voxels = np.zeros(
            shape=(
                individual_agent_scores_agent_type.max() + 1,
                interaction_agent_scores_agent_type.max() + 1,
                safeshift_agent_scores_agent_type.max() + 1,
            ),
            dtype=int,
        )
        for individual, interaction, safeshift in tqdm(
            zip(
                individual_agent_scores_agent_type,
                interaction_agent_scores_agent_type,
                safeshift_agent_scores_agent_type,
                strict=True,
            ),
            desc=f"Plotting agent scores voxel for {criterion}",
            total=len(individual_agent_scores_agent_type),
        ):
            if individual < 0 or interaction < 0 or safeshift < 0:
                logger.warning("Skipping invalid scores: %d, %d, %d", individual, interaction, safeshift)
                continue
            voxels[individual, interaction, safeshift] += 1

        # Create 3D voxel plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        cmap = plt.get_cmap(colormap.get(agent_type, "magma_r"))
        norm = Normalize(vmin=0, vmax=voxels.max(initial=1))
        filled = voxels > 0
        ax.voxels(filled, facecolors=cmap(norm(voxels)), edgecolor="k", alpha=0.85, shade=False)  # pyright: ignore[reportAttributeAccessIssue]

        ax.set_xlabel("Individual Agent Score", labelpad=10)
        ax.set_ylabel("Interaction Agent Score", labelpad=10)
        ax.set_zlabel("Safeshift Agent Score", labelpad=10)  # pyright: ignore[reportAttributeAccessIssue]
        agent_type_name = agent_type.name.split("_")[1].title()
        ax.set_title(f"{agent_type_name} Agent Scores", pad=12, fontsize=14)

        ax.view_init(elev=25, azim=250)  # pyright: ignore[reportAttributeAccessIssue]
        ax.set_box_aspect((1, 1.3, 1.1))  # pyright: ignore[reportArgumentType]
        ax.set_proj_type("ortho")  # pyright: ignore[reportAttributeAccessIssue]

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
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.00)
        cbar.set_label("Agent Count", fontsize=11)

        plt.tight_layout()
        output_filepath = output_dir / f"agent_score_voxel_{criterion}_{agent_type_name}.png"
        plt.savefig(output_filepath, dpi=dpi)
        plt.close()
