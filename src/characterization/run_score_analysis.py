r"""Entrypoint for analyzing and visualizing precomputed scenario scores.

Loads scenario scores, produces density plots and train/test split files, and plots per-agent score distributions,
heatmaps, and voxel plots. Configuration is loaded from ``config/run_analysis.yaml`` by default.

Supports both single-dataset and multi-dataset analysis. In multi-dataset mode (``cfg.datasets`` is set), the analysis
runs independently for each dataset and also produces combined overlay plots comparing all datasets on shared axes.

Example usage::
    uv run python -m characterization.run_score_analysis
    uv run python -m characterization.run_score_analysis scores_path=/path/to/scores output_dir=/path/to/output

    # Multi-dataset
    uv run python -m characterization.run_score_analysis \\
        "datasets=[{label: Waymo, dataset_name: waymo}, {label: nuScenes, dataset_name: nuscenes}]"
"""

from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig

from characterization.utils import analysis, common
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


def _load_and_regroup_scores(
    cfg: DictConfig,
    scores_path: Path,
    features_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Loads and regroups scenario scores and agent types from *scores_path* and *features_path*.

    Uses ``cfg.scenario_types``, ``cfg.criteria``, ``cfg.scores``, and ``cfg.total_scenarios`` to control which
    scenarios are loaded. All datasets in a multi-dataset run share these settings.

    Args:
        cfg (DictConfig): Top-level configuration.
        scores_path (Path): Path to the directory containing pre-computed score files.
        features_path (Path): Path to the directory containing pre-computed feature files (used to load agent types).

    Returns:
        Tuple of (scene_scores_df, agent_scores, agent_scores_valid, agent_types).

    Raises:
        ValueError: If no valid scenarios are found under *scores_path*.
    """
    scenario_types = list(cfg.scenario_types)
    criteria = list(cfg.criteria)

    scenario_ids = analysis.get_valid_scenario_ids(scenario_types, criteria, scores_path)
    if not scenario_ids:
        msg = f"No valid scenarios found in {scores_path} for {scenario_types} and criteria {criteria}"
        raise ValueError(msg)

    total_scenarios = (
        min(len(scenario_ids), cfg.total_scenarios)
        if cfg.total_scenarios and cfg.total_scenarios > 0
        else len(scenario_ids)
    )
    logger.info("Found %d valid scenarios for analysis. Using %d scenarios.", len(scenario_ids), total_scenarios)
    scenario_ids = scenario_ids[:total_scenarios]

    logger.info("Loading agent types from %s", features_path)
    _, _, agent_types = analysis.load_scenario_features(scenario_ids, scenario_types, criteria, features_path)

    logger.info("Loading scores from %s", scores_path)
    scenario_scores = analysis.load_scenario_scores(scenario_ids, scenario_types, criteria, scores_path)
    scene_scores, agent_scores, agent_scores_valid = analysis.regroup_scenario_scores(
        scenario_scores, scenario_ids, scenario_types, list(cfg.scores), criteria
    )

    scene_scores_df = pd.DataFrame(scene_scores)
    float_cols = scene_scores_df.select_dtypes(include="float").columns
    scene_scores_df[float_cols] = scene_scores_df[float_cols].round(3)

    return scene_scores_df, agent_scores, agent_scores_valid, agent_types


def _run_single_dataset(
    cfg: DictConfig,
    scores_path: Path,
    features_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Runs the full per-dataset score analysis pipeline and writes all outputs to *output_dir*.

    Returns:
        Tuple of (scene_scores_df, agent_scores, agent_scores_valid) so the caller can accumulate results for combined
        multi-dataset plots.
    """
    scene_scores_df, agent_scores, agent_scores_valid, agent_types = _load_and_regroup_scores(
        cfg, scores_path, features_path
    )

    output_filepath = output_dir / "scene_to_scores_mapping.csv"
    scene_scores_df.to_csv(output_filepath, index=False)
    logger.info("Saving scene to scores mapping to %s", output_filepath)

    output_filepath = output_dir / f"{cfg.tag}_score_density_plot.png"
    analysis.plot_histograms_from_dataframe(scene_scores_df, output_filepath, cfg.dpi)
    logger.info("Visualizing density function for scores: %s to %s", list(cfg.scores), output_filepath)

    output_filepath = output_dir / "scenario_splits.json"
    logger.info("Generating score split files to %s", output_filepath)
    analysis.get_scenario_splits(scene_scores_df, cfg.test_percentile, output_filepath, add_jaccard_index=True)

    analysis.plot_agent_scores_distributions(agent_scores, agent_scores_valid, output_dir, cfg.dpi)
    logger.info("Visualized agent score distributions to %s", output_dir)

    for scenario_type, criterion in product(cfg.scenario_types, cfg.criteria):
        if "categorical" not in criterion:
            continue

        analysis.plot_agent_scores_heatmap(
            agent_scores, agent_scores_valid, scenario_type, criterion, output_dir, cfg.dpi
        )
        analysis.plot_agent_scores_voxel(
            agent_scores, agent_scores_valid, scenario_type, criterion, output_dir, cfg.dpi
        )
        analysis.plot_agent_scores_voxel_by_agent_type(
            agent_scores, agent_scores_valid, scenario_type, criterion, agent_types, output_dir, cfg.dpi
        )
    logger.info("Visualized agent score heatmaps to %s", output_dir)

    return scene_scores_df, agent_scores, agent_scores_valid


@hydra.main(config_path="config", config_name="run_analysis", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """Runs the scenario score analysis pipeline using the provided configuration.

    Loads scenario scores, generates a score density plot and a scene-to-scores CSV, writes train/test split files, and
    plots per-agent score distributions, 2D heatmaps, and 3D voxel plots for categorical criteria.

    When ``cfg.datasets`` is set, runs the analysis independently for each dataset and also produces combined overlay
    plots comparing all datasets on shared axes.

    Args:
        cfg (DictConfig): Configuration dictionary specifying scoring methods, paths, and output options.

    Raises:
        ValueError: If unsupported scenario types or scorers are specified in the configuration.
    """
    subdir = ""
    if cfg.add_timestamp:
        subdir = f"{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}_"
    subdir = f"{subdir}{cfg.exp_tag}" if cfg.exp_tag else subdir
    output_dir = Path(cfg.output_dir) / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify scenario types are supported
    unsupported_scenario_types = [
        scenario_type for scenario_type in cfg.scenario_types if scenario_type not in common.SUPPORTED_SCENARIO_TYPES
    ]
    if unsupported_scenario_types:
        msg = f"Scenario types {unsupported_scenario_types} not in supported list {common.SUPPORTED_SCENARIO_TYPES}"
        raise ValueError(msg)
    # Verify scorer type is supported
    unsupported_scores = [scorer for scorer in cfg.scores if scorer not in common.SUPPORTED_SCORERS]
    if unsupported_scores:
        msg = f"Scorers {unsupported_scores} not in supported list {common.SUPPORTED_SCORERS}"
        raise ValueError(msg)

    # Single-dataset path — identical behaviour to the original script
    if cfg.datasets is None:
        _run_single_dataset(cfg, Path(cfg.scores_path), Path(cfg.features_path), output_dir)

    # Multi-dataset path — per-dataset analysis + combined overlay plots
    else:
        all_scene_scores: dict[str, pd.DataFrame] = {}
        all_agent_data: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

        for dataset_entry in cfg.datasets:
            label = dataset_entry.label
            # Derive per-dataset paths by substituting dataset_name into the resolved path templates.
            scores_path = Path(str(cfg.scores_path).replace(cfg.paths.dataset_name, dataset_entry.dataset_name))
            features_path = Path(str(cfg.features_path).replace(cfg.paths.dataset_name, dataset_entry.dataset_name))
            ds_output_dir = output_dir / label
            ds_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Processing dataset: %s", label)
            scene_scores_df, agent_scores, agent_scores_valid = _run_single_dataset(
                cfg, scores_path, features_path, ds_output_dir
            )

            all_scene_scores[label] = scene_scores_df
            all_agent_data[label] = (agent_scores, agent_scores_valid)

        # Combined overlay plots across all datasets
        combined_dir = output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating combined overlay plots for %d datasets", len(all_scene_scores))
        analysis.plot_multi_dataset_score_distributions(all_scene_scores, combined_dir, cfg.dpi, cfg.tag)
        analysis.plot_multi_dataset_agent_score_distributions(all_agent_data, combined_dir, cfg.dpi)
        logger.info("Visualized combined score plots to %s", combined_dir)


if __name__ == "__main__":
    run()
