"""Entrypoint for analyzing and visualizing precomputed scenario scores.

Loads scenario scores, produces density plots and train/test split files, and plots per-agent score distributions,
heatmaps, and voxel plots. Configuration is loaded from ``config/run_analysis.yaml`` by default.

Example usage::

    uv run python -m characterization.run_score_analysis
    uv run python -m characterization.run_score_analysis scores_path=/path/to/scores output_dir=/path/to/output
"""

from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from characterization.utils import analysis_utils, common
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="config", config_name="run_analysis", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """Runs the scenario score analysis pipeline using the provided configuration.

    Loads scenario scores, generates a score density plot and a scene-to-scores CSV, writes train/test split files,
    and plots per-agent score distributions, 2D heatmaps, and 3D voxel plots for categorical criteria.

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

    scores_path = Path(cfg.scores_path)
    scenario_ids = analysis_utils.get_valid_scenario_ids(cfg.scenario_types, cfg.criteria, scores_path)
    scenario_ids = (
        scenario_ids[: cfg.total_scenarios]
        if cfg.total_scenarios is not None and cfg.total_scenarios > 0
        else scenario_ids
    )
    if not scenario_ids:
        msg = f"No valid scenarios found in {cfg.scores_path} for {cfg.scenario_types} and criteria {cfg.criteria}"
        raise ValueError(msg)

    # Generate score histogram and density plot
    logger.info("Loading agent types")
    features_path = Path(cfg.features_path)
    _, _, agent_types = analysis_utils.load_scenario_features(
        scenario_ids,
        cfg.scenario_types,
        cfg.criteria,
        features_path,
    )

    logger.info("Loading the scores")
    scenario_scores = analysis_utils.load_scenario_scores(scenario_ids, cfg.scenario_types, cfg.criteria, scores_path)
    scene_scores, agent_scores, agent_scores_valid = analysis_utils.regroup_scenario_scores(
        scenario_scores, scenario_ids, cfg.scenario_types, cfg.scores, cfg.criteria
    )

    scene_scores_df = pd.DataFrame(scene_scores)
    float_cols = scene_scores_df.select_dtypes(include="float").columns
    scene_scores_df[float_cols] = scene_scores_df[float_cols].round(3)
    output_filepath = output_dir / "scene_to_scores_mapping.csv"
    scene_scores_df.to_csv(output_filepath, index=False)
    logger.info("Saving scene to scores mapping to %s", output_filepath)

    output_filepath = output_dir / f"{cfg.tag}_score_density_plot.png"
    analysis_utils.plot_histograms_from_dataframe(scene_scores_df, output_filepath, cfg.dpi)
    logger.info("Visualizing density function for scores: %s to %s", cfg.scores, output_filepath)

    output_filepath = output_dir / "scenario_splits.json"
    logger.info("Generating score split files to %s", output_filepath)
    analysis_utils.get_scenario_splits(scene_scores_df, cfg.test_percentile, output_filepath, add_jaccard_index=True)

    analysis_utils.plot_agent_scores_distributions(agent_scores, agent_scores_valid, output_dir, cfg.dpi)
    logger.info("Visualized agent score distributions to %s", output_dir)

    for scenario_type, criterion in product(cfg.scenario_types, cfg.criteria):
        if "categorical" not in criterion:
            continue

        # Plots the individual and interaction scores as a 2D heatmap
        analysis_utils.plot_agent_scores_heatmap(
            agent_scores, agent_scores_valid, scenario_type, criterion, output_dir, cfg.dpi
        )

        # Plots the (individual, interaction, safeshift) scores as a 3D voxel plot
        analysis_utils.plot_agent_scores_voxel(
            agent_scores,
            agent_scores_valid,
            scenario_type,
            criterion,
            output_dir,
            cfg.dpi,
        )

        # Plots the (individual, interaction, safeshift) scores as a 3D voxel plot separated by agent type
        analysis_utils.plot_agent_scores_voxel_by_agent_type(
            agent_scores,
            agent_scores_valid,
            scenario_type,
            criterion,
            agent_types,
            output_dir,
            cfg.dpi,
        )
    logger.info("Visualized agent score heatmaps to %s", output_dir)


if __name__ == "__main__":
    run()
