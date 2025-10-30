import json
from datetime import UTC, datetime
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from characterization.scorer import score_utils
from characterization.utils import common
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="config", config_name="run_analysis", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """Runs the scenario score visualization pipeline using the provided configuration.

    This function loads scenario scores, generates density plots for each scoring method, and visualizes example
    scenarios across score percentiles. It supports multiple scoring criteria and flexible dataset/visualizer
    instantiation via Hydra.

    Args:
        cfg (DictConfig): Configuration dictionary specifying dataset, visualizer, scoring methods, paths, and output
            options.

    Raises:
        ValueError: If unsupported scorers are specified in the configuration.
    """
    subdir = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    subdir = f"{subdir}_{cfg.exp_tag}" if cfg.exp_tag else subdir
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
    unsupported_scores = [scorer for scorer in cfg.scores if scorer not in score_utils.SUPPORTED_SCORERS]
    if unsupported_scores:
        msg = f"Scorers {unsupported_scores} not in supported list {score_utils.SUPPORTED_SCORERS}"
        raise ValueError(msg)

    scenario_ids = score_utils.get_valid_scenario_ids(cfg.scenario_types, cfg.criteria, cfg.scores_path)
    if not scenario_ids:
        msg = f"No valid scenarios found in {cfg.scores_path} for {cfg.scenario_types} and criteria {cfg.criteria}"
        raise ValueError(msg)

    # Generate score histogram and density plot
    logger.info("Loading the scores")
    scene_scores, _, _, _, _ = score_utils.load_scenario_scores(
        scenario_ids,
        cfg.scenario_types,
        cfg.scores,
        cfg.criteria,
        Path(cfg.scores_path),
    )

    logger.info("Visualizing density function for scores: %s", cfg.scores)
    scene_scores_df = pd.DataFrame(scene_scores)
    output_filepath = output_dir / f"{cfg.tag}_score_density_plot.png"
    logger.info("Saving density plot: %s", output_filepath)
    score_utils.plot_histograms_from_dataframe(scene_scores_df, output_filepath, cfg.dpi)

    logger.info("Generating score split files")
    scenario_splits = score_utils.get_scenario_splits(scene_scores_df, cfg.test_percentile, add_jaccard_index=True)

    output_filepath = output_dir / "scenario_splits.json"
    with output_filepath.open("w") as f:
        json.dump(scenario_splits, f, indent=4)
    logger.info("Saved scenario splits to %s", output_filepath)


if __name__ == "__main__":
    run()  # pyright: ignore[reportCallIssue]
