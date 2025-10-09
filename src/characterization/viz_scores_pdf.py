from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

import characterization.utils.viz.utils as viz_utils
from characterization.scorer import SUPPORTED_SCORERS
from characterization.utils.common import SUPPORTED_SCENARIO_TYPES
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="config", config_name="run_visualization", version_base="1.3")
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
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Verify scenario types are supported
    unsupported_scenario_types = [
        scenario_type for scenario_type in cfg.scenario_types if scenario_type not in SUPPORTED_SCENARIO_TYPES
    ]
    if unsupported_scenario_types:
        msg = f"Scenario types {unsupported_scenario_types} not in supported list {SUPPORTED_SCENARIO_TYPES}"
        raise ValueError(msg)
    # Verify scorer type is supported
    unsupported_scores = [scorer for scorer in cfg.scores if scorer not in SUPPORTED_SCORERS]
    if unsupported_scores:
        msg = f"Scorers {unsupported_scores} not in supported list {SUPPORTED_SCORERS}"
        raise ValueError(msg)

    scenario_ids = viz_utils.get_valid_scenario_ids(cfg.scenario_types, cfg.criteria, cfg.scores_path)
    if not scenario_ids:
        msg = f"No valid scenarios found in {cfg.scores_path} for {cfg.scenario_types} and criteria {cfg.criteria}"
        raise ValueError(msg)

    # Generate score histogram and density plot
    logger.info("Loading the scores")
    scene_scores, _, _ = viz_utils.load_scenario_scores(
        scenario_ids,
        cfg.scenario_types,
        cfg.scores,
        cfg.criteria,
        Path(cfg.scores_path),
    )

    logger.info("Visualizing density function for scores: %s", cfg.scores)
    scene_scores_df = pd.DataFrame(scene_scores)
    output_filepath = Path(cfg.output_dir) / f"{cfg.tag}_score_density_plot.png"
    logger.info("Saving density plot: %s", output_filepath)
    viz_utils.plot_histograms_from_dataframe(scene_scores_df, output_filepath, cfg.dpi)


if __name__ == "__main__":
    run()  # pyright: ignore[reportCallIssue]
