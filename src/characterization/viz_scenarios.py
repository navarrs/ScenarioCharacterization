from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

import characterization.utils.viz.utils as viz_utils
from characterization.scorer import SUPPORTED_SCORERS
from characterization.utils.common import SUPPORTED_SCENARIO_TYPES
from characterization.utils.io_utils import from_pickle, get_logger
from characterization.utils.viz.visualizer import BaseVisualizer

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

    # Instantiate dataset and visualizer
    cfg.dataset.config.load = False
    logger.info("Instatiating dataset: %s", cfg.dataset._target_)
    dataset = hydra.utils.instantiate(cfg.dataset)

    logger.info("Instatiating visualizer: %s", cfg.viz._target_)
    visualizer: BaseVisualizer = hydra.utils.instantiate(cfg.viz)

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
    scene_scores, _, scenario_scores = viz_utils.load_scenario_scores(
        scenario_ids,
        cfg.scenario_types,
        cfg.scores,
        cfg.criteria,
        Path(cfg.scores_path),
    )

    scene_scores_df = pd.DataFrame(scene_scores)
    # Generate scenario visualizations
    if cfg.viz_scenarios:
        # agent_scores_df = pd.DataFrame(agent_scores)
        logger.info("Visualizing scenarios based on scores")

        for key in scene_scores_df:
            if key.find("scenario_ids") != -1:
                continue
            scenario_type, criteria, scorer = key.split("_")
            prefix = f"{scenario_type}_{criteria}"
            scenarios_path = Path(cfg.output_dir) / prefix / scorer
            scenarios_path.mkdir(parents=True, exist_ok=True)

            # Visualize a few scenarios across various percentiles
            # Get score percentiles
            percentiles = np.percentile(scene_scores_df[key], cfg.percentiles)
            logger.info("Percentiles for %s: %s", key, percentiles)
            percentiles_low = np.append(scene_scores_df[key].min(), percentiles)
            percentiles_high = np.append(percentiles, scene_scores_df[key].max())
            percentile_ranges = zip(percentiles_low, percentiles_high, strict=False)

            for min_value, max_value in percentile_ranges:
                rows = viz_utils.get_sample_to_plot(
                    scene_scores_df,
                    str(key),
                    min_value,
                    max_value,
                    cfg.seed,
                    cfg.min_scenarios_to_plot,
                )
                if rows.empty:
                    logger.warning("No rows found for %s in range [%s, %s]", key, min_value, max_value)
                    continue

                for _, row in rows.iterrows():
                    scenario_id = row["scenario_ids"]
                    scores = scenario_scores[prefix][scenario_id]

                    # agent_scores = agent_scores_df[agent_scores_df["scenario_ids"] == scenario_id][key].to_numpy()
                    scenario_id = row["scenario_ids"].split(".")[0]

                    logger.info("Processing %s for scorer %s", scenario_id, key)
                    scenario_input_filepath = str(Path(cfg.paths.scenario_base_path) / f"sample_{scenario_id}.pkl")
                    scenario_data = from_pickle(scenario_input_filepath)  # nosec B301
                    scenario = dataset.transform_scenario_data(scenario_data)
                    _ = visualizer.visualize_scenario(scenario, scores=scores, output_dir=scenarios_path)


if __name__ == "__main__":
    run()  # pyright: ignore[reportCallIssue]
