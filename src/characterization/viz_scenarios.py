from pathlib import Path

import hydra
from omegaconf import DictConfig

import characterization.utils.viz.utils as viz_utils
from characterization.scorer import SUPPORTED_SCORERS
from characterization.utils.common import SUPPORTED_SCENARIO_TYPES
from characterization.utils.io_utils import from_pickle, get_logger
from characterization.utils.viz.visualizer import BaseVisualizer

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

    # Generate scenario visualizations
    for scenario_id in scenario_ids[: cfg.total_scenarios]:
        logger.info("Visualizing scenario %s", scenario_id)
        # scenario_input_filepath = str(Path(cfg.paths.scenario_base_path) / f"sample_{scenario_id}.pkl")
        scenario_input_filepath = str(Path(cfg.paths.scenario_base_path) / f"kbos/{scenario_id}.pkl")
        scenario_data = from_pickle(scenario_input_filepath)  # nosec B301
        scenario = dataset.transform_scenario_data(scenario_data)
        _ = visualizer.visualize_scenario(scenario, output_dir=Path(cfg.output_dir) / "visualizations")

    # agent_scores_df = pd.DataFrame(agent_scores)
    logger.info("Visualizing scenarios based on scores")


if __name__ == "__main__":
    run()  # pyright: ignore[reportCallIssue]
