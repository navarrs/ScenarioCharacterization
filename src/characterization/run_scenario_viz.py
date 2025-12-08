from pathlib import Path

import hydra
from omegaconf import DictConfig

from characterization.schemas import ScenarioScores
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
    scenario_viz_dir = Path(cfg.scenario_viz_dir)
    scenario_viz_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate dataset and visualizer
    cfg.dataset.config.load = False
    logger.info("Instatiating dataset: %s", cfg.dataset._target_)
    dataset = hydra.utils.instantiate(cfg.dataset)

    logger.info("Instatiating visualizer: %s", cfg.viz._target_)
    visualizer: BaseVisualizer = hydra.utils.instantiate(cfg.viz)

    scenario_base_path = Path(cfg.paths.scenario_base_path)
    scenario_filepaths = list(scenario_base_path.rglob("*.pkl"))
    scores_path = Path(cfg.scores_path) / cfg.scores_tag
    if cfg.viz_scored_scenarios:
        valid_scenario_ids = [file.name for file in scores_path.glob("*.pkl")]
        scenario_filepaths = [fp for fp in scenario_filepaths if fp.name in valid_scenario_ids]

    scores = None
    for scenario_filepath in scenario_filepaths[: cfg.total_scenarios]:
        logger.info("Visualizing scenario %s", scenario_filepath)
        scenario_input_filepath = str(scenario_filepath)
        scenario_data = from_pickle(scenario_input_filepath)  # nosec B301
        scenario = dataset.transform_scenario_data(scenario_data)

        if cfg.viz_scored_scenarios:
            score_filepath = scores_path / scenario_filepath.name
            scenario_scores = from_pickle(str(score_filepath))  # nosec B301
            scenario_scores = ScenarioScores.model_validate(scenario_scores)
            match cfg.score_to_visualize:
                case "individual":
                    scores = scenario_scores.individual_scores
                case "interaction":
                    scores = scenario_scores.interaction_scores
                case "safeshift":
                    scores = scenario_scores.safeshift_scores
                case _:
                    scores = None

        _ = visualizer.visualize_scenario(scenario, scores=scores, output_dir=scenario_viz_dir)

    # agent_scores_df = pd.DataFrame(agent_scores)
    logger.info("Visualizing scenarios based on scores")


if __name__ == "__main__":
    run()  # pyright: ignore[reportCallIssue]
