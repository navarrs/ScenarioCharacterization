import random
from datetime import UTC, datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from characterization.schemas import ScenarioScores
from characterization.utils.io_utils import from_pickle, get_logger
from characterization.utils.viz.visualizer import BaseVisualizer

logger = get_logger(__name__)


def _organize_scenarios_by_percentile(
    cfg: DictConfig,
    scenario_filepaths: list[Path],
    scenario_viz_dir: Path,
) -> list[Path]:
    # Load scenario to score mapping file
    scenario_to_score_mapping_filepath = Path(cfg.scenario_to_score_mapping_filepath)
    assert scenario_to_score_mapping_filepath.exists(), (
        f"Scenario to score mapping file {scenario_to_score_mapping_filepath} does not exist."
    )
    scenario_to_score_df = pd.read_csv(scenario_to_score_mapping_filepath)
    score_column = f"{cfg.scores_tag}_{cfg.score_to_visualize}"

    # Compute percentiles and create corresponding subdirectories
    percentile_ranges = [0, *cfg.percentiles, 100]
    percentiles = np.percentile(scenario_to_score_df[score_column], percentile_ranges)
    subdirs = [
        f"percentile_{percentile_ranges[i - 1]}-{percentile_ranges[i]}" for i in range(1, len(percentile_ranges))
    ]
    for subdir in subdirs:
        percentile_dir = scenario_viz_dir / subdir
        percentile_dir.mkdir(parents=True, exist_ok=True)

    # Map each scenario to its corresponding output directory based on score percentiles
    scenario_output_dirs = []
    for scenario_filepath in scenario_filepaths:
        scenario_id = scenario_filepath.name
        score_row = scenario_to_score_df[scenario_to_score_df["scenario_ids"] == scenario_id]
        if score_row.empty:
            logger.warning("Scenario ID %s not found in score mapping. Assigning to 'unknown' directory.", scenario_id)
            output_dir = scenario_viz_dir / "unknown"
            output_dir.mkdir(parents=True, exist_ok=True)
            scenario_output_dirs.append(output_dir)
            continue

        score_value = score_row[score_column].to_numpy()[0]  # pyright: ignore[reportAttributeAccessIssue]
        for i in range(1, len(percentile_ranges)):
            if score_value < percentiles[i]:
                output_dir = scenario_viz_dir / subdirs[i - 1]
                scenario_output_dirs.append(output_dir)
                break
    return scenario_output_dirs


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
    random.seed(cfg.seed)
    date = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_")
    scenario_viz_dir = Path(cfg.scenario_viz_dir) / f"{date}_{cfg.scores_tag}_{cfg.score_to_visualize}"
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
        scenario_filepaths = [fp for fp in scenario_filepaths if Path(fp).name in valid_scenario_ids]

    scenario_output_dirs = (
        _organize_scenarios_by_percentile(cfg, scenario_filepaths, scenario_viz_dir)
        if cfg.organize_by_percentile
        else [scenario_viz_dir] * len(scenario_filepaths)
    )

    scores = None
    total_scenarios = (
        min(cfg.total_scenarios, len(scenario_filepaths)) if cfg.total_scenarios else len(scenario_filepaths)
    )
    for n, (scenario_filepath, output_dir) in enumerate(zip(scenario_filepaths, scenario_output_dirs, strict=False)):
        if n >= total_scenarios:
            break

        logger.info("Visualizing scenario %s", scenario_filepath)
        scenario_data = from_pickle(str(scenario_filepath))  # nosec B301
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

        _ = visualizer.visualize_scenario(scenario, scores=scores, output_dir=output_dir)

    # agent_scores_df = pd.DataFrame(agent_scores)
    logger.info("Visualizing scenarios based on scores")


if __name__ == "__main__":
    run()  # pyright: ignore[reportCallIssue]
