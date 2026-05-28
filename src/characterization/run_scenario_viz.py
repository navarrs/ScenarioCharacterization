"""Entrypoint for visualizing scenarios, optionally organized by score percentile.

Loads scenario pickle files, optionally filters to those with precomputed scores or probed scenarios, groups them into
score-percentile subdirectories, and renders visualizations using the configured dataset and visualizer. Configuration
is loaded from ``config/run_analysis.yaml`` by default.

The active viz config's ``panes_to_plot`` drives data loading automatically:

* ``COUNTERFACTUAL_PROBE`` in panes → scenarios are loaded from ``probed_scenarios_path`` (output of
  :class:`~characterization.processors.probe_processor.ProbeProcessor`), which embeds the probe in the
  :class:`~characterization.schemas.Scenario` object.
* ``CATEGORICAL_AGENTS`` in panes → per-scenario :class:`~characterization.schemas.ScenarioScores` are loaded from
  ``scores_path`` and passed to the visualizer.

Example usage::

    uv run python -m characterization.run_scenario_viz
    uv run python -m characterization.run_scenario_viz score_to_visualize=individual total_scenarios=50
    uv run python -m characterization.run_scenario_viz organize_by_percentile=true viz=categorical_scenario
    uv run python -m characterization.run_scenario_viz scenario_id=<scene_id>
    uv run python -m characterization.run_scenario_viz viz=all_panes_scenario total_scenarios=5
"""

import copy
import random
from datetime import UTC, datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from characterization.schemas import Scenario, ScenarioScores, Score
from characterization.utils.io_utils import from_pickle, get_logger
from characterization.utils.viz.visualizer import BaseVisualizer

logger = get_logger(__name__)

# Maps score_to_visualize values to the attribute name on ScenarioScores.
_SCORE_ATTR: dict[str, str] = {
    "individual": "individual_scores",
    "interaction": "interaction_scores",
    "safeshift": "safeshift_scores",
}

# Maps score_to_visualize values to their corresponding ground-truth categories file name.
_CATEGORIES_FILENAME: dict[str, str] = {
    "individual": "gt_critical_categorical_individual.json",
    "interaction": "gt_critical_categorical_interaction.json",
    "safeshift": "gt_critical_categorical_safeshift.json",
}


def _select_scores(scenario_scores: ScenarioScores, score_to_visualize: str) -> Score | None:
    """Return the appropriate :class:`Score` field from *scenario_scores* based on *score_to_visualize*.

    Args:
        scenario_scores: Validated :class:`ScenarioScores` object.
        score_to_visualize: One of ``"individual"``, ``"interaction"``, or ``"safeshift"``.

    Returns:
        The selected :class:`Score`, or ``None`` if *score_to_visualize* is not recognised.
    """
    attr = _SCORE_ATTR.get(score_to_visualize)
    return getattr(scenario_scores, attr) if attr is not None else None


def _organize_scenarios_by_percentile(
    cfg: DictConfig,
    scenario_filepaths: list[Path],
    scenario_viz_dir: Path,
) -> list[Path]:
    """Map each scenario filepath to an output subdirectory based on its score percentile bucket.

    Reads the scene-to-scores CSV specified in ``cfg.scenario_to_score_mapping_filepath``, computes percentile
    boundaries from ``cfg.percentiles``, and returns one output directory per scenario filepath.

    Args:
        cfg (DictConfig): Configuration containing score mapping path, percentile thresholds, and scoring column info.
        scenario_filepaths: Ordered list of scenario pickle file paths to organize.
        scenario_viz_dir: Root output directory under which percentile subdirectories are created.

    Returns:
        List of output directories, one per entry in ``scenario_filepaths``.
    """
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


def _get_probed_filepaths(cfg: DictConfig) -> list[Path] | None:
    """Load probed scenario filepaths from ``cfg.probed_scenarios_path``, filtering by ``cfg.scenario_id`` if set.

    Returns the list of paths, or ``None`` if the directory does not exist or a requested scenario is not found.
    """
    probed_path = Path(cfg.probed_scenarios_path)
    if not probed_path.exists():
        logger.error("Probed scenarios directory '%s' does not exist.", probed_path)
        return None
    filepaths = list(probed_path.glob("*.pkl"))
    logger.info("Loading %d probed scenarios from %s", len(filepaths), probed_path)
    if cfg.scenario_id:
        filepaths = [fp for fp in filepaths if fp.stem == cfg.scenario_id]
        if not filepaths:
            logger.error("No probed scenario found with ID '%s' under %s", cfg.scenario_id, probed_path)
            return None
        logger.info("Filtering to single probed scenario: %s", cfg.scenario_id)
    return filepaths


@hydra.main(config_path="config", config_name="run_analysis", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """Runs the scenario visualization pipeline using the provided configuration.

    Loads scenario pickle files, optionally filters to those with precomputed scores, and renders visualizations via
    the configured dataset and visualizer. When ``organize_by_percentile`` is enabled, scenarios are grouped into
    subdirectories corresponding to their score percentile bucket.

    Args:
        cfg (DictConfig): Configuration dictionary specifying dataset, visualizer, scenario paths, and output options.
    """
    random.seed(cfg.seed)
    date = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_")
    scenario_viz_dir = Path(cfg.scenario_viz_dir) / f"{date}{cfg.scores_tag}_{cfg.score_to_visualize}"
    scenario_viz_dir.mkdir(parents=True, exist_ok=True)

    # Derive data-loading behaviour from the active viz config's panes_to_plot.
    panes_to_plot = list(cfg.viz.config.get("panes_to_plot", ["ALL_AGENTS"]))
    viz_probed_scenarios = "COUNTERFACTUAL_PROBE" in panes_to_plot
    viz_scored_scenarios = "CATEGORICAL_AGENTS" in panes_to_plot

    # Instantiate dataset and visualizer
    cfg.dataset.config.load = False
    logger.info("Instatiating dataset: %s", cfg.dataset._target_)
    dataset = hydra.utils.instantiate(cfg.dataset)

    scenario_base_path = Path(cfg.paths.scenario_base_path)
    scenario_filepaths = list(scenario_base_path.rglob("*.pkl"))
    scores_path = Path(cfg.scores_path) / cfg.scores_tag

    if cfg.scenario_id and not viz_probed_scenarios:
        scenario_filepaths = [fp for fp in scenario_filepaths if fp.stem == cfg.scenario_id]
        if not scenario_filepaths:
            logger.error("No scenario found with ID '%s' under %s", cfg.scenario_id, scenario_base_path)
            return
        logger.info("Filtering to single scenario: %s", cfg.scenario_id)

    logger.info("Instatiating visualizer: %s", cfg.viz._target_)
    viz_config = copy.deepcopy(cfg.viz)
    if viz_scored_scenarios:
        valid_scenario_ids = [file.name for file in scores_path.glob("*.pkl")]
        scenario_filepaths = [fp for fp in scenario_filepaths if Path(fp).name in valid_scenario_ids]

        if (cat_filename := _CATEGORIES_FILENAME.get(cfg.score_to_visualize)) is not None:
            viz_config.config.categories_file = Path(cfg.meta_path) / cat_filename

    if viz_probed_scenarios:
        probed_filepaths = _get_probed_filepaths(cfg)
        if probed_filepaths is None:
            return
        scenario_filepaths = probed_filepaths

    visualizer: BaseVisualizer = hydra.utils.instantiate(viz_config)
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
        if viz_probed_scenarios:
            assert isinstance(scenario_data, Scenario), f"Expected Scenario, got {type(scenario_data)}"
            scenario = scenario_data
        else:
            scenario = dataset.transform_scenario_data(scenario_data)

        if viz_scored_scenarios:
            score_filepath = scores_path / scenario_filepath.name
            scenario_scores = ScenarioScores.model_validate(from_pickle(str(score_filepath)))  # nosec B301
            scores = _select_scores(scenario_scores, cfg.score_to_visualize)

        _ = visualizer.visualize_scenario(scenario, scores=scores, output_dir=output_dir)

    # agent_scores_df = pd.DataFrame(agent_scores)
    logger.info("Visualizing scenarios based on scores")


if __name__ == "__main__":
    run()
