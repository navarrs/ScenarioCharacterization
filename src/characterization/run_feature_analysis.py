from datetime import UTC, datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from characterization.utils import analysis_utils, common
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

    features_path = Path(cfg.features_path)
    scenario_ids = analysis_utils.get_valid_scenario_ids(cfg.scenario_types, cfg.criteria, features_path)
    if not scenario_ids:
        msg = f"No valid scenarios found in {features_path} for {cfg.scenario_types} and criteria {cfg.criteria}"
        raise ValueError(msg)

    total_scenarios = (
        min(len(scenario_ids), cfg.total_scenarios)
        if cfg.total_scenarios and cfg.total_scenarios > 0
        else len(scenario_ids)
    )
    logger.info("Found %d valid scenarios for analysis. Using %d scenarios.", len(scenario_ids), total_scenarios)
    scenario_ids = scenario_ids[:total_scenarios]

    # Generate score histogram and density plot
    logger.info("Loading the features")
    individual_features, interaction_features = analysis_utils.load_scenario_features(
        scenario_ids,
        cfg.scenario_types,
        cfg.criteria,
        features_path,
    )

    logger.info("Re-grouping individual features by agent type")
    individual_features = analysis_utils.regroup_individual_features(individual_features)

    logger.info("Visualizing feature distribution for individual features.")
    analysis_utils.plot_feature_distributions(
        individual_features,
        output_dir,
        cfg.dpi,
        tag="individual",
        percentile_values=[10, 75, 90, 95, 99],
        show_kde=cfg.show_kde,
        show_percentiles=cfg.show_percentiles,
    )

    logger.info("Re-grouping interaction features by agent-pair type")
    interaction_features = analysis_utils.regroup_interaction_features(interaction_features)

    logger.info("Visualizing feature distribution for interaction features.")
    analysis_utils.plot_feature_distributions(
        interaction_features,
        output_dir,
        cfg.dpi,
        tag="interaction",
        percentile_values=[10, 75, 80, 90, 95, 99],
        show_kde=cfg.show_kde,
        show_percentiles=cfg.show_percentiles,
    )


if __name__ == "__main__":
    run()  # pyright: ignore[reportCallIssue]
