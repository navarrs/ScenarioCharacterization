r"""Entrypoint for analyzing and visualizing precomputed scenario features.

Loads individual and interaction features, regroups them by agent type and agent-pair type, and plots feature
distributions. Configuration is loaded from ``config/run_analysis.yaml`` by default.

Supports both single-dataset and multi-dataset analysis. In multi-dataset mode (``cfg.datasets`` is set),
the analysis runs independently for each dataset and also produces combined overlay plots comparing all
datasets on shared axes.

Example usage::

    uv run python -m characterization.run_feature_analysis
    uv run python -m characterization.run_feature_analysis features_path=/path/to/features output_dir=/path/to/output

    # Multi-dataset
    uv run python -m characterization.run_feature_analysis \\
        "datasets=[{label: Waymo, dataset_name: waymo}, {label: nuScenes, dataset_name: nuscenes}]"
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from characterization.utils import analysis, common
from characterization.utils.io_utils import get_logger
from characterization.utils.scenario_types import AgentPairType, AgentType

logger = get_logger(__name__)


def _load_and_regroup(cfg: DictConfig, features_path: Path) -> tuple[dict[AgentType, Any], dict[AgentPairType, Any]]:
    """Loads and regroups individual and interaction features from *features_path*.

    Uses ``cfg.scenario_types``, ``cfg.criteria``, and ``cfg.total_scenarios`` to control which
    scenarios are loaded. All datasets in a multi-dataset run share these settings.

    Args:
        cfg (DictConfig): Top-level configuration.
        features_path (Path): Path to the directory containing pre-computed feature files.

    Returns:
        Tuple of (individual_features_regrouped, interaction_features_regrouped).

    Raises:
        ValueError: If no valid scenarios are found under *features_path*.
    """
    scenario_types = list(cfg.scenario_types)
    criteria = list(cfg.criteria)

    scenario_ids = analysis.get_valid_scenario_ids(scenario_types, criteria, features_path)
    if not scenario_ids:
        msg = f"No valid scenarios found in {features_path} for {scenario_types} and criteria {criteria}"
        raise ValueError(msg)

    total_scenarios = (
        min(len(scenario_ids), cfg.total_scenarios)
        if cfg.total_scenarios and cfg.total_scenarios > 0
        else len(scenario_ids)
    )
    logger.info("Found %d valid scenarios for analysis. Using %d scenarios.", len(scenario_ids), total_scenarios)
    scenario_ids = scenario_ids[:total_scenarios]

    logger.info("Loading the features from %s", features_path)
    individual_features, interaction_features, _ = analysis.load_scenario_features(
        scenario_ids,
        scenario_types,
        criteria,
        features_path,
    )

    logger.info("Re-grouping individual features by agent type")
    individual_features = analysis.regroup_individual_features(individual_features)

    logger.info("Re-grouping interaction features by agent-pair type")
    interaction_features = analysis.regroup_interaction_features(interaction_features)

    return individual_features, interaction_features


@hydra.main(config_path="config", config_name="run_analysis", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """Runs the scenario feature analysis pipeline using the provided configuration.

    Loads precomputed individual and interaction features, regroups them by agent type and agent-pair type, and plots
    their distributions. When ``cfg.datasets`` is set, runs the analysis independently for each dataset and also
    produces combined overlay plots comparing all datasets on shared axes.

    Args:
        cfg (DictConfig): Configuration dictionary specifying feature paths, output options, and plot settings.

    Raises:
        ValueError: If unsupported scenario types are specified in the configuration.
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

    # Keyword arguments for per-dataset plots
    plot_kwargs: dict[str, Any] = {
        "dpi": cfg.dpi,
        "categories": list(cfg.feature_categories),
        "show_kde": cfg.show_kde,
        "show_percentiles": cfg.show_percentiles,
        "include_pairs_with_no_vehicles": cfg.include_pairs_with_no_vehicles,
    }
    # Keyword arguments for combined overlay plots (no percentile lines, no per-category config)
    combined_plot_kwargs: dict[str, Any] = {
        "dpi": cfg.dpi,
        "show_kde": cfg.show_kde,
        "include_pairs_with_no_vehicles": cfg.include_pairs_with_no_vehicles,
    }

    if cfg.datasets is None:
        # ------------------------------------------------------------------
        # Single-dataset path — identical behaviour to the original script
        # ------------------------------------------------------------------
        individual_features, interaction_features = _load_and_regroup(cfg, Path(cfg.features_path))

        logger.info("Visualizing feature distribution for individual features.")
        analysis.plot_feature_distributions(
            individual_features,
            output_dir,
            tag="individual",
            show_colored_by_agent_type=cfg.show_colored_by_agent_type,
            **plot_kwargs,
        )

        logger.info("Visualizing feature distribution for interaction features.")
        analysis.plot_feature_distributions(
            interaction_features,
            output_dir,
            tag="interaction",
            show_colored_by_agent_type=cfg.show_colored_by_agent_type,
            **plot_kwargs,
        )

    else:
        # ------------------------------------------------------------------
        # Multi-dataset path — per-dataset analysis + combined overlay plots
        # ------------------------------------------------------------------
        all_individual: dict[str, dict[AgentType, Any]] = {}
        all_interaction: dict[str, dict[AgentPairType, Any]] = {}

        for dataset_entry in cfg.datasets:
            label = dataset_entry.label
            # Derive features_path by substituting dataset_name into the resolved features_path template.
            # cfg.features_path is already interpolated (e.g. /data/.../waymo/cache/features), so replacing
            # the current dataset name with the target one gives the correct path for each dataset.
            features_path = Path(str(cfg.features_path).replace(cfg.paths.dataset_name, dataset_entry.dataset_name))
            ds_output_dir = output_dir / label
            ds_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Processing dataset: %s", label)
            individual_features, interaction_features = _load_and_regroup(cfg, features_path)

            logger.info("Visualizing individual feature distributions for dataset: %s", label)
            analysis.plot_feature_distributions(
                individual_features,
                ds_output_dir,
                tag="individual",
                show_colored_by_agent_type=cfg.show_colored_by_agent_type,
                **plot_kwargs,
            )

            logger.info("Visualizing interaction feature distributions for dataset: %s", label)
            analysis.plot_feature_distributions(
                interaction_features,
                ds_output_dir,
                tag="interaction",
                show_colored_by_agent_type=cfg.show_colored_by_agent_type,
                **plot_kwargs,
            )

            all_individual[label] = individual_features
            all_interaction[label] = interaction_features

        # Combined overlay plots across all datasets
        combined_dir = output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating combined overlay plots for %d datasets", len(all_individual))
        analysis.plot_multi_dataset_feature_distributions(
            all_individual,
            combined_dir,
            tag="individual",
            **combined_plot_kwargs,
        )
        analysis.plot_multi_dataset_feature_distributions(
            all_interaction,
            combined_dir,
            tag="interaction",
            **combined_plot_kwargs,
        )


if __name__ == "__main__":
    run()
