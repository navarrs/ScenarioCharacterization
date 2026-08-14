r"""Entrypoint for counting trajectories and interactions per dataset from precomputed features.

Streams the cached feature artifacts and reports, per agent type and per agent-pair type, both the
population the pipeline enumerated and the subset it successfully characterized. Configuration is
loaded from ``config/run_analysis.yaml`` by default.

Supports both single-dataset and multi-dataset analysis. In multi-dataset mode (``cfg.datasets`` is
set), every dataset contributes one row to a shared table.

Example usage::
    uv run python -m characterization.run_dataset_analysis

    # Multi-dataset
    uv run python -m characterization.run_dataset_analysis \\
        "datasets=[{label: WOMD, dataset_name: waymo}, {label: nuPlan, dataset_name: nuplan}]"

    # Multi-dataset with LaTeX row labels. Single-quote each label so YAML keeps the backslashes.
    # Labels are passed through verbatim, so citation keys must match the manuscript's ref.bib.
    uv run python -m characterization.run_dataset_analysis output_dir=./outputs/dataset_analysis \\
        "datasets=[\\
          {label: WOMD, dataset_name: waymo, latex_label: '\womd~\cite{ettinger2021large}'}, \\
          {label: Argoverse2, dataset_name: argoverse2, latex_label: '\argoverse~\cite{wilson2021argoverse}'}, \\
          {label: nuPlan, dataset_name: nuplan, latex_label: '\nuplan~\cite{karnchanachari2024nuplan}'}]"
"""

from datetime import UTC, datetime
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from characterization.utils import analysis, common
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


def _count_dataset(
    cfg: DictConfig,
    label: str,
    dataset_name: str,
    features_path: Path,
    scenario_base_path: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Counts one dataset's trajectories and interactions and returns a single table row.

    Args:
        cfg (DictConfig): Top-level configuration.
        label (str): Display name for this dataset.
        dataset_name (str): Canonical dataset name, used to resolve plot colors independently of *label*.
        features_path (Path): Path to the directory containing this dataset's feature files.
        scenario_base_path (Path): Path to this dataset's raw scenario pickles, used only to report how
            many scenarios exist on disk versus how many were counted.

    Returns:
        tuple[dict[str, object], pd.DataFrame]: One row of the counts table, and the per-scenario counts
            behind it, which the table and the distribution plots use to report spread.

    Raises:
        ValueError: If no valid scenarios are found under *features_path*.
    """
    scenario_type, criterion = cfg.scenario_types[0], cfg.criteria[0]

    scenario_ids = analysis.get_valid_scenario_ids([scenario_type], [criterion], features_path)
    if not scenario_ids:
        msg = f"No valid scenarios found in {features_path} for {scenario_type} and criterion {criterion}"
        raise ValueError(msg)

    num_valid_scenarios = len(scenario_ids)
    scenario_ids = common.sample_scenarios(scenario_ids, cfg.total_scenarios, cfg.seed)
    logger.info(
        "Found %d valid scenarios for %s. Counting %d scenarios.", num_valid_scenarios, label, len(scenario_ids)
    )

    counts, per_scenario_counts = analysis.count_dataset_features(scenario_ids, scenario_type, criterion, features_path)

    row: dict[str, object] = {"dataset": label, "dataset_name": dataset_name}
    if cfg.count_scenarios_on_disk:
        row["num_scenarios_on_disk"] = analysis.count_scenarios_on_disk(scenario_base_path)
    row.update(counts)
    return row, per_scenario_counts


@hydra.main(config_path="config", config_name="run_analysis", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """Runs the dataset composition analysis using the provided configuration.

    Writes ``dataset_counts.csv``, ``dataset_counts.json`` and ``dataset_counts.tex`` with one row per
    dataset, plus grouped-bar count plots and 100%-stacked composition plots.

    Args:
        cfg (DictConfig): Configuration dictionary specifying feature paths and output options.

    Raises:
        ValueError: If unsupported scenario types are specified, or if more than one scenario type or
            criterion is selected.
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
    # Counting several branches would count every scenario once per branch.
    if len(cfg.scenario_types) != 1 or len(cfg.criteria) != 1:
        msg = (
            f"Dataset analysis counts exactly one scenario_type and one criterion, got "
            f"{list(cfg.scenario_types)} and {list(cfg.criteria)}. Re-run once per branch."
        )
        raise ValueError(msg)

    latex_labels: dict[str, str] = {}
    dataset_names: dict[str, str] = {}
    per_scenario_counts: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    if cfg.datasets is None:
        label = cfg.paths.dataset_name
        row, scenario_counts = _count_dataset(
            cfg,
            label,
            cfg.paths.dataset_name,
            Path(cfg.features_path),
            Path(cfg.paths.scenario_base_path),
        )
        rows.append(row)
        dataset_names[label] = cfg.paths.dataset_name
        per_scenario_counts[label] = scenario_counts
    else:
        for dataset_entry in cfg.datasets:
            # Derive per-dataset paths by substituting dataset_name into the resolved path templates.
            features_path = Path(str(cfg.features_path).replace(cfg.paths.dataset_name, dataset_entry.dataset_name))
            scenario_base_path = Path(
                str(cfg.paths.scenario_base_path).replace(cfg.paths.dataset_name, dataset_entry.dataset_name)
            )
            logger.info("Processing dataset: %s", dataset_entry.label)
            row, scenario_counts = _count_dataset(
                cfg, dataset_entry.label, dataset_entry.dataset_name, features_path, scenario_base_path
            )
            rows.append(row)
            dataset_names[dataset_entry.label] = dataset_entry.dataset_name
            per_scenario_counts[dataset_entry.label] = scenario_counts
            if "latex_label" in dataset_entry:
                latex_labels[dataset_entry.label] = dataset_entry.latex_label

    counts_df = pd.DataFrame(rows)

    output_filepath = output_dir / "dataset_counts.csv"
    counts_df.to_csv(output_filepath, index=False)
    logger.info("Saved dataset counts -> %s", output_filepath)

    analysis.save_dataset_counts_json(counts_df, output_dir)

    # Kept so the choice of summary statistic can be revisited without re-reading every feature pickle.
    per_scenario_df = pd.concat(
        [frame.assign(dataset=label) for label, frame in per_scenario_counts.items()], ignore_index=True
    )
    output_filepath = output_dir / "dataset_counts_per_scenario.csv"
    per_scenario_df.to_csv(output_filepath, index=False)
    logger.info("Saved per-scenario counts -> %s", output_filepath)

    agent_types, pair_types = list(cfg.latex_agent_types), list(cfg.latex_pair_types)
    analysis.write_latex_table(
        counts_df,
        output_dir / "dataset_counts.tex",
        agent_types,
        pair_types,
        latex_labels,
        per_scenario_counts if cfg.latex_per_scenario else None,
        compact_numbers=cfg.latex_compact_numbers,
    )
    analysis.plot_dataset_counts(counts_df, output_dir, agent_types, pair_types, cfg.dpi)
    analysis.plot_dataset_composition(counts_df, output_dir, agent_types, pair_types, cfg.dpi)
    analysis.plot_dataset_distributions(
        per_scenario_counts, output_dir, agent_types, pair_types, cfg.dpi, dataset_names
    )


if __name__ == "__main__":
    run()
