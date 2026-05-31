r"""Probe summary CSV analysis and visualization entrypoint.

Loads the ``probe_summary.csv`` produced by ``run_scenario_probing`` and generates a suite of statistical plots and a
JSON summary characterising probe outcomes across scenarios.

Outputs are written under ``output_dir`` (default: ``${paths.cache_path}/analysis``):
  - ``probe_outcome_pie.png``         — No probe / Ego probe / Non-ego probe breakdown
  - ``score_distributions.png``       — score_before vs score_after density curves
  - ``score_delta_density.png``       — distribution of score_delta with mean/median markers
  - ``score_delta_by_agent_type.png`` — score_delta violin split by ego vs non-ego
  - ``score_scatter.png``             — score_before vs score_after scatter with y=x reference
  - ``affected_agents_histogram.png`` — number of affected agents per probe, split by agent type
  - ``probe_analysis_summary.json``   — key aggregate statistics

Supports both single-dataset and multi-dataset analysis. In multi-dataset mode (``cfg.datasets`` is set), the analysis
runs independently for each dataset and also produces combined overlay plots comparing all datasets on shared axes
(under ``output_dir/combined/``).

Example usage::

    uv run python -m characterization.run_probe_analysis probe_csv=data/probed/constant_velocity/probe_summary.csv

    uv run python -m characterization.run_probe_analysis probe_csv=data/probed/constant_velocity/probe_summary.csv \\
        output_dir=data/probe_analysis dpi=200 add_timestamp=false

    # Multi-dataset
    uv run python -m characterization.run_probe_analysis \\
        "datasets=[{label: Waymo, dataset_name: waymo}, {label: nuScenes, dataset_name: nuscenes}]"
"""

from datetime import UTC, datetime
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from characterization.utils import analysis
from characterization.utils.io_utils import get_logger, print_config

_LOGGER = get_logger(__name__)


def _run_single_dataset(cfg: DictConfig, csv_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full per-dataset probe analysis pipeline and write all outputs to *output_dir*.

    Args:
        cfg: Hydra configuration (used for ``dpi``).
        csv_path: Path to the ``probe_summary.csv`` for this dataset.
        output_dir: Directory where plots and summary JSON are written.

    Returns:
        Tuple of ``(df, df_probed)`` — the full and probed-only DataFrames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("dpi", 300))

    _LOGGER.info("Loading probe summary CSV: %s", csv_path)
    df = analysis.load_probe_csv(csv_path)
    df_probed = analysis.build_probed_df(df)

    n_total = len(df)
    n_probed = len(df_probed)
    _LOGGER.info("Loaded %d scenarios (%d probed, %d no probe)", n_total, n_probed, n_total - n_probed)

    analysis.plot_probe_outcome_pie(df, output_dir, dpi)
    analysis.plot_score_distributions(df_probed, output_dir, dpi)
    analysis.plot_score_delta_density(df_probed, output_dir, dpi)
    analysis.plot_score_delta_by_agent_type(df_probed, output_dir, dpi)
    analysis.plot_score_scatter(df_probed, output_dir, dpi)
    analysis.plot_affected_agents_histogram(df_probed, output_dir, dpi)
    analysis.save_probe_summary_json(df, df_probed, output_dir)

    return df, df_probed


@hydra.main(config_path="config", config_name="run_analysis", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """Run probe summary analysis and visualization.

    Reads the CSV produced by ``run_scenario_probing`` and generates a set of plots and a JSON summary in
    ``output_dir``. When ``cfg.datasets`` is set, runs the analysis independently for each dataset and also produces
    combined overlay plots under ``output_dir/combined/``.

    Args:
        cfg: Hydra configuration loaded from ``config/run_analysis.yaml``.
             Single-dataset: ``probe_csv=<path/to/probe_summary.csv>``.
             Multi-dataset: ``datasets=[{label: X, dataset_name: x}, ...]``.
    """
    print_config(cfg, theme="native")

    subdir = ""
    if cfg.add_timestamp:
        subdir = f"{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}_probe_analysis"
    subdir = f"{subdir}_{cfg.exp_tag}" if cfg.exp_tag else subdir
    output_dir = Path(cfg.output_dir) / subdir if subdir else Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dpi = int(cfg.get("dpi", 300))

    # Single-dataset path — identical behaviour to the original script
    if cfg.datasets is None:
        _run_single_dataset(cfg, Path(cfg.probe_csv), output_dir)

    # Multi-dataset path — per-dataset analysis + combined overlay plots
    else:
        all_dfs: dict[str, pd.DataFrame] = {}
        all_dfs_probed: dict[str, pd.DataFrame] = {}

        for dataset_entry in cfg.datasets:
            label = dataset_entry.label
            # Derive per-dataset CSV path by substituting dataset_name into the resolved path template. cfg.probe_csv
            # is already interpolated (e.g. /data/.../waymo/cache/probes/.../probe_summary.csv), so replacing the
            # current dataset name with the target one gives the correct path.
            csv_path = Path(str(cfg.probe_csv).replace(cfg.paths.dataset_name, dataset_entry.dataset_name))
            ds_output_dir = output_dir / label

            _LOGGER.info("Processing dataset: %s", label)
            df, df_probed = _run_single_dataset(cfg, csv_path, ds_output_dir)
            all_dfs[label] = df
            all_dfs_probed[label] = df_probed

        # Combined overlay plots across all datasets
        combined_dir = output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        _LOGGER.info("Generating combined overlay plots for %d datasets", len(all_dfs))
        analysis.plot_multi_dataset_probe_outcomes(all_dfs, combined_dir, dpi)
        analysis.plot_multi_dataset_probe_score_distributions(all_dfs_probed, combined_dir, dpi)
        analysis.plot_multi_dataset_probe_score_distributions_grid(all_dfs_probed, combined_dir, dpi)
        analysis.plot_multi_dataset_probe_score_delta_density(all_dfs_probed, combined_dir, dpi)

    _LOGGER.info("Done. Results written to %s", output_dir)


if __name__ == "__main__":
    run()
