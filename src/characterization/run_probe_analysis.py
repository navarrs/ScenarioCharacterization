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

Example usage::

    uv run python -m characterization.run_probe_analysis probe_csv=data/probed/constant_velocity/probe_summary.csv

    uv run python -m characterization.run_probe_analysis \\
        probe_csv=data/probed/constant_velocity/probe_summary.csv \\
        output_dir=data/probe_analysis dpi=200 add_timestamp=false
"""

from datetime import UTC, datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from characterization.utils import analysis
from characterization.utils.io_utils import get_logger, print_config

_LOGGER = get_logger(__name__)


@hydra.main(config_path="config", config_name="run_analysis", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """Run probe summary analysis and visualization.

    Reads the CSV produced by ``run_scenario_probing`` and generates a set of plots and a JSON
    summary in ``output_dir``.

    Args:
        cfg: Hydra configuration loaded from ``config/run_analysis.yaml``.
             Required CLI override: ``probe_csv=<path/to/probe_summary.csv>``.
    """
    print_config(cfg, theme="native")

    subdir = ""
    if cfg.add_timestamp:
        subdir = f"{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}_probe_analysis"
    subdir = f"{subdir}_{cfg.exp_tag}" if cfg.exp_tag else subdir
    output_dir = Path(cfg.output_dir) / subdir if subdir else Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(cfg.probe_csv)
    _LOGGER.info("Loading probe summary CSV: %s", csv_path)
    df = analysis.load_probe_csv(csv_path)
    df_probed = analysis.build_probed_df(df)

    n_total = len(df)
    n_probed = len(df_probed)
    _LOGGER.info("Loaded %d scenarios (%d probed, %d no probe)", n_total, n_probed, n_total - n_probed)

    dpi = int(cfg.get("dpi", 300))

    analysis.plot_probe_outcome_pie(df, output_dir, dpi)
    analysis.plot_score_distributions(df_probed, output_dir, dpi)
    analysis.plot_score_delta_density(df_probed, output_dir, dpi)
    analysis.plot_score_delta_by_agent_type(df_probed, output_dir, dpi)
    analysis.plot_score_scatter(df_probed, output_dir, dpi)
    analysis.plot_affected_agents_histogram(df_probed, output_dir, dpi)
    analysis.save_probe_summary_json(df, df_probed, output_dir)

    _LOGGER.info("Done. Results written to %s", output_dir)


if __name__ == "__main__":
    run()
