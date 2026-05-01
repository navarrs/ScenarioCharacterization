"""Scenario characterization runner for SafeAir.

Example usage:
    Run with default configs:
        uv run -m safeair.run_scenario_characterization

    Run with custom scenarios directory:
        uv run -m safeair.run_scenario_characterization scenarios_dir=data/mydata

    Increase parallelism and disable optional outputs:
        uv run -m safeair.run_scenario_characterization n_jobs=4 viz_dir=null maps_dir=null
"""

import itertools
import json
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig

from safeair import utils
from safeair.scenario_characterization.features import SafeAirFeatures
from safeair.scenario_characterization.scores import SafeAirScorer
from safeair.schemas.scenario import MapData, Scenario
from safeair.schemas.scenario_features import ScenarioFeatures
from safeair.schemas.scenario_scores import ScenarioScores

if TYPE_CHECKING:
    from safeair.scenario_visualizer.scenario_visualizer import ScenarioVisualizer

_LOGGER = utils.get_pylogger(__name__)


def _compute_features_and_scores(
    scenario: Scenario,
    extractor: SafeAirFeatures,
    scorer: SafeAirScorer,
) -> tuple[ScenarioFeatures, ScenarioScores] | None:
    """Compute features and scores for *scenario*, returning None on failure."""
    try:
        features = extractor.compute(scenario)
    except Exception:
        _LOGGER.exception("Failed to compute features for %s", scenario.metadata.scenario_id)
        return None

    try:
        scores = scorer.compute(scenario, features)
    except Exception:
        _LOGGER.exception("Failed to compute scores for %s", scenario.metadata.scenario_id)
        return None

    return features, scores


def _save_optional_outputs(
    scenario: Scenario,
    features: ScenarioFeatures,
    airport_id: str,
    summary_text: str,
    summaries_dir: Path | None,
    plots_dir: Path | None,
) -> None:
    """Write optional per-scenario text summary and feature distribution plot."""
    if summaries_dir is not None:
        summary_path = summaries_dir / airport_id / f"{scenario.metadata.scenario_id}.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text)
        print(f"  Summary  -> {summary_path}")  # noqa: T201

    if plots_dir is not None:
        plot_path = plots_dir / "scenarios" / airport_id / f"{scenario.metadata.scenario_id}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            utils.plot_features(
                [features],
                plot_path,
                title=f"Scenario {scenario.metadata.scenario_id} - {airport_id}",
            )
            print(f"  Plot     -> {plot_path}\n")  # noqa: T201
        except Exception:
            _LOGGER.exception("Failed to plot features for scenario %s", scenario.metadata.scenario_id)


@hydra.main(config_path="configs", config_name="scenario_characterization", version_base=None)
def main(cfg: DictConfig) -> None:
    """Load every scenario under ``cfg.scenarios_dir``, compute features and scores, print summaries, and write JSON.

    The output mirrors the input directory layout: for each ``{scenarios_dir}/{airport}/{scenario_id}.pkl``:
    - Features are written to ``{output_dir}/{airport}/{scenario_id}.json``
    - Scores are written to ``{scores_dir}/{airport}/{scenario_id}.json``
    - Text summaries (if ``summaries_dir`` is set) -> ``{summaries_dir}/{airport}/{scenario_id}.txt``
    - Feature plots (if ``plots_dir`` is set) -> ``{plots_dir}/{airport}/{scenario_id}.png``

    After all scenarios are processed, per-airport aggregate summaries and plots are also written.

    Args:
        cfg: Hydra configuration. See ``configs/scenario_characterization.yaml`` for all parameters.
    """
    scenarios_dir = Path(cfg.scenarios_dir)
    output_dir = Path(cfg.output_dir)
    scores_dir = Path(cfg.scores_dir)
    maps_dir = Path(cfg.maps_dir) if cfg.maps_dir else None
    viz_dir = Path(cfg.viz_dir) if cfg.viz_dir else None
    summaries_dir = Path(cfg.summaries_dir) if cfg.summaries_dir else None
    plots_dir = Path(cfg.plots_dir) if cfg.plots_dir else None

    pkl_files = sorted(scenarios_dir.rglob("*.pkl"))
    if not pkl_files:
        _LOGGER.warning("No .pkl files found in %s", scenarios_dir)
        return

    _LOGGER.info("Found %d scenario(s) in %s", len(pkl_files), scenarios_dir)

    extractor = SafeAirFeatures(n_jobs=cfg.n_jobs)
    scorer = SafeAirScorer()

    map_cache: dict[str, MapData | None] = {}
    viz_cache: dict[str, ScenarioVisualizer | None] = {}

    ok = 0
    # pkl_files is sorted, so scenarios for the same airport are consecutive.
    for airport_id, group in itertools.groupby(pkl_files, key=lambda p: p.parent.name):
        airport_feat: list[ScenarioFeatures] = []
        airport_scor: list[ScenarioScores] = []

        for pkl_path in group:
            scenario = utils.load_scenario(pkl_path, maps_dir, airport_id, map_cache)
            if scenario is None:
                continue

            result = _compute_features_and_scores(scenario, extractor, scorer)
            if result is None:
                continue
            features, scores = result

            relative = pkl_path.relative_to(scenarios_dir)
            out_path = output_dir / relative.with_suffix(".json")
            scores_path = scores_dir / relative.with_suffix(".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            scores_path.parent.mkdir(parents=True, exist_ok=True)

            summary_text = utils.build_scenario_summary(
                scenario, pkl_path, features, scores, cfg.max_agents, cfg.max_pairs
            )
            print(summary_text, end="")  # noqa: T201

            out_path.write_text(json.dumps(features.model_dump(), indent=2))
            scores_path.write_text(json.dumps(scores.model_dump(), indent=2))
            print(f"\n  Features -> {out_path}")  # noqa: T201
            print(f"  Scores   -> {scores_path}")  # noqa: T201

            _save_optional_outputs(scenario, features, airport_id, summary_text, summaries_dir, plots_dir)
            if viz_dir is not None:
                utils.visualize_scenario(scenario, airport_id, viz_dir, viz_cache, cfg, scores=scores)

            airport_feat.append(features)
            airport_scor.append(scores)
            ok += 1

        # Generate airport-level summary and plot immediately after all its scenarios are done
        utils.generate_airport_summaries(
            {airport_id: airport_feat}, {airport_id: airport_scor}, summaries_dir, plots_dir
        )

    print(f"\n{'=' * 70}")  # noqa: T201
    _LOGGER.info("Done. %d / %d scenario(s) processed successfully.", ok, len(pkl_files))


if __name__ == "__main__":
    main()
