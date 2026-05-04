"""Domain-agnostic scenario characterization runner.

Example usage:
    Run aviation domain with default config:
        uv run -m characterization.run_scenario_characterization

    Run AD domain:
        uv run -m characterization.run_scenario_characterization domain=ad

    Run with custom scenarios directory:
        uv run -m characterization.run_scenario_characterization scenarios_dir=data/mydata

    Increase parallelism and disable optional outputs:
        uv run -m characterization.run_scenario_characterization n_jobs=4 viz_dir=null summaries_dir=null
"""

import itertools
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
from pydantic import BaseModel

from characterization.schemas.scenario_scores import ScenarioScores
from characterization.utils.logging_utils import get_pylogger
from characterization.utils.scenario_runner import BaseScenarioRunner

_LOGGER = get_pylogger(__name__)


@hydra.main(config_path="config", config_name="run_scenario_characterization", version_base=None)
def main(cfg: DictConfig) -> None:
    """Load every scenario under ``cfg.scenarios_dir``, compute features and scores, print summaries, and write JSON.

    The output mirrors the input directory layout: for each ``{scenarios_dir}/{group}/{scenario_id}.pkl``:
    - Features are written to ``{output_dir}/{group}/{scenario_id}.json``
    - Scores are written to ``{scores_dir}/{group}/{scenario_id}.json``
    - Text summaries (if ``summaries_dir`` is set) → ``{summaries_dir}/{group}/{scenario_id}.txt``

    After all scenarios in a group are processed, per-group aggregate summaries and plots are written:
    - ``{summaries_dir}/{group}/group_summary.txt``
    - ``{plots_dir}/{group}/group_summary.png``

    Args:
        cfg: Hydra configuration. See ``config/run_scenario_characterization.yaml`` for all parameters.
    """
    scenarios_dir = Path(cfg.scenarios_dir)
    output_dir = Path(cfg.output_dir)
    scores_dir = Path(cfg.scores_dir)
    viz_dir = Path(cfg.viz_dir) if cfg.viz_dir else None
    summaries_dir = Path(cfg.summaries_dir) if cfg.summaries_dir else None
    plots_dir = Path(cfg.plots_dir) if cfg.plots_dir else None

    pkl_files = sorted(scenarios_dir.rglob("*.pkl"))
    if not pkl_files:
        _LOGGER.warning("No .pkl files found in %s", scenarios_dir)
        return

    _LOGGER.info("Found %d scenario(s) in %s", len(pkl_files), scenarios_dir)

    runner: BaseScenarioRunner = hydra.utils.instantiate(cfg.domain_runner)
    extractor = hydra.utils.instantiate(cfg.extractor)
    scorer = hydra.utils.instantiate(cfg.scorer)

    ok = 0
    # pkl_files is sorted, so scenarios for the same group are consecutive.
    for group_id, group in itertools.groupby(pkl_files, key=runner.group_key):
        group_feat: list[BaseModel] = []
        group_scor: list[ScenarioScores] = []

        for pkl_path in group:
            scenario = runner.load_scenario(pkl_path, group_id)
            if scenario is None:
                continue

            try:
                features: BaseModel = extractor.compute(scenario)  # type: ignore[union-attr]
            except Exception:
                _LOGGER.exception("Failed to compute features for %s", pkl_path.stem)
                continue

            try:
                scores: ScenarioScores = scorer.compute(scenario, features)  # type: ignore[union-attr]
            except Exception:
                _LOGGER.exception("Failed to compute scores for %s", pkl_path.stem)
                continue

            relative = pkl_path.relative_to(scenarios_dir)
            out_path = output_dir / relative.with_suffix(".json")
            scores_path = scores_dir / relative.with_suffix(".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            scores_path.parent.mkdir(parents=True, exist_ok=True)

            summary_text = runner.build_summary(scenario, pkl_path, features, scores, cfg.max_agents, cfg.max_pairs)
            print(summary_text, end="")  # noqa: T201

            out_path.write_text(json.dumps(features.model_dump(), indent=2))  # type: ignore[union-attr]
            scores_path.write_text(json.dumps(scores.model_dump(), indent=2))
            print(f"\n  Features -> {out_path}")  # noqa: T201
            print(f"  Scores   -> {scores_path}")  # noqa: T201

            if summaries_dir is not None:
                summary_path = summaries_dir / group_id / f"{pkl_path.stem}.txt"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(summary_text)
                print(f"  Summary  -> {summary_path}")  # noqa: T201

            if viz_dir is not None:
                runner.visualize(scenario, group_id, viz_dir, scores=scores)

            group_feat.append(features)
            group_scor.append(scores)
            ok += 1

        # Generate group-level summary and plot immediately after all its scenarios are done
        runner.generate_group_summaries(group_id, group_feat, group_scor, summaries_dir, plots_dir)

    print(f"\n{'=' * 70}")  # noqa: T201
    _LOGGER.info("Done. %d / %d scenario(s) processed successfully.", ok, len(pkl_files))


if __name__ == "__main__":
    main()
