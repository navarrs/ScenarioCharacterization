"""Counterfactual scenario probing entrypoint for autonomous driving scenarios.

For each scenario in the dataset, applies a constant-velocity counterfactual probe to every
ego/non-ego agent pair. The single most-impactful probe (if any exceeds ``min_score_delta``)
is attached to the ``Scenario.critical_probe`` field, the updated scenario is serialised, and an
optional JSON summary and visualisation are saved.

Outputs are written under ``probe_dir/<probing_strategy>/``:
  - Pickled scenarios  → ``probed_scenarios/<scenario_id>.pkl``
  - JSON probe summaries → ``probe_summaries/<scenario_id>.json``
  - Visualisations     → ``scenario_viz/``
  - CSV summary        → ``probe_summary.csv``

Example usage::

    uv run python -m characterization.run_scenario_probing probe_dir=data/probed
    uv run python -m characterization.run_scenario_probing probe_dir=data/probed num_scenarios=50
"""

import csv
import json
import pickle  # nosec B403
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from characterization.datasets import BaseDataset
from characterization.probing.prober import CounterfactualProber
from characterization.schemas import Scenario
from characterization.schemas.critical_probe import CriticalProbe
from characterization.utils.io_utils import get_logger, print_config
from characterization.utils.viz.scenario import ScenarioVisualizer

_LOGGER = get_logger(__name__)

_CSV_FIELDS: tuple[str, ...] = (
    "scenario_id",
    "probe_found",
    "probe_type",
    "is_ego_agent",
    "probed_agent_id",
    "score_before",
    "score_after",
    "score_delta",
    "affected_agent_ids",
)


def _probe_to_csv_row(scenario_id: str, probe: CriticalProbe | None) -> dict[str, str]:
    if probe is None:
        return {
            "scenario_id": scenario_id,
            "probe_found": "no",
            "probe_type": "N/A",
            "is_ego_agent": "N/A",
            "probed_agent_id": "",
            "score_before": "",
            "score_after": "",
            "score_delta": "",
            "affected_agent_ids": "",
        }
    return {
        "scenario_id": scenario_id,
        "probe_found": "yes",
        "probe_type": probe.probe_type.value,
        "is_ego_agent": "yes" if probe.is_ego_agent else "no",
        "probed_agent_id": str(probe.probed_agent_id),
        "score_before": f"{probe.score_before:.6f}",
        "score_after": f"{probe.score_after:.6f}",
        "score_delta": f"{probe.score_after - probe.score_before:.6f}",
        "affected_agent_ids": ";".join(str(a) for a in probe.affected_agent_ids),
    }


def _save_scenario(scenario: Scenario, output_dir: Path) -> Path:
    """Pickle the full Scenario object (with probe attached) to ``output_dir``."""
    out_path = output_dir / f"{scenario.metadata.scenario_id}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(scenario, f, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B301
    return out_path


def _save_probe_json(probe: CriticalProbe, scenario_id: str, probes_dir: Path) -> Path:
    out_path = probes_dir / f"{scenario_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclude the large trajectory array from the JSON summary.
    probe_dict = {k: v for k, v in probe.model_dump().items() if k != "probed_agent_trajectory"}
    out_path.write_text(json.dumps(probe_dict, indent=2))
    return out_path


@hydra.main(config_path="config", config_name="run_prober", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """Run counterfactual probing for every scenario in the configured dataset.

    Args:
        cfg: Hydra configuration. See ``config/scenario_probing.yaml`` for all parameters.
    """
    print_config(cfg, theme="native")

    probing_strategy = str(cfg.probing.probe_type).lower()
    strategy_dir = Path(cfg.probe_dir) / probing_strategy
    pkl_dir = strategy_dir / "probed_scenarios"
    json_dir = strategy_dir / "probe_summaries"
    viz_dir = strategy_dir / "scenario_viz"
    summary_csv_path = strategy_dir / "probe_summary.csv"
    for d in (pkl_dir, json_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)
    num_scenarios: int | None = cfg.get("num_scenarios")

    _LOGGER.info("Loading dataset: %s", cfg.dataset._target_)
    dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)

    dataloader = DataLoader(  # pyright: ignore[reportUnknownMemberType]
        dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=bool(cfg.get("shuffle", False)),
        num_workers=int(cfg.get("num_workers", 1)),
        collate_fn=dataset.collate_batch,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportArgumentType]
        persistent_workers=cfg.get("num_workers", 1) > 0,
    )

    probing_cfg_dict = OmegaConf.to_container(cfg.probing, resolve=True)
    probing_cfg = OmegaConf.create(probing_cfg_dict)
    assert isinstance(probing_cfg, DictConfig)
    prober = CounterfactualProber(probing_cfg)

    visualizer: ScenarioVisualizer = hydra.utils.instantiate(cfg.viz)

    csv_rows: list[dict[str, str]] = []
    n_probed = 0
    n_total = 0

    for batch in tqdm(dataloader, desc="Probing scenarios"):
        for scenario in batch["scenario"]:
            if num_scenarios is not None and n_total >= num_scenarios:
                break
            n_total += 1

            scenario_id = scenario.metadata.scenario_id
            try:
                probe = prober.probe_scenario(scenario)
            except Exception:
                _LOGGER.exception("Failed to probe scenario %s", scenario_id)
                csv_rows.append(_probe_to_csv_row(scenario_id, None))
                continue

            csv_rows.append(_probe_to_csv_row(scenario_id, probe))

            if probe is None:
                _LOGGER.info("No impactful probe for scenario %s", scenario_id)
                continue

            n_probed += 1
            scenario.critical_probe = probe

            saved_path = _save_scenario(scenario, pkl_dir)
            print(  # noqa: T201
                f"  [{scenario_id}] agent {probe.probed_agent_id}"
                f"  before={probe.score_before:.4f}"
                f"  after={probe.score_after:.4f}"
                f"  delta={probe.score_after - probe.score_before:+.4f}"
                f"  -> {saved_path}"
            )

            json_path = _save_probe_json(probe, scenario_id, json_dir)
            print(f"  Probe JSON -> {json_path}")  # noqa: T201

            try:
                viz_path = visualizer.visualize_scenario(scenario, output_dir=viz_dir)
                print(f"  Viz       -> {viz_path}")  # noqa: T201
            except Exception:
                _LOGGER.exception("Visualization failed for scenario %s", scenario_id)

        if num_scenarios is not None and n_total >= num_scenarios:
            break

    if csv_rows:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Summary CSV -> {summary_csv_path}")  # noqa: T201

    print(f"\n{'=' * 70}")  # noqa: T201
    _LOGGER.info("Done. %d / %d scenario(s) had an impactful probe.", n_probed, n_total)


if __name__ == "__main__":
    run()
