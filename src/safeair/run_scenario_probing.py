"""Counterfactual scenario probing runner for SafeAir.

For each scenario under ``scenarios_dir``, applies a constant-velocity counterfactual probe to every ego/other agent
pair. The single most-impactful probe (if any exceeds ``min_score_delta``) is added to the ``critical_probe`` field in
the ``Scenario`` schema, and the updated scenario is re-serialised.

Example usage:
    Run with default configs:
        uv run -m safeair.run_scenario_probing

    Run with custom scenarios directory:
        uv run -m safeair.run_scenario_probing scenarios_dir=data/mydata

    Increase parallelism and disable visualization:
        uv run -m safeair.run_scenario_probing num_workers=4 viz_dir=null
"""

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf

from safeair import utils
from safeair.scenario_characterization.probing import CounterfactualProber
from safeair.schemas.critical_probe import CriticalProbe
from safeair.schemas.scenario import MapData

if TYPE_CHECKING:
    from safeair.scenario_visualizer.scenario_visualizer import ScenarioVisualizer

_LOGGER = utils.get_pylogger(__name__)

_CSV_FIELDS: tuple[str, ...] = (
    "airport_id",
    "event_subfolder",
    "scenario_id",
    "probe_found",
    "probe_type",
    "is_ego_agent",
    "score_before",
    "score_after",
    "score_delta",
    "affected_agent_ids",
)


def _airport_from_path(pkl_path: Path, scenarios_dir: Path) -> str:
    return pkl_path.relative_to(scenarios_dir).parts[0]


def _event_subfolder_from_path(pkl_path: Path, scenarios_dir: Path) -> str:
    middle = pkl_path.relative_to(scenarios_dir).parts[1:-1]
    return str(Path(*middle)) if middle else ""


def _probe_to_csv_row(
    airport_id: str,
    event_subfolder: str,
    scenario_id: str,
    probe: CriticalProbe | None,
) -> dict[str, str]:
    if probe is None:
        return {
            "airport_id": airport_id,
            "event_subfolder": event_subfolder,
            "scenario_id": scenario_id,
            "probe_found": "no",
            "probe_type": "N/A",
            "is_ego_agent": "N/A",
            "score_before": "",
            "score_after": "",
            "score_delta": "",
            "affected_agent_ids": "",
        }
    return {
        "airport_id": airport_id,
        "event_subfolder": event_subfolder,
        "scenario_id": scenario_id,
        "probe_found": "yes",
        "probe_type": probe.probe_type.value,
        "is_ego_agent": "yes" if probe.is_ego_agent else "no",
        "score_before": f"{probe.score_before:.6f}",
        "score_after": f"{probe.score_after:.6f}",
        "score_delta": f"{probe.score_after - probe.score_before:.6f}",
        "affected_agent_ids": ";".join(str(a) for a in probe.affected_agent_ids),
    }


def _print_probe_summary(scenario_id: str, airport_id: str, probe_result: CriticalProbe | None) -> None:
    """Print a one-line probe result summary to stdout."""
    if probe_result is None:
        print(f"  [{airport_id}/{scenario_id}] no impactful probe found")  # noqa: T201
        return

    ego_label = "ego" if probe_result.is_ego_agent else "other"
    delta = probe_result.score_after - probe_result.score_before
    print(  # noqa: T201
        f"  [{airport_id}/{scenario_id}] agent {probe_result.probed_agent_id} ({ego_label})"
        f"  before={probe_result.score_before:.4f}"
        f"  after={probe_result.score_after:.4f}"
        f"  delta={delta:+.4f}"
        f"  affected={probe_result.affected_agent_ids}"
        f"  crit_ts={probe_result.criticality_results}"
    )


def _process_one_scenario(
    pkl_path: Path,
    maps_dir: Path | None,
    output_dir: Path,
    probes_dir: Path | None,
    viz_dir: Path | None,
    scenarios_dir: Path,
    probing_cfg: DictConfig,
    cfg: DictConfig,
) -> dict[str, str] | None:
    """Process a single scenario file. Returns a CSV row dict or None on load/probe failure."""
    airport_id = _airport_from_path(pkl_path, scenarios_dir)
    map_cache: dict[str, MapData | None] = {}
    viz_cache: dict[str, ScenarioVisualizer | None] = {}

    scenario = utils.load_scenario(pkl_path, maps_dir, airport_id, map_cache)
    if scenario is None:
        return None

    prober = CounterfactualProber(probing_cfg)
    try:
        probe = prober.probe_scenario(scenario)
    except Exception:
        _LOGGER.exception("Failed to probe scenario %s", pkl_path)
        return None

    event_subfolder = _event_subfolder_from_path(pkl_path, scenarios_dir)
    row = _probe_to_csv_row(airport_id, event_subfolder, scenario.metadata.scenario_id, probe)

    if probe is not None:
        scenario.critical_probe = probe
        _print_probe_summary(scenario.metadata.scenario_id, airport_id, probe)
        relative = pkl_path.relative_to(scenarios_dir)
        out_path = output_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scenario.to_pickle(out_path.parent)
        print(f"  Saved    -> {out_path}")  # noqa: T201

        if probes_dir is not None:
            probe_path = probes_dir / relative.with_suffix(".json")
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            probe_path.write_text(
                json.dumps(
                    {k: v for k, v in probe.model_dump().items() if k != "probed_agent_trajectory"},
                    indent=2,
                )
            )
            print(f"  Probe    -> {probe_path}")  # noqa: T201

        if viz_dir is not None:
            utils.visualize_scenario(scenario, airport_id, viz_dir, viz_cache, cfg)

    return row


@hydra.main(config_path="configs", config_name="scenario_probing", version_base=None)
def main(cfg: DictConfig) -> None:
    """Load every scenario under ``cfg.scenarios_dir``, apply counterfactual probing, and write results.

    For each ``{scenarios_dir}/{airport}/[event/]{scenario_id}.pkl``:
    - The updated scenario with ``critical_probe`` attached is written to ``{output_dir}/...``
    - A JSON summary of the probe is written to ``{probes_dir}/...`` (if the probe is not None)
    - A visualization is saved to ``{viz_dir}/{airport}/`` (if ``viz_dir`` is set)
    - A CSV row is appended to ``{summary_csv}`` for every scenario (probe found or not)

    Args:
        cfg: Hydra configuration. See ``configs/scenario_probing.yaml`` for all parameters.
    """
    scenarios_dir = Path(cfg.scenarios_dir)
    output_dir = Path(cfg.output_dir)
    maps_dir = Path(cfg.maps_dir) if cfg.maps_dir else None
    viz_dir = Path(cfg.viz_dir) if cfg.viz_dir else None
    probes_dir = Path(cfg.probes_dir) if cfg.probes_dir else None
    summary_csv_path = Path(cfg.summary_csv) if cfg.get("summary_csv") else None
    num_workers: int = int(cfg.get("num_workers", 1))

    pkl_files = sorted(scenarios_dir.rglob("*.pkl"))
    if not pkl_files:
        _LOGGER.warning("No .pkl files found in %s", scenarios_dir)
        return

    _LOGGER.info("Found %d scenario(s) in %s", len(pkl_files), scenarios_dir)

    # When running multiple scenario workers, clamp pair-level parallelism inside each worker to 1 to avoid nested
    # joblib processes and OOM issues.
    probing_cfg_dict = OmegaConf.to_container(cfg.probing, resolve=True)
    if num_workers > 1:
        probing_cfg_dict["n_jobs"] = 1  # type: ignore[index]
    _raw_cfg = OmegaConf.create(probing_cfg_dict)
    assert isinstance(_raw_cfg, DictConfig)
    probing_cfg = _raw_cfg

    csv_rows: list[dict[str, str]] = []
    ok = 0

    if num_workers > 1:
        raw_results: list[dict[str, str] | None] = Parallel(n_jobs=num_workers)(
            delayed(_process_one_scenario)(
                p, maps_dir, output_dir, probes_dir, viz_dir, scenarios_dir, probing_cfg, cfg
            )
            for p in pkl_files
        )  # type: ignore[assignment]
        csv_rows = [r for r in raw_results if r is not None]
        ok = sum(1 for r in csv_rows if r["probe_found"] == "yes")
    else:
        # Serial path: single prober instance + shared map/viz caches across scenarios
        prober = CounterfactualProber(probing_cfg)
        map_cache: dict[str, MapData | None] = {}
        viz_cache: dict[str, ScenarioVisualizer | None] = {}

        for pkl_path in pkl_files:
            airport_id = _airport_from_path(pkl_path, scenarios_dir)
            scenario = utils.load_scenario(pkl_path, maps_dir, airport_id, map_cache)
            if scenario is None:
                continue

            try:
                probe = prober.probe_scenario(scenario)
            except Exception:
                _LOGGER.exception("Failed to probe scenario %s", pkl_path)
                continue

            event_subfolder = _event_subfolder_from_path(pkl_path, scenarios_dir)
            csv_rows.append(_probe_to_csv_row(airport_id, event_subfolder, scenario.metadata.scenario_id, probe))

            if probe is None:
                _LOGGER.info("No useful probe found for scenario %s", pkl_path)
                continue

            scenario.critical_probe = probe
            _print_probe_summary(scenario.metadata.scenario_id, airport_id, probe)

            relative = pkl_path.relative_to(scenarios_dir)
            out_path = output_dir / relative
            out_path.parent.mkdir(parents=True, exist_ok=True)
            scenario.to_pickle(out_path.parent)
            print(f"  Saved    -> {out_path}")  # noqa: T201

            if probes_dir is not None:
                probe_path = probes_dir / relative.with_suffix(".json")
                probe_path.parent.mkdir(parents=True, exist_ok=True)
                probe_path.write_text(
                    json.dumps(
                        {k: v for k, v in probe.model_dump().items() if k != "probed_agent_trajectory"},
                        indent=2,
                    )
                )
                print(f"  Probe    -> {probe_path}")  # noqa: T201

            if viz_dir is not None:
                utils.visualize_scenario(scenario, airport_id, viz_dir, viz_cache, cfg)

            ok += 1

    if summary_csv_path is not None and csv_rows:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Summary  -> {summary_csv_path}")  # noqa: T201

    print(f"\n{'=' * 70}")  # noqa: T201
    _LOGGER.info("Done. %d / %d scenario(s) had an impactful probe.", ok, len(pkl_files))


if __name__ == "__main__":
    main()
