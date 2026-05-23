"""Processor that runs counterfactual probing over a dataset.

Integrates the probing loop from the legacy ``run_scenario_probing`` entrypoint into the standard
:class:`~characterization.processors.base_processor.BaseProcessor` pipeline so that
:mod:`characterization.run_processor` can be used as the single entrypoint for all characterization work.

Outputs are written under ``output_path/``:
  - Pickled scenarios  → ``probed_scenarios/<scenario_id>.pkl``
  - JSON probe summaries → ``probe_summaries/<scenario_id>.json``
  - Visualisations     → ``scenario_viz/``
  - CSV summary        → ``probe_summary.csv``

Example usage::

    uv run python -m characterization.run_processor characterizer=cvm_probe
    uv run python -m characterization.run_processor characterizer=cvm_probe viz=probe_scenario
    uv run python -m characterization.run_processor characterizer=cvm_probe num_scenarios=50
"""

import csv
import json
import pickle  # nosec B403
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from characterization.datasets import BaseDataset
from characterization.probing.base_prober import BaseProber
from characterization.processors.base_processor import BaseProcessor
from characterization.schemas.critical_probe import CriticalProbe
from characterization.schemas.scenario import Scenario
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)

CSV_FIELDS: tuple[str, ...] = (
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


class ProbeProcessor(BaseProcessor[BaseProber]):
    """Processor for running counterfactual probing across a dataset.

    Iterates over every scenario, applies a :class:`~characterization.probing.base_prober.BaseProber`,
    and persists the most-impactful probe (if any) together with a CSV summary.  An optional
    :class:`~characterization.utils.viz.scenario.ScenarioVisualizer` can be supplied via ``config.viz`` to
    generate per-scenario visualisations.
    """

    def __init__(self, config: DictConfig, dataset: BaseDataset, characterizer: BaseProber) -> None:
        """Initializes the ProbeProcessor.

        Args:
            config (DictConfig): Processor configuration. In addition to the base parameters (batch_size,
                num_workers, shuffle, save, overwrite, output_path, scenario_type) this accepts:

                * ``num_scenarios`` (int | None): Stop after this many scenarios.  ``null`` processes all.
                * ``scenario_id`` (str | None): When set, only the scenario with this ID is probed.
                * ``viz`` (DictConfig | None): Hydra-instantiable config for a
                  :class:`~characterization.utils.viz.scenario.ScenarioVisualizer`.  ``null`` disables viz.

            dataset (BaseDataset): Dataset to iterate over.
            characterizer (BaseProber): The counterfactual prober to apply to each scenario.

        Raises:
            ValueError: If ``save=True`` but ``output_path`` is not specified (raised by base class).
        """
        super().__init__(config, dataset, characterizer)

        # Output sub-directories (only meaningful when save=True; base class enforces output_path is set then).
        self.pkl_dir: Path | None = None
        self.json_dir: Path | None = None
        self.viz_dir: Path | None = None
        self.summary_csv_path: Path | None = None
        if self.output_path is not None:
            output = Path(self.output_path)
            self.pkl_dir = output / "probed_scenarios"
            self.json_dir = output / "probe_summaries"
            self.viz_dir = output / "scenario_viz"
            self.summary_csv_path = output / "probe_summary.csv"

        self.num_scenarios: int | None = config.get("num_scenarios", None)
        self.scenario_id: str | None = config.get("scenario_id", None)

        viz_cfg = config.get("viz", None)
        self.visualizer = hydra.utils.instantiate(viz_cfg) if viz_cfg is not None else None

    def run(self) -> None:
        """Runs counterfactual probing across every scenario in the dataset.

        For each scenario, applies the configured prober and — when a probe is found — saves a pickled
        :class:`~characterization.schemas.scenario.Scenario` (with ``critical_probe`` attached), a JSON summary,
        and an optional visualisation.  A CSV summary is written to ``output_path/probe_summary.csv`` at the end.

        Returns:
            None
        """
        logger.info("Probing %s with %s", self.dataset.name, self.characterizer.name)

        if self.save:
            for d in (self.pkl_dir, self.json_dir, self.viz_dir):
                if d is not None:
                    d.mkdir(parents=True, exist_ok=True)

        csv_rows: list[dict[str, str]] = []
        n_probed = 0
        n_total = 0

        for batch in tqdm(self.dataloader, total=len(self.dataloader), desc="Probing scenarios..."):
            for scenario in batch["scenario"]:
                if self.num_scenarios is not None and n_total >= self.num_scenarios:
                    break
                n_total += 1
                scenario_id: str = scenario.metadata.scenario_id
                if self.scenario_id is not None and scenario_id != str(self.scenario_id):
                    continue
                probe = self._probe_scenario_safe(scenario)
                csv_rows.append(_probe_to_csv_row(scenario_id, probe))
                if probe is not None:
                    n_probed += 1
                    self._persist_probe(scenario, probe, scenario_id)
            if self.num_scenarios is not None and n_total >= self.num_scenarios:
                break

        if csv_rows and self.save and self.summary_csv_path is not None:
            self.summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.summary_csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(csv_rows)
            logger.info("Probe summary CSV -> %s", self.summary_csv_path)

        logger.info("Done. %d / %d scenario(s) had an impactful probe.", n_probed, n_total)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _probe_scenario_safe(self, scenario: Scenario) -> CriticalProbe | None:
        """Apply the prober to one scenario, returning ``None`` on any exception."""
        scenario_id = scenario.metadata.scenario_id
        try:
            probe = self.characterizer.compute(scenario)
        except Exception:
            logger.exception("Failed to probe scenario %s", scenario_id)
            return None
        if probe is None:
            logger.info("No impactful probe for scenario %s", scenario_id)
        return probe

    def _persist_probe(self, scenario: Scenario, probe: CriticalProbe, scenario_id: str) -> None:
        """Attach the probe to the scenario, save outputs, and optionally visualise."""
        scenario.critical_probe = probe
        logger.info(
            "[%s] agent %s  before=%.4f  after=%.4f  delta=%+.4f",
            scenario_id,
            probe.probed_agent_id,
            probe.score_before,
            probe.score_after,
            probe.score_after - probe.score_before,
        )
        if self.save:
            self._save_scenario_pkl(scenario, scenario_id)
            self._save_probe_json(probe, scenario_id)

        if self.visualizer is not None and self.viz_dir is not None:
            try:
                self.visualizer.visualize_scenario(scenario, output_dir=self.viz_dir)
            except Exception:
                logger.exception("Visualisation failed for scenario %s", scenario_id)

    def _save_scenario_pkl(self, scenario: Scenario, scenario_id: str) -> None:
        """Pickle the full Scenario object (with probe attached)."""
        if self.pkl_dir is None:
            return
        out_path = self.pkl_dir / f"{scenario_id}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(scenario, f, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B301
        logger.debug("Saved probed scenario -> %s", out_path)

    def _save_probe_json(self, probe: CriticalProbe, scenario_id: str) -> None:
        """Save a JSON summary of the probe (excluding the large trajectory array)."""
        if self.json_dir is None:
            return
        out_path = self.json_dir / f"{scenario_id}.json"
        probe_dict = {k: v for k, v in probe.model_dump().items() if k != "probed_agent_trajectory"}
        out_path.write_text(json.dumps(probe_dict, indent=2))
        logger.debug("Saved probe JSON -> %s", out_path)
