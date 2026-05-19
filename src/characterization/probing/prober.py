"""Counterfactual probing orchestrator for autonomous driving scenarios."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from characterization.features.interaction_features import InteractionFeatures, InteractionStatus
from characterization.probing.common import (
    CandidateProbeResult,
    CriticalityMetric,
    ProbeType,
    ProbeValidity,
    ValidatorType,
)
from characterization.probing.probes import constant_velocity_probe
from characterization.probing.validators import find_criticality_timestamp, max_score_delta_validator
from characterization.schemas.critical_probe import CriticalityResult, CriticalProbe
from characterization.schemas.detections import FeatureDetections, FeatureWeights
from characterization.schemas.scenario import Scenario, ScenarioMetadata
from characterization.schemas.scenario_features import Interaction
from characterization.scorer.score_functions import simple_interaction_score
from characterization.utils.common import EPSILON, AgentTrajectoryMasker
from characterization.utils.io_utils import get_logger
from characterization.utils.scenario_types import AgentType

_LOGGER = get_logger(__name__)

ProbeFn = Callable[[AgentTrajectoryMasker, int, float], tuple[NDArray[np.float32], ProbeValidity]]
ValidatorFn = Callable[[list[CandidateProbeResult]], int | None]

_PROBE_REGISTRY: dict[ProbeType, ProbeFn] = {
    ProbeType.CONSTANT_VELOCITY: constant_velocity_probe,
}

_VALIDATOR_REGISTRY: dict[ValidatorType, ValidatorFn] = {
    ValidatorType.MAX_SCORE_DELTA: max_score_delta_validator,
}


def _pair_key(id_a: int, id_b: int) -> tuple[int, int]:
    return (id_a, id_b) if id_a < id_b else (id_b, id_a)


class _ProbingBaseline(NamedTuple):
    """Pre-computed unperturbed baseline used by the probe selection loop.

    Attributes:
        interaction: Full ``Interaction`` object for the unperturbed scenario (may be ``None`` if the scenario has fewer
            than two agents).
        scene_score: Interaction scene score for the unperturbed scenario.
        pair_score_cache: Raw pair scores keyed by canonical ``(min_id, max_id)`` for O(1) delta lookup.
        id_to_idx: Maps agent ID to its index in ``scenario.agent_data.agent_ids``.
        id_to_type: Maps agent ID to its ``AgentType``, used to filter out skipped-type pairs.
    """

    interaction: Interaction | None
    scene_score: float
    pair_score_cache: dict[tuple[int, int], float]
    id_to_idx: dict[int, int]
    id_to_type: dict[int, AgentType]


class CounterfactualProber:
    """Runs counterfactual probing for ego-vs-others and others-vs-ego agent pairs.

    For each agent, the probe function replaces its future trajectory with a counterfactual. The probe with the greatest
    pair-score increase (if any exceed ``min_score_delta``) is selected, and its full interaction scene score is
    computed once. Agents whose type appears in ``config.skip_agent_types`` are never probed.

    Args:
        config: Probing configuration. See ``config/probing/default.yaml`` for all parameters.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the prober from a Hydra config."""
        self._probe_type = ProbeType[config.probe_type]
        self._probe_fn = _PROBE_REGISTRY[self._probe_type]

        self._validator_type = ValidatorType[config.validator_type]
        self._validator_fn = _VALIDATOR_REGISTRY[self._validator_type]

        self._min_score_delta: float = float(config.min_score_delta)
        self._skip_agent_types: set[AgentType] = {AgentType[t] for t in config.get("skip_agent_types", [])}
        self._single_affected_agent: bool = bool(config.get("single_affected_agent", True))

        extractor_cfg = OmegaConf.create(
            {
                "return_criterion": config.get("return_criterion", "critical"),
                "feature_type": "continuous",
                "inv_stability_cap": 10.0,
                "compute_agent_to_agent_closest_dists": False,
            }
        )
        self._extractor = InteractionFeatures(extractor_cfg)
        self._detections = FeatureDetections()
        self._weights = FeatureWeights()

    def probe_scenario(self, scenario: Scenario) -> CriticalProbe | None:
        """Run all counterfactual probes and return the most impactful result.

        Probes the ego agent against all others and each non-ego agent against the ego. Non-ego vs
        non-ego pairs are not probed to avoid combinatorial blowup.

        Args:
            scenario: The scenario to probe.

        Returns:
            The :class:`~characterization.schemas.critical_probe.CriticalProbe` with the highest pair-score delta,
            or ``None`` if no probe exceeds ``min_score_delta``.
        """
        baseline = self._compute_baseline(scenario)
        if baseline.interaction is None:
            _LOGGER.warning("Scenario %s has fewer than 2 agents; skipping probe.", scenario.metadata.scenario_id)
            return None
        return self._find_critical_probe(scenario, baseline)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_pair_from_dict(self, feature_dict: dict[str, float]) -> float:
        d = self._detections
        w = self._weights
        return simple_interaction_score(
            collision=feature_dict.get("collision", 0.0),
            collision_weight=w.collision,
            collision_detection=d.collision,
            inv_mttcp=feature_dict.get("inv_mttcp", 0.0),
            inv_mttcp_weight=w.mttcp,
            inv_mttcp_detection=1.0 / (d.mttcp + EPSILON),
            inv_thw=feature_dict.get("inv_thw", 0.0),
            inv_thw_weight=w.thw,
            inv_thw_detection=1.0 / (d.thw + EPSILON),
            inv_ttc=feature_dict.get("inv_ttc", 0.0),
            inv_ttc_weight=w.ttc,
            inv_ttc_detection=1.0 / (d.ttc + EPSILON),
            drac=feature_dict.get("drac", 0.0),
            drac_weight=w.drac,
            drac_detection=d.drac,
        )

    @staticmethod
    def _interaction_to_feature_dict(interaction: Interaction, n: int) -> dict[str, float]:
        def _safe(arr: NDArray[np.float32] | None, idx: int) -> float:
            if arr is None:
                return 0.0
            v = float(arr[idx])
            return 0.0 if not np.isfinite(v) else v

        return {
            "collision": _safe(interaction.collision, n),
            "inv_mttcp": _safe(interaction.inv_mttcp, n),
            "inv_thw": _safe(interaction.inv_thw, n),
            "inv_ttc": _safe(interaction.inv_ttc, n),
            "drac": _safe(interaction.drac, n),
        }

    def _compute_scene_score(self, pair_score_cache: dict[tuple[int, int], float]) -> float:
        scores = list(pair_score_cache.values())
        return sum(scores) / max(len(scores), 1) if scores else 0.0

    def _compute_baseline(self, scenario: Scenario) -> _ProbingBaseline:
        interaction = self._extractor.compute_interaction_features(scenario, max_workers=1)
        agent_ids = scenario.agent_data.agent_ids
        agent_types = scenario.agent_data.agent_types

        id_to_idx = {int(aid): idx for idx, aid in enumerate(agent_ids)}
        id_to_type = {int(aid): atype for aid, atype in zip(agent_ids, agent_types, strict=True)}

        pair_score_cache: dict[tuple[int, int], float] = {}
        if (
            interaction is not None
            and interaction.interaction_agent_indices is not None
            and interaction.interaction_status is not None
        ):
            for n, (i, j) in enumerate(interaction.interaction_agent_indices):
                status = interaction.interaction_status[n]
                if status not in (InteractionStatus.COMPUTED_OK, InteractionStatus.PARTIAL_INVALID_HEADING):
                    continue
                id_a, id_b = int(agent_ids[i]), int(agent_ids[j])
                if id_to_type.get(id_a) in self._skip_agent_types or id_to_type.get(id_b) in self._skip_agent_types:
                    continue
                feature_dict = self._interaction_to_feature_dict(interaction, n)
                pair_score_cache[_pair_key(id_a, id_b)] = self._score_pair_from_dict(feature_dict)

        return _ProbingBaseline(
            interaction=interaction,
            scene_score=self._compute_scene_score(pair_score_cache),
            pair_score_cache=pair_score_cache,
            id_to_idx=id_to_idx,
            id_to_type=id_to_type,
        )

    def _get_affected_index_pairs(
        self,
        interaction: Interaction,
        id_to_type: dict[int, AgentType],
        probed_idx: int,
        ego_idx: int,
        agent_ids: list[int],
        *,
        is_ego: bool,
    ) -> list[tuple[int, int]]:
        """Return (i, j) agent-index pairs from the baseline that need recomputation for this probe.

        For an ego probe, all pairs where the ego index appears are recomputed. For a non-ego probe,
        only the single ``(ego_idx, probed_idx)`` pair is recomputed.
        """
        result: list[tuple[int, int]] = []
        for i, j in interaction.interaction_agent_indices or []:
            if probed_idx not in (i, j):
                continue
            if not is_ego and ego_idx not in (i, j):
                continue
            id_a, id_b = int(agent_ids[i]), int(agent_ids[j])
            if id_to_type.get(id_a) in self._skip_agent_types or id_to_type.get(id_b) in self._skip_agent_types:
                continue
            result.append((i, j))
        return result

    def _score_affected_pairs(
        self,
        pair_results: list[tuple[tuple[int, int], InteractionStatus, dict[str, float] | None]],
        baseline: _ProbingBaseline,
        probed_agent_id: int,
        agent_ids: list[int],
    ) -> tuple[float, list[int], dict[str, float], dict[str, float]]:
        """Score each recomputed pair and return delta + affected IDs + before/after score dicts."""
        pair_delta: float = 0.0
        affected_ids: list[int] = []
        scores_before: dict[str, float] = {}
        scores_after: dict[str, float] = {}

        for (i, j), status, feature_dict in pair_results:
            if status not in (InteractionStatus.COMPUTED_OK, InteractionStatus.PARTIAL_INVALID_HEADING):
                continue
            if feature_dict is None:
                continue
            id_a, id_b = int(agent_ids[i]), int(agent_ids[j])
            other_id = id_b if id_a == probed_agent_id else id_a
            if baseline.id_to_type.get(other_id) in self._skip_agent_types:
                continue

            canon = _pair_key(id_a, id_b)
            before = baseline.pair_score_cache.get(canon, 0.0)
            after = self._score_pair_from_dict(feature_dict)
            delta = after - before
            pair_delta = max(pair_delta, delta)

            if delta >= self._min_score_delta:
                key = f"{canon[0]}:{canon[1]}"
                scores_before[key] = before
                scores_after[key] = after
                if other_id not in affected_ids:
                    affected_ids.append(other_id)

        return pair_delta, affected_ids, scores_before, scores_after

    def _find_criticality_timestamps(
        self,
        probed_traj: NDArray[np.float32],
        all_trajs: NDArray[np.float32],
        candidate_ids: list[int],
        agent_ids: list[int],
        metadata: ScenarioMetadata,
        current_time_index: int,
    ) -> dict[int, CriticalityResult]:
        results: dict[int, CriticalityResult] = {}
        for cid in candidate_ids:
            matches = [idx for idx, aid in enumerate(agent_ids) if int(aid) == cid]
            if not matches:
                continue
            result = find_criticality_timestamp(
                AgentTrajectoryMasker(probed_traj),
                AgentTrajectoryMasker(all_trajs[matches[0]]),
                metadata,
                current_time_index,
            )
            if result is not None:
                results[cid] = result
        return results

    def _build_probe_result(
        self,
        scenario: Scenario,
        baseline: _ProbingBaseline,
        best_agent_id: int,
        best_is_ego: bool,  # noqa: FBT001
        best_perturbed_traj: NDArray[np.float32],
        best_affected_pair_results: list[tuple[tuple[int, int], InteractionStatus, dict[str, float] | None]],
        best_affected_ids: list[int],
        best_pair_scores_before: dict[str, float],
        best_pair_scores_after: dict[str, float],
    ) -> CriticalProbe | None:
        """Build the final CriticalProbe from the winner candidate. Returns None if no criticality is found."""
        agent_ids = scenario.agent_data.agent_ids
        trajs = scenario.agent_data.agent_trajectories
        current_time_index = scenario.metadata.current_time_index

        winner_pair_scores: dict[tuple[int, int], float] = {}
        for (vi, vj), status, fd in best_affected_pair_results:
            if status not in (InteractionStatus.COMPUTED_OK, InteractionStatus.PARTIAL_INVALID_HEADING):
                continue
            if fd is None:
                continue
            id_a = int(agent_ids[vi])
            id_b = int(agent_ids[vj])
            winner_pair_scores[_pair_key(id_a, id_b)] = self._score_pair_from_dict(fd)
        scene_score_after = self._compute_scene_score({**baseline.pair_score_cache, **winner_pair_scores})

        id_to_crit = self._find_criticality_timestamps(
            best_perturbed_traj, trajs, best_affected_ids, list(agent_ids), scenario.metadata, current_time_index
        )
        if not id_to_crit:
            return None

        if self._single_affected_agent:
            best_aid, best_crit = min(
                id_to_crit.items(),
                key=lambda item: (0 if item[1].metric == CriticalityMetric.TTC else 1, item[1].timestamp),
            )
            id_to_crit = {best_aid: best_crit}

        affected_ids = list(id_to_crit.keys())
        final_scores_before: dict[str, float] = {}
        final_scores_after: dict[str, float] = {}
        for key, score_before in best_pair_scores_before.items():
            a_str, b_str = key.split(":")
            other_id = int(b_str) if int(a_str) == best_agent_id else int(a_str)
            if other_id in id_to_crit:
                final_scores_before[key] = score_before
                final_scores_after[key] = best_pair_scores_after[key]

        return CriticalProbe(
            probed_agent_id=best_agent_id,
            probed_agent_trajectory=best_perturbed_traj,
            is_ego_agent=best_is_ego,
            probe_type=self._probe_type,
            affected_agent_ids=affected_ids,
            criticality_results={str(aid): r for aid, r in id_to_crit.items()},
            score_before=baseline.scene_score,
            score_after=scene_score_after,
            affected_pair_scores_before=final_scores_before,
            affected_pair_scores_after=final_scores_after,
        )

    def _find_critical_probe(self, scenario: Scenario, baseline: _ProbingBaseline) -> CriticalProbe | None:
        assert baseline.interaction is not None

        ego_idx = scenario.metadata.ego_vehicle_index
        current_time_index = scenario.metadata.current_time_index
        frequency_hz = scenario.metadata.frequency_hz
        agent_ids = scenario.agent_data.agent_ids
        trajs = scenario.agent_data.agent_trajectories

        candidates: list[CandidateProbeResult] = []
        candidate_pair_results: list[list[tuple[tuple[int, int], InteractionStatus, dict[str, float] | None]]] = []

        for i in range(scenario.agent_data.num_agents):
            agent_id = int(agent_ids[i])
            if baseline.id_to_type.get(agent_id) in self._skip_agent_types:
                continue

            perturbed_traj, validity = self._probe_fn(AgentTrajectoryMasker(trajs[i]), current_time_index, frequency_hz)
            if validity == ProbeValidity.INVALID:
                continue

            is_ego = i == ego_idx
            affected_idx_pairs = self._get_affected_index_pairs(
                baseline.interaction,
                baseline.id_to_type,
                i,
                ego_idx,
                list(agent_ids),
                is_ego=is_ego,
            )
            if not affected_idx_pairs:
                continue

            perturbed_trajs = trajs.copy()
            perturbed_trajs[i] = perturbed_traj
            perturbed_scenario = scenario.model_copy(
                update={"agent_data": scenario.agent_data.model_copy(update={"agent_trajectories": perturbed_trajs})}
            )
            pair_results = self._extractor.compute_pairs(perturbed_scenario, affected_idx_pairs)

            pair_delta, affected_ids, scores_before, scores_after = self._score_affected_pairs(
                pair_results, baseline, agent_id, list(agent_ids)
            )
            if not affected_ids:
                continue

            candidates.append(
                CandidateProbeResult(
                    agent_id=agent_id,
                    is_ego=is_ego,
                    perturbed_traj=perturbed_traj,
                    pair_delta=pair_delta,
                    affected_ids=affected_ids,
                    scores_before=scores_before,
                    scores_after=scores_after,
                )
            )
            candidate_pair_results.append(pair_results)

        best_idx = self._validator_fn(candidates)
        if best_idx is None:
            return None

        best = candidates[best_idx]
        return self._build_probe_result(
            scenario,
            baseline,
            best.agent_id,
            best.is_ego,
            best.perturbed_traj,
            candidate_pair_results[best_idx],
            best.affected_ids,
            best.scores_before,
            best.scores_after,
        )
