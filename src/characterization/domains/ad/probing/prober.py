"""Counterfactual probing orchestrator for the AD (autonomous driving) domain."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from characterization.domains.ad.features.interaction_features import InteractionFeatures
from characterization.domains.ad.probing.counterfactual_probes import constant_velocity_probe
from characterization.domains.ad.probing.criticality.interaction_criticality import find_criticality_timestamp
from characterization.domains.ad.scenario_types import AgentType
from characterization.domains.ad.schemas import InteractionPairFeatures, ScenarioFeatures
from characterization.domains.ad.schemas.scenario import Scenario
from characterization.domains.ad.scorer import InteractionScorer
from characterization.probing.common import ProbeValidity
from characterization.schemas.critical_probe import CriticalityMetric, CriticalityResult, CriticalProbe, ProbeType
from characterization.utils.common import ReturnCriterion
from characterization.utils.constants import SCALE_FACTOR_TO_M

ProbeFn = Callable[[NDArray[np.float32], int, float], tuple[NDArray[np.float32], ProbeValidity]]

_PROBE_REGISTRY: dict[ProbeType, ProbeFn] = {
    ProbeType.CONSTANT_VELOCITY: constant_velocity_probe,
}


class _ProbingBaseline(NamedTuple):
    """Pre-computed unperturbed baseline used by the probe selection loop.

    Attributes:
        pair_features: All candidate pair features for the unperturbed scenario. ``agent_id_a/b`` are agent indices.
        scene_score: Interaction scene score for the unperturbed scenario.
        pair_score_cache: Raw pair scores keyed by canonical ``(min_idx, max_idx)`` for O(1) delta lookup.
        idx_to_id: Maps agent array index to actual agent ID (for CriticalProbe output).
        idx_to_type: Maps agent array index to its ``AgentType``, used to filter out skipped-type pairs.
    """

    pair_features: list[InteractionPairFeatures]
    scene_score: float
    pair_score_cache: dict[tuple[int, int], float]
    idx_to_id: dict[int, int]
    idx_to_type: dict[int, AgentType]


class CounterfactualProber:
    """Runs counterfactual probing for ego-vs-others and others-vs-ego agent pairs in the AD domain.

    For each agent, the probe function replaces its future trajectory with a constant-velocity counterfactual.
    The probe with the greatest pair-score increase (if any exceed ``min_score_delta``) is selected, and its full
    interaction scene score is computed once. Agents whose type appears in ``config.skip_agent_types`` are never probed.

    Note:
        In the AD domain, ``agent_id_a/b`` fields in :class:`InteractionPairFeatures` are array **indices** into
        ``scenario.agent_data.agent_trajectories``, not actual agent IDs. Actual IDs are stored in
        ``scenario.agent_data.agent_ids``. The prober works with indices internally and converts to IDs in the output
        :class:`CriticalProbe`.

    Args:
        config: Probing configuration. See ``config/probing/cvm.yaml`` for all parameters.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the prober from a Hydra config."""
        self._probe_type = ProbeType[config.probe_type]
        self._probe_fn = _PROBE_REGISTRY[self._probe_type]
        self._min_score_delta = config.min_score_delta
        self._skip_agent_types = {AgentType[t] for t in config.skip_agent_types}
        self._single_affected_agent: bool = bool(config.get("single_affected_agent", True))

        self._interaction_extractor = InteractionFeatures(
            return_criterion=ReturnCriterion[config.return_criterion],
        )
        self._interaction_scorer = InteractionScorer()

    def probe_scenario(self, scenario: Scenario) -> CriticalProbe | None:
        """Run all ego/other counterfactual probes and return the most impactful result.

        Args:
            scenario: The scenario to probe.

        Returns:
            The :class:`CriticalProbe` with the highest pair-score delta, or ``None`` if no probe exceeds
            ``min_score_delta``.
        """
        baseline = self._compute_baseline(scenario)
        return self._find_critical_probe(scenario, baseline)

    def _compute_baseline(self, scenario: Scenario) -> _ProbingBaseline:
        scenario_features = self._interaction_extractor.compute(scenario)
        idx_to_type = dict(enumerate(scenario.agent_data.agent_types))
        pair_features = [
            p
            for p in (scenario_features.interaction_features or [])
            if idx_to_type.get(p.agent_id_a) not in self._skip_agent_types
            and idx_to_type.get(p.agent_id_b) not in self._skip_agent_types
        ]
        scene_score = self._score_interaction(scenario, pair_features)
        return _ProbingBaseline(
            pair_features=pair_features,
            scene_score=scene_score,
            pair_score_cache={
                self._pair_key(p.agent_id_a, p.agent_id_b): self._interaction_scorer.score_pair(p)
                for p in pair_features
            },
            idx_to_id=dict(enumerate(scenario.agent_data.agent_ids)),
            idx_to_type=idx_to_type,
        )

    def _find_critical_probe(self, scenario: Scenario, baseline: _ProbingBaseline) -> CriticalProbe | None:
        ego_idx = scenario.metadata.ego_agent_index
        ego_id = scenario.metadata.ego_vehicle_id

        current_time_index = scenario.metadata.current_time_index
        frequency_hz = scenario.metadata.frequency_hz
        trajs = scenario.agent_data.agent_trajectories
        scale_to_m = SCALE_FACTOR_TO_M[scenario.metadata.xyz_scale]

        best_pair_delta: float = 0.0
        best_i: int = -1
        best_agent_idx: int = -1
        best_is_ego: bool = False
        best_perturbed_traj: NDArray[np.float32] | None = None
        best_affected_pair_features: list[InteractionPairFeatures] = []
        best_affected_idxs: list[int] = []
        best_pair_scores_before: dict[str, float] = {}
        best_pair_scores_after: dict[str, float] = {}

        for i in range(scenario.agent_data.num_agents):
            if baseline.idx_to_type.get(i) in self._skip_agent_types:
                continue

            perturbed_traj, valid = self._probe_fn(trajs[i], current_time_index, frequency_hz)
            if valid == ProbeValidity.INVALID:
                continue

            is_ego = i == ego_idx

            # Determine which baseline pairs are affected by this probe.
            affected_idx_pairs = self._get_affected_index_pairs(
                baseline.pair_features,
                baseline.idx_to_type,
                probed_idx=i,
                ego_idx=ego_idx,
                is_ego=is_ego,
            )
            if not affected_idx_pairs:
                continue

            # Recompute interaction features for the full perturbed scenario, then filter to affected pairs.
            trajs_with_perturbed_i = trajs.copy()
            trajs_with_perturbed_i[i] = perturbed_traj
            perturbed_scenario = scenario.model_copy(
                update={
                    "agent_data": scenario.agent_data.model_copy(update={"agent_trajectories": trajs_with_perturbed_i}),
                },
            )
            all_perturbed_pairs = self._interaction_extractor.compute_interaction_features(perturbed_scenario) or []
            affected_idx_pair_set = {(a, b) for a, b in affected_idx_pairs} | {(b, a) for a, b in affected_idx_pairs}
            affected_pair_features = [
                p for p in all_perturbed_pairs if (p.agent_id_a, p.agent_id_b) in affected_idx_pair_set
            ]

            pair_delta, affected_idxs, pair_scores_before, pair_scores_after = self._score_affected_pairs(
                affected_pair_features,
                baseline,
                probed_idx=i,
            )

            if not affected_idxs:
                continue

            if pair_delta >= best_pair_delta:
                best_pair_delta = pair_delta
                best_i = i
                best_agent_idx = i
                best_is_ego = is_ego
                best_perturbed_traj = perturbed_traj
                best_affected_pair_features = affected_pair_features
                best_affected_idxs = affected_idxs
                best_pair_scores_before = pair_scores_before
                best_pair_scores_after = pair_scores_after

        if best_perturbed_traj is None:
            return None

        # Merge the winning probe's pairs into the baseline and run the full scene scorer once.
        affected_features_map: dict[tuple[int, int], InteractionPairFeatures] = {
            self._pair_key(pair.agent_id_a, pair.agent_id_b): pair for pair in best_affected_pair_features
        }
        merged_pairs = [
            affected_features_map.get(self._pair_key(pair.agent_id_a, pair.agent_id_b), pair)
            for pair in baseline.pair_features
        ]
        trajs_with_best_probe = trajs.copy()
        trajs_with_best_probe[best_i] = best_perturbed_traj
        probed_scenario = scenario.model_copy(
            update={"agent_data": scenario.agent_data.model_copy(update={"agent_trajectories": trajs_with_best_probe})},
        )
        scene_score_after = self._score_interaction(probed_scenario, merged_pairs)

        max_distance = scenario.metadata.agent_to_agent_max_distance
        max_decel = scenario.metadata.agent_max_deceleration
        idx_to_crit = self._find_criticality_timestamps(
            best_perturbed_traj,
            trajs,
            best_affected_idxs,
            scale_to_m,
            max_distance,
            max_decel,
            current_time_index,
        )

        if not idx_to_crit:
            return None

        if self._single_affected_agent:
            best_idx, best_crit = min(
                idx_to_crit.items(),
                key=lambda item: (0 if item[1].metric == CriticalityMetric.TTC else 1, item[1].timestamp),
            )
            idx_to_crit = {best_idx: best_crit}

        # Convert from indices to actual agent IDs for the CriticalProbe output.
        id_to_crit = {baseline.idx_to_id[idx]: crit for idx, crit in idx_to_crit.items()}
        affected_ids = list(id_to_crit.keys())
        best_agent_id = baseline.idx_to_id[best_agent_idx]
        ego_agent_id = ego_id

        # Rebuild pair score dicts, converting index-based keys to id-based keys.
        affected_pair_scores_before: dict[str, float] = {}
        affected_pair_scores_after: dict[str, float] = {}
        for key in best_pair_scores_before:
            a_str, b_str = key.split(":")
            other_idx = int(b_str) if int(a_str) == best_agent_idx else int(a_str)
            if other_idx in idx_to_crit:
                other_id = baseline.idx_to_id[other_idx]
                min_id = min(best_agent_id, other_id)
                max_id = max(best_agent_id, other_id)
                id_key = f"{min_id}:{max_id}"
                affected_pair_scores_before[id_key] = best_pair_scores_before[key]
                affected_pair_scores_after[id_key] = best_pair_scores_after[key]

        return CriticalProbe(
            probed_agent_id=best_agent_id,
            probed_agent_trajectory=best_perturbed_traj,
            is_ego_agent=best_is_ego,
            probe_type=self._probe_type,
            affected_agent_ids=affected_ids,
            criticality_results={str(aid): r for aid, r in id_to_crit.items()},
            score_before=baseline.scene_score,
            score_after=scene_score_after,
            affected_pair_scores_before=affected_pair_scores_before,
            affected_pair_scores_after=affected_pair_scores_after,
        )

    @staticmethod
    def _pair_key(id_a: int, id_b: int) -> tuple[int, int]:
        return (id_a, id_b) if id_a < id_b else (id_b, id_a)

    def _score_interaction(self, scenario: Scenario, pair_features: list[InteractionPairFeatures]) -> float:
        scenario_features = ScenarioFeatures(
            metadata=scenario.metadata,
            interaction_features=pair_features,
        )
        scores = self._interaction_scorer.compute(scenario, scenario_features)
        return scores.interaction_scene_score or 0.0

    def _score_affected_pairs(
        self,
        affected_pair_features: list[InteractionPairFeatures],
        baseline: _ProbingBaseline,
        probed_idx: int,
    ) -> tuple[float, list[int], dict[str, float], dict[str, float]]:
        pair_delta: float = 0.0
        affected_idxs: list[int] = []
        pair_scores_before: dict[str, float] = {}
        pair_scores_after: dict[str, float] = {}
        for pair in affected_pair_features:
            other_idx = pair.agent_id_b if pair.agent_id_a == probed_idx else pair.agent_id_a
            if baseline.idx_to_type.get(other_idx) in self._skip_agent_types:
                continue
            pair_key = self._pair_key(pair.agent_id_a, pair.agent_id_b)
            before = baseline.pair_score_cache.get(pair_key, 0.0)
            after = self._interaction_scorer.score_pair(pair)
            delta = after - before
            pair_delta = max(pair_delta, delta)
            if delta >= self._min_score_delta:
                key = f"{pair_key[0]}:{pair_key[1]}"
                pair_scores_before[key] = before
                pair_scores_after[key] = after
                if other_idx not in affected_idxs:
                    affected_idxs.append(other_idx)
        return pair_delta, affected_idxs, pair_scores_before, pair_scores_after

    def _get_affected_index_pairs(
        self,
        baseline_pair_features: list[InteractionPairFeatures],
        idx_to_type: dict[int, AgentType],
        probed_idx: int,
        ego_idx: int,
        *,
        is_ego: bool,
    ) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for pair in baseline_pair_features:
            if probed_idx not in (pair.agent_id_a, pair.agent_id_b):
                continue
            if not is_ego and ego_idx not in (pair.agent_id_a, pair.agent_id_b):
                continue
            if (
                idx_to_type.get(pair.agent_id_a) in self._skip_agent_types
                or idx_to_type.get(pair.agent_id_b) in self._skip_agent_types
            ):
                continue
            result.append((pair.agent_id_a, pair.agent_id_b))
        return result

    def _find_criticality_timestamps(
        self,
        probed_traj: NDArray[np.float32],
        all_trajectories: NDArray[np.float32],
        candidate_idxs: list[int],
        scale_to_m: float,
        max_distance: float,
        max_decel: float,
        current_time_index: int,
    ) -> dict[int, CriticalityResult]:
        results: dict[int, CriticalityResult] = {}
        probed_xy = probed_traj[current_time_index, :2] * scale_to_m
        for candidate_idx in candidate_idxs:
            if candidate_idx >= len(all_trajectories):
                continue
            traj_b = all_trajectories[candidate_idx]
            if traj_b[current_time_index, 9] < 1.0:
                continue
            candidate_xy = traj_b[current_time_index, :2] * scale_to_m
            if np.linalg.norm(probed_xy - candidate_xy) > max_distance:
                continue
            result = find_criticality_timestamp(
                probed_traj,
                traj_b,
                max_distance=max_distance,
                max_deceleration=max_decel,
                current_time_index=current_time_index,
            )
            if result is not None:
                results[candidate_idx] = result
        return results
