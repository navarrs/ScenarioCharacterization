"""Counterfactual probing orchestrator for the aviation domain."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from characterization.domains.aviation.features.interaction_features import InteractionFeatures
from characterization.domains.aviation.probing.counterfactual_probes import constant_velocity_probe
from characterization.domains.aviation.probing.criticality.interaction_criticality import find_criticality_timestamp
from characterization.domains.aviation.scenario_types import (
    VALID_STATE_VALUE,
    AgentTrajectory,
    AgentType,
    raw_to_agent_type,
)
from characterization.domains.aviation.schemas.scenario import Scenario
from characterization.domains.aviation.schemas.scenario_features import (
    CharacterizationParameters,
    InteractionPairFeatures,
    ScenarioFeatures,
)
from characterization.domains.aviation.scorer import InteractionScorer, ScorerConfig
from characterization.probing.common import ProbeValidity
from characterization.schemas.critical_probe import CriticalityMetric, CriticalityResult, CriticalProbe, ProbeType
from characterization.utils.common import ReturnCriterion, SpeedUnits, XYZScale
from characterization.utils.constants import SCALE_FACTOR_TO_M, SPEED_TO_MS

ProbeFn = Callable[[AgentTrajectory, int, float, XYZScale, SpeedUnits], tuple[NDArray[np.float32], ProbeValidity]]

_PROBE_REGISTRY: dict[ProbeType, ProbeFn] = {
    ProbeType.CONSTANT_VELOCITY: constant_velocity_probe,
}
_DEFAULT_CHARACTERIZATION = CharacterizationParameters()


class _ProbingBaseline(NamedTuple):
    """Pre-computed unperturbed baseline used by the probe selection loop.

    Attributes:
        pair_features: All candidate pair features for the unperturbed scenario.
        scene_score: Interaction scene score for the unperturbed scenario.
        pair_score_cache: Raw pair scores keyed by canonical ``(min_id, max_id)`` for O(1) delta lookup.
        id_to_idx: Maps agent ID to its index in ``scenario.agent_data.agent_ids``.
        id_to_type: Maps agent ID to its ``AgentType``, used to filter out skipped-type pairs.
    """

    pair_features: list[InteractionPairFeatures]
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
        config: Probing configuration. See ``config/probing/cvm.yaml`` for all parameters.
        characterization: Thresholds controlling feature computation. Defaults to the standard aviation values.
    """

    def __init__(
        self,
        config: DictConfig,
        *,
        characterization: CharacterizationParameters = _DEFAULT_CHARACTERIZATION,
    ) -> None:
        """Initialize the prober from a Hydra config."""
        self._probe_type = ProbeType[config.probe_type]
        self._probe_fn = _PROBE_REGISTRY[self._probe_type]
        self._min_score_delta = config.min_score_delta
        self._skip_agent_types = {AgentType[t] for t in config.skip_agent_types}
        self._single_affected_agent: bool = bool(config.get("single_affected_agent", True))
        self._characterization = characterization

        self._interaction_extractor = InteractionFeatures(
            return_criterion=ReturnCriterion[config.return_criterion],
            characterization=characterization,
            n_jobs=config.n_jobs,
        )
        self._interaction_scorer = InteractionScorer(config=ScorerConfig())

    def probe_scenario(self, scenario: Scenario) -> CriticalProbe | None:
        """Run all ego/other counterfactual probes and return the most impactful result.

        Probes the ego agent against all others, and each other agent against the ego. Probing others vs others is not
        supported to avoid computational blowup.

        Args:
            scenario: The scenario to probe.

        Returns:
            The :class:`CriticalProbe` with the highest pair-score delta, or ``None`` if no probe exceeds
            ``min_score_delta``.

        Raises:
            ValueError: If ``scenario.metadata.current_time_index`` is ``None``.
        """
        if scenario.metadata.current_time_index is None:
            msg = "scenario.metadata.current_time_index must be set to run counterfactual probing"
            raise ValueError(msg)

        baseline = self._compute_baseline(scenario)
        return self._find_critical_probe(scenario, baseline)

    def _compute_baseline(self, scenario: Scenario) -> _ProbingBaseline:
        scenario_features = self._interaction_extractor.compute(scenario)
        id_to_type = {
            int(aid): raw_to_agent_type(atype)
            for aid, atype in zip(scenario.agent_data.agent_ids, scenario.agent_data.agent_types, strict=True)
        }
        pair_features = [
            p
            for p in scenario_features.interaction_features
            if id_to_type.get(p.agent_id_a) not in self._skip_agent_types
            and id_to_type.get(p.agent_id_b) not in self._skip_agent_types
        ]
        scene_score = self._score_interaction(scenario, pair_features)
        return _ProbingBaseline(
            pair_features=pair_features,
            scene_score=scene_score,
            pair_score_cache={
                self._pair_key(p.agent_id_a, p.agent_id_b): self._interaction_scorer.score_pair(p)
                for p in pair_features
            },
            id_to_idx={int(aid): idx for idx, aid in enumerate(scenario.agent_data.agent_ids)},
            id_to_type=id_to_type,
        )

    def _find_critical_probe(self, scenario: Scenario, baseline: _ProbingBaseline) -> CriticalProbe | None:
        ego_idx = scenario.metadata.ego_agent_index
        ego_id = int(scenario.agent_data.agent_ids[ego_idx])

        assert scenario.metadata.current_time_index is not None
        current_time_index = scenario.metadata.current_time_index
        frequency_hz = scenario.metadata.frequency_hz
        xyz_scale = scenario.metadata.xyz_scale
        speed_units = scenario.metadata.speed_units
        agent_ids = scenario.agent_data.agent_ids
        trajs = scenario.agent_data.agent_trajectories

        best_pair_delta: float = 0.0
        best_idx: int = -1
        best_agent_id: int = -1
        best_is_ego: bool = False
        best_perturbed_traj: NDArray[np.float32] | None = None
        best_affected_pair_features: list[InteractionPairFeatures] = []
        best_affected_ids: list[int] = []
        best_pair_scores_before: dict[str, float] = {}
        best_pair_scores_after: dict[str, float] = {}

        for i in range(scenario.agent_data.num_agents):
            if raw_to_agent_type(scenario.agent_data.agent_types[i]) in self._skip_agent_types:
                continue
            agent_traj = AgentTrajectory(trajs[i])
            perturbed_traj, valid = self._probe_fn(agent_traj, current_time_index, frequency_hz, xyz_scale, speed_units)
            if valid == ProbeValidity.INVALID:
                continue

            is_ego = i == ego_idx
            agent_id = int(agent_ids[i])

            affected_idx_pairs = self._get_affected_index_pairs(
                baseline.pair_features,
                baseline.id_to_idx,
                baseline.id_to_type,
                agent_id,
                ego_id,
                is_ego=is_ego,
            )
            if not affected_idx_pairs:
                continue

            trajs_with_perturbed_i = trajs.copy()
            trajs_with_perturbed_i[i] = perturbed_traj
            perturbed_scenario = scenario.model_copy(
                update={
                    "agent_data": scenario.agent_data.model_copy(update={"agent_trajectories": trajs_with_perturbed_i}),
                },
            )
            affected_pair_features = self._interaction_extractor.compute_pairs(perturbed_scenario, affected_idx_pairs)

            pair_delta, affected_ids, pair_scores_before, pair_scores_after = self._score_affected_pairs(
                affected_pair_features,
                baseline,
                agent_id,
            )

            if not affected_ids:
                continue

            if pair_delta >= best_pair_delta:
                best_pair_delta = pair_delta
                best_idx = i
                best_agent_id = agent_id
                best_is_ego = is_ego
                best_perturbed_traj = perturbed_traj
                best_affected_pair_features = affected_pair_features
                best_affected_ids = affected_ids
                best_pair_scores_before = pair_scores_before
                best_pair_scores_after = pair_scores_after

        if best_perturbed_traj is None:
            return None

        affected_features_map: dict[tuple[int, int], InteractionPairFeatures] = {
            self._pair_key(pair.agent_id_a, pair.agent_id_b): pair for pair in best_affected_pair_features
        }
        merged_pairs = [
            affected_features_map.get(self._pair_key(pair.agent_id_a, pair.agent_id_b), pair)
            for pair in baseline.pair_features
        ]
        trajs_with_best_probe = trajs.copy()
        trajs_with_best_probe[best_idx] = best_perturbed_traj
        probed_scenario = scenario.model_copy(
            update={"agent_data": scenario.agent_data.model_copy(update={"agent_trajectories": trajs_with_best_probe})},
        )
        scene_score_after = self._score_interaction(probed_scenario, merged_pairs)

        scale_to_m = SCALE_FACTOR_TO_M[scenario.metadata.xyz_scale]
        speed_to_ms = SPEED_TO_MS[scenario.metadata.speed_units]
        id_to_crit = self._find_criticality_timestamps(
            best_perturbed_traj,
            trajs,
            best_affected_ids,
            agent_ids,
            scale_to_m,
            speed_to_ms,
            current_time_index,
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

        affected_pair_scores_before: dict[str, float] = {}
        affected_pair_scores_after: dict[str, float] = {}
        for key in best_pair_scores_before:
            a_str, b_str = key.split(":")
            other_id = int(b_str) if int(a_str) == best_agent_id else int(a_str)
            if other_id in id_to_crit:
                affected_pair_scores_before[key] = best_pair_scores_before[key]
                affected_pair_scores_after[key] = best_pair_scores_after[key]

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
            scenario_id=scenario.metadata.scenario_id,
            individual_features=[],
            interaction_features=pair_features,
        )
        scores = self._interaction_scorer.compute(scenario, scenario_features)
        return scores.interaction_scene_score or 0.0

    def _score_affected_pairs(
        self,
        affected_pair_features: list[InteractionPairFeatures],
        baseline: _ProbingBaseline,
        agent_id: int,
    ) -> tuple[float, list[int], dict[str, float], dict[str, float]]:
        pair_delta: float = 0.0
        affected_ids: list[int] = []
        pair_scores_before: dict[str, float] = {}
        pair_scores_after: dict[str, float] = {}
        for pair in affected_pair_features:
            other_id = pair.agent_id_b if pair.agent_id_a == agent_id else pair.agent_id_a
            if baseline.id_to_type.get(other_id) in self._skip_agent_types:
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
                if other_id not in affected_ids:
                    affected_ids.append(other_id)
        return pair_delta, affected_ids, pair_scores_before, pair_scores_after

    def _get_affected_index_pairs(
        self,
        baseline_pair_features: list[InteractionPairFeatures],
        id_to_idx: dict[int, int],
        id_to_type: dict[int, AgentType],
        probed_id: int,
        ego_id: int,
        *,
        is_ego: bool,
    ) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for pair in baseline_pair_features:
            if probed_id not in (pair.agent_id_a, pair.agent_id_b):
                continue
            if not is_ego and ego_id not in (pair.agent_id_a, pair.agent_id_b):
                continue
            if (
                id_to_type.get(pair.agent_id_a) in self._skip_agent_types
                or id_to_type.get(pair.agent_id_b) in self._skip_agent_types
            ):
                continue
            idx_a = id_to_idx.get(pair.agent_id_a)
            idx_b = id_to_idx.get(pair.agent_id_b)
            if idx_a is not None and idx_b is not None:
                result.append((idx_a, idx_b))
        return result

    def _find_criticality_timestamps(
        self,
        probed_traj_arr: NDArray[np.float32],
        all_trajectories: NDArray[np.float32],
        candidate_agent_ids: list[int],
        agent_ids: NDArray[np.int32],
        scale_to_m: float,
        speed_to_ms: float,
        current_time_index: int,
    ) -> dict[int, CriticalityResult]:
        traj_a = AgentTrajectory(probed_traj_arr)
        results: dict[int, CriticalityResult] = {}

        probed_xy = traj_a.xyz_position[current_time_index, :2] * scale_to_m
        for candidate_id in candidate_agent_ids:
            matches = np.where(agent_ids == candidate_id)[0]
            if len(matches) == 0:
                continue
            traj_b = AgentTrajectory(all_trajectories[matches[0]])
            if traj_b.valid[current_time_index] < VALID_STATE_VALUE:
                continue
            candidate_xy = traj_b.xyz_position[current_time_index, :2] * scale_to_m
            if np.linalg.norm(probed_xy - candidate_xy) > self._characterization.agent_to_agent_max_distance:
                continue
            result = find_criticality_timestamp(
                traj_a,
                traj_b,
                horizontal_separation_breach=self._characterization.horizontal_separation_breach,
                vertical_separation_breach=self._characterization.vertical_separation_breach,
                agent_max_deceleration=self._characterization.agent_max_deceleration,
                scale_to_m=scale_to_m,
                speed_to_ms=speed_to_ms,
                current_time_index=current_time_index,
            )
            if result is not None:
                results[candidate_id] = result

        return results
