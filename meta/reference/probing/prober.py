"""Counterfactual probing orchestrator."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from safeair.scenario_characterization.common import ProbeValidity, ReturnCriterion, raw_to_agent_type
from safeair.scenario_characterization.features.interaction_features import InteractionFeatures
from safeair.scenario_characterization.probing.counterfactual_probes import constant_velocity_probe
from safeair.scenario_characterization.probing.criticality.interaction_criticality import find_criticality_timestamp
from safeair.scenario_characterization.scores.base_scorer import ScorerConfig
from safeair.scenario_characterization.scores.interaction_scorer import InteractionScorer
from safeair.schemas.critical_probe import CriticalityMetric, CriticalityResult, CriticalProbe, ProbeType
from safeair.schemas.scenario import Scenario
from safeair.schemas.scenario_features import CharacterizationParameters, InteractionPairFeatures, ScenarioFeatures
from safeair.utils.constants import SCALE_FACTOR_TO_M, SPEED_TO_MS, SpeedUnits, XYZScale
from safeair.utils.scenario_types import VALID_STATE_VALUE, AgentTrajectory, AgentType

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
        config: Probing configuration. See ``configs/probing/default.yaml`` for all parameters.
        characterization: Thresholds controlling feature computation. Defaults to the standard SafeAir values.
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

        self._interaction_extractor = InteractionFeatures(
            return_criterion=ReturnCriterion[config.return_criterion],
            characterization=characterization,
            n_jobs=config.n_jobs,
        )
        self._interaction_scorer = InteractionScorer(config=ScorerConfig())

    def probe_scenario(self, scenario: Scenario) -> CriticalProbe | None:
        """Run all ego/other counterfactual probes and return the most impactful result.

        Probes the ego agent against all others, and each other agent against the ego. Probing
        others vs others is not supported to avoid computational blowup.

        Args:
            scenario: The scenario to probe.

        Returns:
            The ``CriticalProbe`` with the highest pair-score delta, or ``None`` if no probe exceeds
            ``min_score_delta``.

        Raises:
            ValueError: If ``scenario.metadata.current_time_index`` is None.
        """
        if scenario.metadata.current_time_index is None:
            error_message = "scenario.metadata.current_time_index must be set to run counterfactual probing"
            raise ValueError(error_message)

        baseline = self._compute_baseline(scenario)
        return self._find_critical_probe(scenario, baseline)

    def _compute_baseline(self, scenario: Scenario) -> _ProbingBaseline:
        """Compute baseline interaction features and scores for the unperturbed scenario.

        Runs ``InteractionFeatures.compute()`` to warm the unit-conversion factors and extract all candidate pairs. Pair
        scores are cached by canonical key for O(1) delta lookup in the probe loop.

        Args:
            scenario: The unperturbed scenario.

        Returns:
            A ``_ProbingBaseline`` containing pair features, scene score, and pair score cache.
        """
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
        """Iterate over all agents, find the probe with the highest pair-score delta, and return its ``CriticalProbe``.

        Each pair is scored once during the selection loop. After the loop, the winning probe's full scene score and
        criticality timestamp are computed once.

        Args:
            scenario: The unperturbed scenario.
            baseline: Pre-computed baseline used to determine affected pairs and score deltas.

        Returns:
            The ``CriticalProbe`` with the highest pair-score delta, or ``None`` if no probe produces a
            qualifying pair-score increase.
        """
        ego_idx = scenario.metadata.ego_agent_index
        ego_id = int(scenario.agent_data.agent_ids[ego_idx])

        assert scenario.metadata.current_time_index is not None
        current_time_index = scenario.metadata.current_time_index
        frequency_hz = scenario.metadata.frequency_hz
        xyz_scale = scenario.metadata.xyz_scale
        speed_units = scenario.metadata.speed_units
        agent_ids = scenario.agent_data.agent_ids
        trajs = scenario.agent_data.agent_trajectories

        # Best probe tracking variables
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
            # If the agent type is not in the skip list, probe it and check if the perturbed trajectory is valid,
            # otherwise, skip it.
            if raw_to_agent_type(scenario.agent_data.agent_types[i]) in self._skip_agent_types:
                continue
            agent_traj = AgentTrajectory(trajs[i])
            perturbed_traj, valid = self._probe_fn(agent_traj, current_time_index, frequency_hz, xyz_scale, speed_units)
            if valid == ProbeValidity.INVALID:
                continue

            # If the probe is valid, get the affected pairs for this agent. If the agent is the ego, all pairs involving
            # the ego are affected; otherwise, only the pair involving the ego and this agent is affected.
            is_ego = i == ego_idx
            agent_id = int(agent_ids[i])

            affected_idx_pairs = self._get_affected_index_pairs(
                baseline.pair_features, baseline.id_to_idx, baseline.id_to_type, agent_id, ego_id, is_ego=is_ego
            )
            if not affected_idx_pairs:
                continue

            # Construct a perturbed scenario with the counterfactual trajectory substituted in, and run the feature
            # extractor on the affected pairs.
            trajs_with_perturbed_i = trajs.copy()
            trajs_with_perturbed_i[i] = perturbed_traj
            perturbed_scenario = scenario.model_copy(
                update={
                    "agent_data": scenario.agent_data.model_copy(update={"agent_trajectories": trajs_with_perturbed_i})
                }
            )
            affected_pair_features = self._interaction_extractor.compute_pairs(perturbed_scenario, affected_idx_pairs)

            # Score each affected pair. Pairs whose delta meets the threshold contribute to affected_ids and the
            # pair score dicts. Using the max delta (not sum) captures the most dangerous individual interaction.
            pair_delta, affected_ids, pair_scores_before, pair_scores_after = self._score_affected_pairs(
                affected_pair_features, baseline, agent_id
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

        # Winner found. Merge its fresh pair features back into the baseline and run the full scene scorer once.
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
            update={"agent_data": scenario.agent_data.model_copy(update={"agent_trajectories": trajs_with_best_probe})}
        )
        scene_score_after = self._score_interaction(probed_scenario, merged_pairs)

        scale_to_m = SCALE_FACTOR_TO_M[scenario.metadata.xyz_scale]
        speed_to_ms = SPEED_TO_MS[scenario.metadata.speed_units]
        id_to_crit = self._find_criticality_timestamp(
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

        # Optionally filter to the single most-critical affected agent.
        # Priority: TTC beats DRAC; among same metric, earliest timestamp wins.
        if self._single_affected_agent:
            best_aid, best_crit = min(
                id_to_crit.items(),
                key=lambda item: (0 if item[1].metric == CriticalityMetric.TTC else 1, item[1].timestamp),
            )
            id_to_crit = {best_aid: best_crit}

        affected_ids = list(id_to_crit.keys())

        # Rebuild pair score dicts to only include pairs whose other agent has criticality.
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
        """Return a canonical ``(min_id, max_id)`` key for a pair of agent IDs.

        Args:
            id_a: First agent ID.
            id_b: Second agent ID.

        Returns:
            Tuple with the smaller ID first, so symmetric pairs always map to the same key.
        """
        return (id_a, id_b) if id_a < id_b else (id_b, id_a)

    def _score_interaction(self, scenario: Scenario, pair_features: list[InteractionPairFeatures]) -> float:
        """Run the interaction scorer over the given pair features and return the interaction scene score.

        Args:
            scenario: The scenario whose agent IDs and metadata are used for scoring.
            pair_features: Pre-computed interaction features for the pairs to score.

        Returns:
            The interaction scene score.
        """
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
        """Score each affected pair and return the delta, affected IDs, and per-pair score dicts.

        Pairs involving skip-type agents are silently ignored. Only pairs whose score delta meets
        ``_min_score_delta`` contribute to ``affected_ids`` and the score dicts.

        Args:
            affected_pair_features: Freshly computed features for the perturbed pairs.
            baseline: Baseline containing the pair score cache and agent type map.
            agent_id: ID of the probed agent, used to identify the other agent in each pair.

        Returns:
            ``(pair_delta, affected_ids, pair_scores_before, pair_scores_after)`` where ``pair_delta`` is
            the maximum score delta across all non-skip pairs.
        """
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
        """Return agent index pairs that need recomputation for this probe.

        For an ego probe, all pairs where ``ego_id`` appears are recomputed (ego vs all others). For a non-ego probe,
        only the single ``(ego_id, probed_id)`` pair is recomputed. Pairs where either agent has a type in
        ``_skip_agent_types`` are excluded. Pairs that newly enter the distance threshold after perturbation are not
        included; this edge case has negligible impact.

        Args:
            baseline_pair_features: Pair features from the unperturbed scenario.
            id_to_idx: Mapping from agent ID to trajectory array index.
            id_to_type: Mapping from agent ID to ``AgentType``, used to skip undesired agent types.
            probed_id: ID of the agent being probed.
            ego_id: ID of the ego agent.
            is_ego: Whether the probed agent is the ego.

        Returns:
            List of ``(index_a, index_b)`` tuples identifying the pairs to recompute.
        """
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

    def _find_criticality_timestamp(
        self,
        probed_traj_arr: NDArray[np.float32],
        all_trajectories: NDArray[np.float32],
        candidate_agent_ids: list[int],
        agent_ids: NDArray[np.int32],
        scale_to_m: float,
        speed_to_ms: float,
        current_time_index: int,
    ) -> dict[int, CriticalityResult]:
        """Return criticality results for each candidate agent that passes the spatial gate.

        Only considers future timesteps (at or after ``current_time_index``). Agents for which TTC/DRAC
        produces no finite result (e.g. because they are too far from the probed agent) are excluded.
        Skipped agent types are filtered before this method is called.

        Args:
            probed_traj_arr: Counterfactual trajectory for the winning probed agent, shape ``(T, D)``.
            all_trajectories: All agent trajectories from the unperturbed scenario, shape ``(N, T, D)``.
            candidate_agent_ids: IDs of agents that passed the pair-score delta threshold and type filter.
            agent_ids: Agent ID array used to look up trajectory indices.
            scale_to_m: Conversion factor from scenario position units to metres.
            speed_to_ms: Conversion factor from scenario speed units to metres per second.
            current_time_index: First future timestep (inclusive); past frames are excluded from search.

        Returns:
            Mapping from agent ID to :class:`CriticalityResult` for each agent with a finite criticality.
        """
        traj_a = AgentTrajectory(probed_traj_arr)
        characterization = self._interaction_extractor.characterization
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
            if np.linalg.norm(probed_xy - candidate_xy) > characterization.agent_to_agent_max_distance:
                continue
            result = find_criticality_timestamp(
                traj_a, traj_b, characterization, scale_to_m, speed_to_ms, current_time_index
            )
            if result is not None:
                results[candidate_id] = result

        return results
