"""Counterfactual probing orchestrator for autonomous driving scenarios."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from characterization.features.interaction_features import InteractionFeatures, InteractionStatus
from characterization.probing.base_prober import BaseProber
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
from characterization.schemas.scenario import Scenario, ScenarioMetadata
from characterization.schemas.scenario_features import Interaction, ScenarioFeatures
from characterization.scorer.interaction_scorer import InteractionScorer
from characterization.utils.common import AgentTrajectoryMasker
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
    """Return a canonical, order-independent key for an agent pair.

    Pairs are stored with the smaller ID first so that ``(a, b)`` and ``(b, a)`` map to the same cache entry regardless
    of the order in which the pair was encountered.

    Args:
        id_a: First agent ID.
        id_b: Second agent ID.

    Returns:
        ``(min_id, max_id)`` tuple suitable for use as a dict key.
    """
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
        weights: Per-agent weight array (index-aligned with ``scenario.agent_data.agent_ids``) computed by the
            scorer for the unperturbed scenario. Reused for pair-delta ranking; updated per-probe in
            :meth:`CounterfactualProber._build_probe_result` for the final ``score_after``.
    """

    interaction: Interaction | None
    scene_score: float
    pair_score_cache: dict[tuple[int, int], float]
    id_to_idx: dict[int, int]
    id_to_type: dict[int, AgentType]
    weights: NDArray[np.float32]


class CounterfactualProber(BaseProber):
    """Runs counterfactual probing for ego-vs-others and others-vs-ego agent pairs.

    For each agent, the probe function replaces its future trajectory with a counterfactual. The probe with the greatest
    pair-score increase (if any exceed ``min_score_delta``) is selected, and its full interaction scene score is
    computed once. Agents whose type appears in ``config.skip_agent_types`` are never probed.

    Args:
        config: Probing configuration. See ``config/probing/default.yaml`` for all parameters.
    """

    @property
    def name(self) -> str:
        """Returns the name of this prober.

        Returns:
            str: Human-readable prober name.
        """
        return "counterfactual prober"

    def __init__(self, config: DictConfig) -> None:
        """Initialize the prober from a Hydra config.

        Looks up the probe function and validator from the respective registries using the enum keys specified in
        ``config``. An :class:`~characterization.scorer.interaction_scorer.InteractionScorer` is instantiated as the
        single source of truth for the score function, detection thresholds, and feature weights — ensuring that
        pair-level scores and the aggregated scene score are computed identically to the scores processor pipeline.

        Args:
            config: Hydra ``DictConfig`` with the following keys:
                - ``probe_type`` (:class:`ProbeType` name): Counterfactual perturbation strategy.
                - ``validator_type`` (:class:`ValidatorType` name): Candidate-selection strategy.
                - ``min_score_delta`` (float): Minimum pair-score increase for a probe to be considered critical.
                - ``skip_agent_types`` (list[str], optional): Agent type names to exclude from probing and scoring.
                - ``single_affected_agent`` (bool, optional): If ``True`` (default), only the single
                    most critical affected agent is retained in the output.
                - ``return_criterion`` (str, optional): Passed to :class:`InteractionFeatures`; controls which
                    interactions are returned (default ``"critical"``).
                - ``inv_stability_cap`` (float, optional): Cap on inverse-feature values passed to the interaction
                    extractor (default ``10.0``).
                - ``score_weighting_method`` (str, optional): Agent weighting strategy forwarded to the scorer
                    (default ``"uniform"``). ``"distance_to_ego_agent"`` is also supported; baseline weights are
                    computed once from the original trajectories and updated efficiently (O(T) per non-ego probe,
                    O(N·T) for ego probes) when building the final result.
                - ``score_clip`` (dict, optional): Score clipping bounds forwarded to the scorer
                    (default ``{"min": 0.0, "max": 200.0}``).
                - ``detections`` (dict, optional): Override detection thresholds.
                - ``weights`` (dict, optional): Override feature weights.
        """
        self._probe_type = ProbeType[config.probe_type]
        self._probe_fn = _PROBE_REGISTRY[self._probe_type]

        self._validator_type = ValidatorType[config.validator_type]
        self._validator_fn = _VALIDATOR_REGISTRY[self._validator_type]

        self._min_score_delta: float = float(config.min_score_delta)
        self._skip_agent_types: set[AgentType] = {AgentType[t] for t in config.get("skip_agent_types", [])}
        self._single_affected_agent: bool = bool(config.get("single_affected_agent", True))

        # The extractor is configured for lightweight, single-threaded scoring inside the probe loop.
        # ``compute_agent_to_agent_closest_dists`` is disabled because the probe only needs scalar
        # interaction features (TTC, THW, DRAC, etc.), not spatial distance arrays.
        extractor_cfg = OmegaConf.create(
            {
                "return_criterion": config.get("return_criterion", "critical"),
                "feature_type": "continuous",
                "inv_stability_cap": config.get("inv_stability_cap", 10.0),
                "compute_agent_to_agent_closest_dists": False,
            }
        )
        self._extractor = InteractionFeatures(extractor_cfg)

        # Single source of truth for scoring: the InteractionScorer owns the score function, detection thresholds,
        # and feature weights. This guarantees that pair-level scores and scene-level aggregation in the prober are
        # computed identically to the scores processor, resolving the prior score-range discrepancy.
        scorer_cfg = OmegaConf.create(
            {
                "interaction_score_function": "simple",
                "score_weighting_method": config.get("score_weighting_method", "uniform"),
                "score_clip": config.get("score_clip", {"min": 0.0, "max": 200.0}),
                "detections": OmegaConf.to_container(config.detections) if "detections" in config else None,
                "weights": OmegaConf.to_container(config.weights) if "weights" in config else None,
            }
        )
        self._scorer = InteractionScorer(scorer_cfg)

    def compute(self, scenario: Scenario) -> CriticalProbe | None:
        """Run all counterfactual probes and return the most impactful result.

        Probes the ego agent against all others and each non-ego agent against the ego. Non-ego vs non-ego pairs are not
        probed to avoid combinatorial blowup.

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

    @staticmethod
    def _interaction_to_feature_dict(interaction: Interaction, n: int) -> dict[str, float]:
        """Extract interaction features for a single pair index into a plain dict.

        Reads the ``n``-th element of each per-pair feature array in ``interaction``. Any ``None`` array or non-finite
        value (NaN / ±Inf) is replaced with 0.0 so that downstream scoring always receives clean scalars.

        Args:
            interaction: Full :class:`~characterization.schemas.scenario_features.Interaction` object.
            n: Index into the per-pair arrays (i.e. the n-th entry in ``interaction.interaction_agent_indices``).

        Returns:
            Dict with keys ``"collision"``, ``"inv_mttcp"``, ``"inv_thw"``, ``"inv_ttc"``,
            and ``"drac"``.
        """

        def _safe(arr: NDArray[np.float32] | None, idx: int) -> float:
            """Return ``arr[idx]`` as a float, or 0.0 if the array is ``None`` or the value is non-finite."""
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

    def _compute_baseline(self, scenario: Scenario) -> _ProbingBaseline:
        """Compute the unperturbed interaction and score cache for a scenario.

        Runs the :class:`~characterization.features.interaction_features.InteractionFeatures` extractor once on the
        original scenario, then scores every valid pair with :meth:`~InteractionScorer.score_pair`. Pairs involving
        skip-typed agents or with a non-OK status are excluded from the cache so they don't inflate scene scores or act
        as probe targets.

        Per-agent weights are computed via the scorer (respecting the configured ``score_weighting_method``). For
        :attr:`~characterization.scorer.base_scorer.ScoreWeightingMethod.DISTANCE_TO_EGO_AGENT`, distances are derived
        from the original trajectories using the ``BaseScorer`` fallback (no separate distance matrix is pre-computed).
        Weights are stored in the returned baseline so they can be reused for candidate ranking and updated efficiently
        for the final ``score_after`` computation.

        Args:
            scenario: The original (unperturbed) scenario.

        Returns:
            :class:`_ProbingBaseline` with the full
            :class:`~characterization.schemas.scenario_features.Interaction`, the scene score,
            per-agent weights, and lookup dicts for O(1) access during the probe loop.
        """
        interaction = self._extractor.compute_interaction_features(scenario, max_workers=1)
        agent_ids = scenario.agent_data.agent_ids
        agent_types = scenario.agent_data.agent_types

        id_to_idx = {int(aid): idx for idx, aid in enumerate(agent_ids)}
        id_to_type = {int(aid): atype for aid, atype in zip(agent_ids, agent_types, strict=True)}

        # Wrap the Interaction in a minimal ScenarioFeatures so the scorer can compute per-agent weights.
        # agent_to_agent_closest_dists is left None; BaseScorer._get_agent_to_agent_closest_dists falls back to
        # computing distances from trajectories automatically.
        scenario_features = ScenarioFeatures(metadata=scenario.metadata, interaction_features=interaction)
        weights = self._scorer.get_weights(scenario, scenario_features)

        pair_score_cache: dict[tuple[int, int], float] = {}
        if (
            interaction is not None
            and interaction.interaction_agent_indices is not None
            and interaction.interaction_status is not None
        ):
            for n, (i, j) in enumerate(interaction.interaction_agent_indices):
                status = interaction.interaction_status[n]
                # Accept fully computed pairs and those with only a partial heading issue; reject pairs that failed
                # entirely (e.g. insufficient valid frames).
                if status not in (InteractionStatus.COMPUTED_OK, InteractionStatus.PARTIAL_INVALID_HEADING):
                    continue
                id_a, id_b = int(agent_ids[i]), int(agent_ids[j])
                if id_to_type.get(id_a) in self._skip_agent_types or id_to_type.get(id_b) in self._skip_agent_types:
                    continue
                feature_dict = self._interaction_to_feature_dict(interaction, n)
                pair_score_cache[_pair_key(id_a, id_b)] = self._scorer.score_pair(feature_dict)

        return _ProbingBaseline(
            interaction=interaction,
            scene_score=self._scorer.compute_scene_score_from_pair_cache(
                scenario, pair_score_cache, weights, id_to_idx
            ),
            pair_score_cache=pair_score_cache,
            id_to_idx=id_to_idx,
            id_to_type=id_to_type,
            weights=weights,
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

        For an ego probe, all pairs where the ego index appears are recomputed. For a non-ego probe, only the single
        ``(ego_idx, probed_idx)`` pair is recomputed. This scoping avoids unnecessary pairwise recomputation: a non-ego
        trajectory change can only affect interactions it participates in, and only the ego-involving subset is
        evaluated to prevent combinatorial blowup.

        Args:
            interaction: The unperturbed baseline interaction object.
            id_to_type: Maps agent ID to :class:`~characterization.utils.scenario_types.AgentType`.
            probed_idx: Index (in ``agent_ids``) of the agent being perturbed.
            ego_idx: Index (in ``agent_ids``) of the ego vehicle.
            agent_ids: Ordered list of all agent IDs.
            is_ego: Whether ``probed_idx`` is the ego vehicle.

        Returns:
            List of ``(i, j)`` index pairs from ``interaction.interaction_agent_indices`` that must
            be recomputed with the perturbed trajectory.
        """
        result: list[tuple[int, int]] = []
        for i, j in interaction.interaction_agent_indices or []:
            if probed_idx not in (i, j):
                continue
            # For non-ego probes, only recompute the pair with the ego to avoid O(N²) evaluations.
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
        """Score each recomputed pair and return the worst-case delta with supporting metadata.

        For each pair result, looks up the baseline score and computes ``after - before``. The returned ``pair_delta``
        is the *maximum* delta across all pairs (worst-case increase), not a sum — this prevents a single probe from
        appearing critical merely because it affects many low-delta pairs. Only pairs whose delta meets
        ``min_score_delta`` contribute to ``affected_ids`` and the before/after score dicts.

        Args:
            pair_results: Sequence of ``(index_pair, status, feature_dict)`` triples as returned by
                :meth:`~characterization.features.interaction_features.InteractionFeatures.compute_pairs`.
            baseline: Pre-computed unperturbed baseline used for score delta calculation.
            probed_agent_id: ID of the agent whose trajectory was replaced; used to identify the *other* agent in each
                pair when building ``affected_ids``.
            agent_ids: Ordered list of all agent IDs in the scenario.

        Returns:
            A 4-tuple ``(pair_delta, affected_ids, scores_before, scores_after)`` where:

            - ``pair_delta``: Maximum pair-score increase across all valid pairs.
            - ``affected_ids``: Unique IDs of non-probed agents in pairs that exceeded
              ``min_score_delta``.
            - ``scores_before``: Baseline scores for significant pairs, keyed as ``"id_a:id_b"``.
            - ``scores_after``: Post-probe scores for significant pairs, keyed as ``"id_a:id_b"``.
        """
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
            # Identify the partner in this pair (the agent that was *not* perturbed).
            other_id = id_b if id_a == probed_agent_id else id_a
            if baseline.id_to_type.get(other_id) in self._skip_agent_types:
                continue

            canon = _pair_key(id_a, id_b)
            # A pair absent from the baseline cache was previously below the detection threshold; treat its baseline
            # score as 0.0.
            before = baseline.pair_score_cache.get(canon, 0.0)
            after = self._scorer.score_pair(feature_dict)
            delta = after - before
            # Track the single largest delta — this is the signal used by the validator.
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
        """Find the frame of peak criticality between the probed agent and each candidate.

        For each ``candidate_id``, locates the first matching trajectory in ``all_trajs`` and delegates to
        :func:`~characterization.probing.validators.find_criticality_timestamp`. Candidates for which no valid
        criticality frame is found (e.g. agents too far apart or with insufficient valid frames) are silently omitted
        from the result.

        Args:
            probed_traj: Counterfactual trajectory for the probed agent, shape ``(T, F)``.
            all_trajs: All agent trajectories in the scenario, shape ``(N, T, F)``.
            candidate_ids: IDs of agents to evaluate against the probed trajectory.
            agent_ids: Ordered list of all agent IDs (index-aligned with ``all_trajs``).
            metadata: Scenario metadata supplying distance/heading thresholds and the deceleration cap.
            current_time_index: First future timestep (inclusive); past frames are excluded.

        Returns:
            Dict mapping each agent ID to its
            :class:`~characterization.schemas.critical_probe.CriticalityResult`. Only agents with a
            finite criticality frame are included.
        """
        results: dict[int, CriticalityResult] = {}
        for cid in candidate_ids:
            # Locate the trajectory array for this candidate by matching against the full agent list.
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
        """Build the final CriticalProbe from the winner candidate.

        Computes the post-probe scene score, finds the criticality timestamp for each affected agent, and — when
        ``single_affected_agent`` is ``True`` — retains only the single most critical agent (TTC-critical preferred over
        DRAC-critical; earlier timestamp preferred within the same metric). The pair score dicts are then filtered to
        only include pairs involving the retained agents.

        Args:
            scenario: Original scenario (provides agent IDs, trajectories, and metadata).
            baseline: Unperturbed baseline used to compute the scene-score delta.
            best_agent_id: ID of the winning (most critical) probed agent.
            best_is_ego: Whether the winning probed agent is the ego vehicle.
            best_perturbed_traj: Counterfactual trajectory for the winning probed agent.
            best_affected_pair_results: Raw pair-extraction results for all affected pairs of the
                winning probe (output of :meth:`InteractionFeatures.compute_pairs`).
            best_affected_ids: IDs of agents whose pair score exceeded ``min_score_delta``.
            best_pair_scores_before: Pre-probe pair scores, keyed as ``"id_a:id_b"``.
            best_pair_scores_after: Post-probe pair scores, keyed as ``"id_a:id_b"``.

        Returns:
            Populated :class:`~characterization.schemas.critical_probe.CriticalProbe`, or ``None``
            if no criticality timestamp could be found for any affected agent.
        """
        agent_ids = scenario.agent_data.agent_ids
        trajs = scenario.agent_data.agent_trajectories
        current_time_index = scenario.metadata.current_time_index

        # Build a per-pair score dict from the winner's raw pair results so we can merge it with the unchanged baseline
        # pairs to compute the full post-probe scene score.
        winner_pair_scores: dict[tuple[int, int], float] = {}
        for (vi, vj), status, fd in best_affected_pair_results:
            if status not in (InteractionStatus.COMPUTED_OK, InteractionStatus.PARTIAL_INVALID_HEADING):
                continue
            if fd is None:
                continue
            id_a = int(agent_ids[vi])
            id_b = int(agent_ids[vj])
            winner_pair_scores[_pair_key(id_a, id_b)] = self._scorer.score_pair(fd)

        # Recompute only the distances that changed due to the winning probe, then derive updated weights.
        # For non-ego probes this is O(T); for ego probes O(N*T) — far cheaper than a full N*N matrix.
        probed_idx = baseline.id_to_idx[best_agent_id]
        perturbed_weights = self._scorer.compute_weights_for_probe(
            scenario, baseline.weights, probed_idx, best_perturbed_traj
        )

        # Merge winner scores into the baseline cache (winner_pair_scores wins on key collisions), then recompute the
        # scene score over all pairs using the updated weights so that score_after reflects both the feature change
        # and the weight change caused by the trajectory perturbation.
        scene_score_after = self._scorer.compute_scene_score_from_pair_cache(
            scenario,
            {**baseline.pair_score_cache, **winner_pair_scores},
            perturbed_weights,
            baseline.id_to_idx,
        )

        id_to_crit = self._find_criticality_timestamps(
            best_perturbed_traj, trajs, best_affected_ids, list(agent_ids), scenario.metadata, current_time_index
        )
        if not id_to_crit:
            return None

        if self._single_affected_agent:
            # Keep only the most safety-critical affected agent. TTC-based criticality is preferred over DRAC-based,
            # within the same metric, an earlier timestamp is considered more urgent.
            best_aid, best_crit = min(
                id_to_crit.items(),
                key=lambda item: (0 if item[1].metric == CriticalityMetric.TTC else 1, item[1].timestamp),
            )
            id_to_crit = {best_aid: best_crit}

        affected_ids = list(id_to_crit.keys())
        # Filter the pair score dicts to only include pairs involving the retained affected agents, so the output
        # CriticalProbe does not reference agents that were pruned above.
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
        """Iterate over all probe candidates, select the best, and build the final result.

        For each agent (excluding skip-typed ones):

        1. Applies the probe function to generate a counterfactual trajectory.
        2. Determines which baseline pairs need recomputation (limited to ego-involving pairs for non-ego probes to
            avoid O(N²) evaluations).
        3. Builds a shallow copy of the scenario with only the probed agent's trajectory replaced, then re-runs the
            feature extractor on the affected pairs only.
        4. Scores the recomputed pairs and collects candidates whose score delta exceeds ``min_score_delta``.

        After the loop, the validator selects the best candidate and :meth:`_build_probe_result` assembles the final
            :class:`~characterization.schemas.critical_probe.CriticalProbe`.

        Args:
            scenario: The scenario to probe.
            baseline: Pre-computed unperturbed baseline.

        Returns:
            :class:`~characterization.schemas.critical_probe.CriticalProbe` for the best probe,
            or ``None`` if no candidate exceeds ``min_score_delta``.
        """
        assert baseline.interaction is not None

        ego_idx = scenario.metadata.ego_vehicle_index
        current_time_index = scenario.metadata.current_time_index
        frequency_hz = scenario.metadata.frequency_hz
        agent_ids = scenario.agent_data.agent_ids
        trajs = scenario.agent_data.agent_trajectories

        candidates: list[CandidateProbeResult] = []
        # Parallel list to ``candidates``; stores the raw pair-extraction results for each candidate so we can pass them
        # directly to ``_build_probe_result`` without re-running the extractor.
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

            # Build a lightweight perturbed scenario: shallow-copy the trajectory array and swap in only the one
            # trajectory. model_copy avoids revalidating all other fields.
            perturbed_trajs = trajs.copy()
            perturbed_trajs[i] = perturbed_traj
            perturbed_scenario = scenario.model_copy(
                update={"agent_data": scenario.agent_data.model_copy(update={"agent_trajectories": perturbed_trajs})}
            )
            pair_results = self._extractor.compute_pairs(perturbed_scenario, affected_idx_pairs)

            pair_delta, affected_ids, scores_before, scores_after = self._score_affected_pairs(
                pair_results, baseline, agent_id, list(agent_ids)
            )
            # Skip this probe if no pair exceeded min_score_delta — nothing critical to report.
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
