import json
from pathlib import Path
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from characterization.features.interaction_features import InteractionStatus
from characterization.schemas import Scenario, ScenarioFeatures, ScenarioScores, Score
from characterization.scorer.base_scorer import BaseScorer, ScoreWeightingMethod
from characterization.utils.common import EPSILON
from characterization.utils.io_utils import get_logger
from characterization.utils.scenario_types import AgentType

from .score_functions import INTERACTION_SCORE_FUNCTIONS

logger = get_logger(__name__)


class InteractionScorer(BaseScorer):
    """Class to compute interaction scores for agent pairs and a scene-level score from scenario features."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the InteractionScorer with a configuration.

        Args:
            config (DictConfig): Configuration for the scorer.
        """
        super().__init__(config)

        interaction_score_function = self.config.get("interaction_score_function")
        if not interaction_score_function:
            warning_message = (
                "No interaction_score_function specified. Defaulting to 'simple'."
                f"If this is not intended, specify one of the supported functions: {INTERACTION_SCORE_FUNCTIONS.keys()}"
            )
            interaction_score_function = "simple"
            logger.warning(warning_message)

        if interaction_score_function not in INTERACTION_SCORE_FUNCTIONS:
            error_message = (
                f"Score function {interaction_score_function} not supported. "
                f"Supported functions are: {list(INTERACTION_SCORE_FUNCTIONS.keys())}"
            )
            raise ValueError(error_message)
        self.score_function = INTERACTION_SCORE_FUNCTIONS[interaction_score_function]

        if self.categorize_scores:
            categorization_file = Path(self.config.get("interaction_categorization_file", ""))
            if not categorization_file.is_file():
                msg = f"Categorization file {categorization_file} does not exist."
                raise FileNotFoundError(msg)
            with categorization_file.open("r") as f:
                self.categories = json.load(f)

    def compute_interaction_score(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> Score:
        """Computes interaction scores for agent pairs and a scene-level score from scenario features.

        Args:
            scenario (Scenario): Scenario object containing scenario information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing computed features.

        Returns:
            ScenarioScores: An object containing computed interaction agent-pair scores and the scene-level score.
        """
        features = scenario_features.interaction_features
        if features is None or features.interaction_agent_indices is None or features.interaction_status is None:
            warning_message = f"Invalid interaction_features for {scenario.metadata.scenario_id}."
            warn(warning_message, UserWarning, stacklevel=2)
            return Score(agent_scores=None, agent_scores_valid=None, scene_score=None)

        # Get the agent weights
        scores = np.zeros(shape=(scenario.agent_data.num_agents,), dtype=np.float32)
        valid = np.zeros(shape=(scenario.agent_data.num_agents,), dtype=bool)
        weights = self.get_weights(scenario, scenario_features)

        # Get the interaction to consider
        interaction_agent_indices = features.interaction_agent_indices
        interaction_idxs = np.arange(len(interaction_agent_indices))
        if self.score_weighting_method == ScoreWeightingMethod.DISTANCE_TO_EGO_AGENT:
            interaction_idxs = [
                n for n, (i, j) in enumerate(interaction_agent_indices) if scenario.metadata.ego_vehicle_index in (i, j)
            ]
            interaction_agent_indices = [
                (i, j) for (i, j) in interaction_agent_indices if scenario.metadata.ego_vehicle_index in (i, j)
            ]

        for n, (i, j) in zip(interaction_idxs, interaction_agent_indices, strict=False):
            status = features.interaction_status[n]
            if status not in [InteractionStatus.COMPUTED_OK, InteractionStatus.PARTIAL_INVALID_HEADING]:
                continue

            # Compute the agent-pair scores using pre-capped inverse metrics from the feature pipeline.
            # Detection thresholds are stored in seconds so convert to 1/s for the inverse features.
            agent_pair_score = self.score_function(
                collision=features.collision[n] if features.collision is not None else 0.0,
                collision_weight=self.weights.collision,
                collision_detection=self.detections.collision,
                inv_mttcp=features.inv_mttcp[n] if features.inv_mttcp is not None else 0.0,
                inv_mttcp_weight=self.weights.mttcp,
                inv_mttcp_detection=1.0 / (self.detections.mttcp + EPSILON),
                inv_thw=features.inv_thw[n] if features.inv_thw is not None else 0.0,
                inv_thw_weight=self.weights.thw,
                inv_thw_detection=1.0 / (self.detections.thw + EPSILON),
                inv_ttc=features.inv_ttc[n] if features.inv_ttc is not None else 0.0,
                inv_ttc_weight=self.weights.ttc,
                inv_ttc_detection=1.0 / (self.detections.ttc + EPSILON),
                drac=features.drac[n] if features.drac is not None else 0.0,
                drac_weight=self.weights.drac,
                drac_detection=self.detections.drac,
            )
            # NOTE: this can be improved in the future. Currently, if "self.categorize_scores" is True, we compute the
            # weighted score value first and then categorize it. In this case is because we want to categorize the
            # agent accounting for its relevance to the ego-agent.
            scores[i] += weights[i] * agent_pair_score
            scores[j] += weights[j] * agent_pair_score
            valid[i] = True
            valid[j] = True

        if self.categorize_scores:
            # Categorize the scores
            for idx in range(scores.shape[0]):
                scores[idx] = self.categorize(scores[idx])

        # Replace NaNs with zeros as a safeguard
        scores = np.nan_to_num(scores, nan=0.0)

        # Normalize by the number of agents involved in at least one scored interaction
        denom = max(int(valid.sum()), 1)
        scene_score = np.clip(scores.sum() / denom, a_min=self.score_clip.min, a_max=self.score_clip.max)
        return Score(agent_scores=scores, agent_scores_valid=valid, scene_score=scene_score)

    def score_pair(self, feature_dict: dict[str, float]) -> float:
        """Compute the interaction score for a single agent pair from its feature values.

        Delegates to :attr:`score_function` with the scorer's configured detection thresholds and feature weights so
        that pair-level scoring is always consistent with :meth:`compute_interaction_score`. Missing keys in
        ``feature_dict`` default to 0.0. Detection thresholds for inverse features (``inv_mttcp``, ``inv_thw``,
        ``inv_ttc``) are pre-inverted here because the stored features are already reciprocal values.

        Args:
            feature_dict: Feature values for the pair, with keys ``"collision"``, ``"inv_mttcp"``, ``"inv_thw"``,
                ``"inv_ttc"``, and ``"drac"``. Missing keys are treated as 0.0.

        Returns:
            Scalar interaction score in ``[0, ∞)``.
        """
        return self.score_function(
            collision=feature_dict.get("collision", 0.0),
            collision_weight=self.weights.collision,
            collision_detection=self.detections.collision,
            inv_mttcp=feature_dict.get("inv_mttcp", 0.0),
            inv_mttcp_weight=self.weights.mttcp,
            inv_mttcp_detection=1.0 / (self.detections.mttcp + EPSILON),
            inv_thw=feature_dict.get("inv_thw", 0.0),
            inv_thw_weight=self.weights.thw,
            inv_thw_detection=1.0 / (self.detections.thw + EPSILON),
            inv_ttc=feature_dict.get("inv_ttc", 0.0),
            inv_ttc_weight=self.weights.ttc,
            inv_ttc_detection=1.0 / (self.detections.ttc + EPSILON),
            drac=feature_dict.get("drac", 0.0),
            drac_weight=self.weights.drac,
            drac_detection=self.detections.drac,
        )

    def compute_scene_score_from_pair_cache(
        self,
        scenario: Scenario,
        pair_score_cache: dict[tuple[int, int], float],
        weights: NDArray[np.float32],
        id_to_idx: dict[int, int],
    ) -> float:
        """Compute the scene-level interaction score from a pre-built pair-score cache.

        Mirrors the aggregation loop in :meth:`compute_interaction_score` — applying per-agent weights and the
        configured ``score_weighting_method`` policy — but accepts a caller-maintained dict of pair scores rather than
        an :class:`~characterization.schemas.scenario_features.Interaction` object. This lets the
        :class:`~characterization.probing.counterfactual_prober.CounterfactualProber` incrementally update only the
        pairs that change without re-running the full feature extractor.

        When ``score_weighting_method`` is ``DISTANCE_TO_EGO_AGENT``, only pairs where at least one agent index equals
        ``scenario.metadata.ego_vehicle_index`` contribute to the scene score, matching the filter applied inside
        :meth:`compute_interaction_score`.

        Args:
            scenario: Scenario containing the ego vehicle index and agent count.
            pair_score_cache: Mapping from canonical ``(min_id, max_id)`` agent-ID pairs to their scalar interaction
                scores.
            weights: Per-agent weight array of shape ``(num_agents,)``, index-aligned with
                ``scenario.agent_data.agent_ids``.
            id_to_idx: Mapping from agent ID to its position in the agent array; used to look up ``weights`` for each
                pair.

        Returns:
            Scene score clipped to the scorer's configured range, or 0.0 when the cache is empty.
        """
        n = scenario.agent_data.num_agents
        scores = np.zeros(n, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        ego_idx = scenario.metadata.ego_vehicle_index

        for (id_a, id_b), pair_score in pair_score_cache.items():
            i = id_to_idx[id_a]
            j = id_to_idx[id_b]
            if self.score_weighting_method == ScoreWeightingMethod.DISTANCE_TO_EGO_AGENT and ego_idx not in (i, j):
                continue
            scores[i] += weights[i] * pair_score
            scores[j] += weights[j] * pair_score
            valid[i] = True
            valid[j] = True

        scores = np.nan_to_num(scores, nan=0.0)
        denom = max(int(valid.sum()), 1)
        return float(np.clip(scores.sum() / denom, a_min=self.score_clip.min, a_max=self.score_clip.max))

    def compute_weights_for_probe(
        self,
        scenario: Scenario,
        baseline_weights: NDArray[np.float32],
        probed_idx: int,
        perturbed_traj: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Return updated per-agent weights that reflect a single perturbed agent trajectory.

        Only the distances that are actually affected by the trajectory change are recomputed, making this considerably
        cheaper than calling :meth:`~characterization.scorer.base_scorer.BaseScorer.get_weights` again on the full
        perturbed scenario.

        For :attr:`~characterization.scorer.base_scorer.ScoreWeightingMethod.UNIFORM` weighting the baseline weights are
        returned unchanged (no distances are needed). For
        :attr:`~characterization.scorer.base_scorer.ScoreWeightingMethod.DISTANCE_TO_EGO_AGENT`:

        - **Non-ego probe** (``probed_idx != ego_vehicle_index``): recomputes only the probed agent's distance to
          the ego trajectory. O(T) work.
        - **Ego probe** (``probed_idx == ego_vehicle_index``): recomputes the distance from each non-ego agent to
          the perturbed ego trajectory. O(N·T) work. The ego weight is always fixed at 1.0.

        Args:
            scenario: Original (unperturbed) scenario; provides ego index, agent types, and trajectories.
            baseline_weights: Per-agent weight array computed for the unperturbed scenario, shape ``(num_agents,)``.
            probed_idx: Index of the agent whose trajectory was replaced (position in the agent array).
            perturbed_traj: Counterfactual trajectory for the probed agent, shape ``(T, D)``.

        Returns:
            Updated per-agent weight array of the same shape as ``baseline_weights``. Weights for agents whose
            distances did not change are preserved from ``baseline_weights``.
        """
        if self.score_weighting_method != ScoreWeightingMethod.DISTANCE_TO_EGO_AGENT:
            return baseline_weights

        ego_idx = scenario.metadata.ego_vehicle_index
        trajs = scenario.agent_data.agent_trajectories  # (N, T, D)
        agent_types = scenario.agent_data.agent_types
        updated = baseline_weights.copy()

        def _new_weight(agent_index: int, dist: float) -> float:
            is_vru = agent_types[agent_index] in (AgentType.TYPE_CYCLIST, AgentType.TYPE_PEDESTRIAN)
            return self._weight_from_dist(
                dist,
                self.max_critical_distance,
                self.vru_priority_weight,
                is_vru=is_vru,
                reduce_distance_penalty=self.reduce_distance_penalty,
            )

        if probed_idx == ego_idx:
            # Ego trajectory changed → all non-ego agents' distances to ego change.
            for k in range(scenario.agent_data.num_agents):
                if k == ego_idx:
                    continue  # Ego weight is always 1.0
                dist = self._min_dist_between_trajs(trajs[k], perturbed_traj)
                updated[k] = _new_weight(k, dist)
        else:
            # Non-ego probe → only that agent's distance to ego changes.
            dist = self._min_dist_between_trajs(perturbed_traj, trajs[ego_idx])
            updated[probed_idx] = _new_weight(probed_idx, dist)

        return updated

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Computes interaction scores for agent pairs and a scene-level score from scenario features.

        Args:
            scenario (Scenario): Scenario object containing scenario information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing computed features.

        Returns:
            ScenarioScores: An object containing computed interaction agent-pair scores and the scene-level score.

        Raises:
            ValueError: If any required feature (agent_to_agent_closest_dists, interaction_agent_indices,
                interaction_status, collision, mttcp) is missing in scenario_features.
        """
        return ScenarioScores(
            metadata=scenario.metadata,
            interaction_scores=self.compute_interaction_score(scenario, scenario_features),
        )
