"""Pairwise interaction scorer for SafeAir scenarios."""

import numpy as np

from safeair.scenario_characterization.scores.base_scorer import BaseScorer, ScorerConfig
from safeair.scenario_characterization.scores.score_functions import INTERACTION_SCORE_REGISTRY
from safeair.schemas.scenario import Scenario
from safeair.schemas.scenario_features import InteractionPairFeatures, ScenarioFeatures
from safeair.schemas.scenario_scores import AgentScore, ScenarioScores


class InteractionScorer(BaseScorer):
    """Computes per-agent accumulated interaction scores and a scene-level aggregate.

    For each agent pair the pairwise interaction score is computed and added to both agents' running totals (scaled by
    each agent's weight). The scene score is the sum of all positive agent scores divided by the number of agents with
    a positive score, then clipped to ``config.score_clip``.

    Args:
        config: Scorer configuration. Defaults to ``ScorerConfig()`` if not provided.
        score_function: Name of the scoring function to use. Currently only ``"simple"`` is supported.
    """

    def __init__(self, config: ScorerConfig | None = None, *, score_function: str = "simple") -> None:
        """Intitialize with the given configuration and scoring function."""
        super().__init__(config)
        if score_function not in INTERACTION_SCORE_REGISTRY:
            error_message = (
                f"Interaction score function '{score_function}' not supported. "
                f"Supported: {list(INTERACTION_SCORE_REGISTRY.keys())}"
            )
            raise ValueError(error_message)

        self._score_fn = INTERACTION_SCORE_REGISTRY[score_function]

    def score_pair(self, pair: InteractionPairFeatures) -> float:
        """Compute the raw pairwise score for a single agent pair.

        Args:
            pair: Pre-computed interaction features for one pair.

        Returns:
            Scalar score in [0, 1].
        """
        raw = self._score_fn(  # pyright: ignore[reportCallIssue]
            loss_of_separation=pair.loss_of_separation if pair.loss_of_separation is not None else 0.0,
            loss_of_separation_weight=self.config.weights.loss_of_separation,
            loss_of_separation_detection=self.config.detections.loss_of_separation,
            mttcp=pair.mttcp if pair.mttcp is not None else float("inf"),
            mttcp_weight=self.config.weights.mttcp,
            mttcp_detection=self.config.detections.mttcp,
            thw=pair.thw if pair.thw is not None else float("inf"),
            thw_weight=self.config.weights.thw,
            thw_detection=self.config.detections.thw,
            ttc=pair.ttc if pair.ttc is not None else float("inf"),
            ttc_weight=self.config.weights.ttc,
            ttc_detection=self.config.detections.ttc,
            drac=pair.drac if pair.drac is not None else 0.0,
            drac_weight=self.config.weights.drac,
            drac_detection=self.config.detections.drac,
        )
        return float(np.nan_to_num(raw, nan=0.0))

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Compute per-agent interaction scores and a scene-level score.

        Args:
            scenario: The scenario to score.
            scenario_features: Pre-computed features containing ``interaction_features``.

        Returns:
            ScenarioScores with ``interaction_scores`` and ``interaction_scene_score`` populated.
             ``individual_scores`` and ``individual_scene_score`` are left empty/None.
        """
        agent_id_to_idx: dict[int, int] = {int(aid): i for i, aid in enumerate(scenario.agent_data.agent_ids)}
        weights = self._compute_agent_weights(scenario)
        raw_scores = np.zeros(scenario.agent_data.num_agents, dtype=np.float32)
        ego_id = int(scenario.metadata.ego_agent_id) if self.config.ego_pairs_only else None

        for pair in scenario_features.interaction_features:
            if ego_id is not None and ego_id not in (pair.agent_id_a, pair.agent_id_b):
                continue

            idx_a = agent_id_to_idx.get(pair.agent_id_a)
            idx_b = agent_id_to_idx.get(pair.agent_id_b)
            if idx_a is None or idx_b is None:
                continue

            pair_score = self._score_fn(  # pyright: ignore[reportCallIssue]
                loss_of_separation=pair.loss_of_separation if pair.loss_of_separation is not None else 0.0,
                loss_of_separation_weight=self.config.weights.loss_of_separation,
                loss_of_separation_detection=self.config.detections.loss_of_separation,
                mttcp=pair.mttcp if pair.mttcp is not None else float("inf"),
                mttcp_weight=self.config.weights.mttcp,
                mttcp_detection=self.config.detections.mttcp,
                thw=pair.thw if pair.thw is not None else float("inf"),
                thw_weight=self.config.weights.thw,
                thw_detection=self.config.detections.thw,
                ttc=pair.ttc if pair.ttc is not None else float("inf"),
                ttc_weight=self.config.weights.ttc,
                ttc_detection=self.config.detections.ttc,
                drac=pair.drac if pair.drac is not None else 0.0,
                drac_weight=self.config.weights.drac,
                drac_detection=self.config.detections.drac,
            )
            pair_score = float(np.nan_to_num(pair_score, nan=0.0))

            raw_scores[idx_a] += weights[idx_a] * pair_score
            raw_scores[idx_b] += weights[idx_b] * pair_score

        agent_scores = [
            AgentScore(agent_id=int(aid), score=float(raw_scores[i]))
            for i, aid in enumerate(scenario.agent_data.agent_ids)
        ]

        denom = max(int(np.sum(raw_scores > 0.0)), 1)
        scene_score = self.config.score_clip.clip(float(raw_scores.sum() / denom))

        return ScenarioScores(
            scenario_id=scenario_features.scenario_id,
            interaction_scores=agent_scores,
            interaction_scene_score=scene_score,
        )
