"""Aviation-domain pairwise interaction scorer."""

import numpy as np

from characterization.domains.aviation.schemas.scenario import Scenario
from characterization.domains.aviation.schemas.scenario_features import InteractionPairFeatures, ScenarioFeatures
from characterization.domains.aviation.scorer.base_scorer import AviationBaseScorer, AviationScorerConfig
from characterization.domains.aviation.scorer.score_functions import INTERACTION_SCORE_REGISTRY
from characterization.schemas.scenario_scores import AgentScore, ScenarioScores


class InteractionScorer(AviationBaseScorer):
    """Computes per-agent accumulated interaction scores and a scene-level aggregate.

    For each agent pair the pairwise interaction score is computed and added to both agents' running totals (scaled by
    each agent's weight). The scene score is the sum of all positive agent scores divided by the number of agents with
    a positive score, then clipped to ``config.score_clip``.
    """

    def __init__(self, config: AviationScorerConfig | None = None) -> None:
        """Initialize with aviation scorer configuration."""
        super().__init__(config)
        score_function = self.config.interaction_score_function
        if score_function not in INTERACTION_SCORE_REGISTRY:
            error_message = (
                f"Interaction score function '{score_function}' not supported. "
                f"Supported: {list(INTERACTION_SCORE_REGISTRY.keys())}"
            )
            raise ValueError(error_message)
        self._score_fn = INTERACTION_SCORE_REGISTRY[score_function]
        if self.config.categorize_scores:
            self.categories = self._load_categorization_file(self.config.interaction_categorization_file)

    def score_pair(self, pair: InteractionPairFeatures) -> float:
        """Compute the raw pairwise score for a single agent pair.

        Args:
            pair: Pre-computed interaction features for one pair.

        Returns:
            Scalar score for the pair.
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

            pair_score = self.score_pair(pair)
            raw_scores[idx_a] += weights[idx_a] * pair_score
            raw_scores[idx_b] += weights[idx_b] * pair_score

        if self.config.categorize_scores:
            for i in range(len(raw_scores)):
                raw_scores[i] = self.categorize(float(raw_scores[i]))

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
