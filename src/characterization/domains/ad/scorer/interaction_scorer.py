"""AD-domain interaction scorer."""

from warnings import warn

import numpy as np
from omegaconf import DictConfig

from characterization.domains.ad.schemas import InteractionPairFeatures, Scenario, ScenarioFeatures
from characterization.domains.ad.schemas.scenario_scores import Score
from characterization.domains.ad.scorer.base_scorer import ADBaseScorer, ADScorerConfig
from characterization.schemas.scenario_scores import AgentScore, ScenarioScores
from characterization.scorer.base_scorer import ScoreWeightingMethod
from characterization.utils.logging_utils import get_pylogger

from .score_functions import INTERACTION_SCORE_FUNCTIONS

logger = get_pylogger(__name__)


class InteractionScorer(ADBaseScorer):
    """Computes pairwise interaction scores and a scene-level score from scenario features."""

    def __init__(self, config: DictConfig | ADScorerConfig | None = None) -> None:
        """Initialize with AD scorer configuration.

        Args:
            config: Configuration as an :class:`ADScorerConfig`, an OmegaConf ``DictConfig``, or ``None``.
        """
        super().__init__(config)

        score_function_name = self.config.interaction_score_function
        if score_function_name not in INTERACTION_SCORE_FUNCTIONS:
            error_message = (
                f"Score function {score_function_name} not supported. "
                f"Supported functions are: {list(INTERACTION_SCORE_FUNCTIONS.keys())}"
            )
            raise ValueError(error_message)
        self._score_fn = INTERACTION_SCORE_FUNCTIONS[score_function_name]

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
            collision=pair.collision or 0.0,
            collision_weight=self.config.weights.collision,
            collision_detection=self.config.detections.collision,
            mttcp=pair.mttcp if pair.mttcp is not None else float("inf"),
            mttcp_weight=self.config.weights.mttcp,
            mttcp_detection=1.0 / (self.config.detections.mttcp + 1e-6),
            thw=pair.thw if pair.thw is not None else float("inf"),
            thw_weight=self.config.weights.thw,
            thw_detection=1.0 / (self.config.detections.thw + 1e-6),
            ttc=pair.ttc if pair.ttc is not None else float("inf"),
            ttc_weight=self.config.weights.ttc,
            ttc_detection=1.0 / (self.config.detections.ttc + 1e-6),
            drac=pair.drac or 0.0,
            drac_weight=self.config.weights.drac,
            drac_detection=self.config.detections.drac,
        )
        return float(np.nan_to_num(raw, nan=0.0))

    def _compute_interaction_score(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> Score:
        """Compute pairwise interaction scores as an internal numpy-based Score object.

        Args:
            scenario: Scenario with agent metadata.
            scenario_features: Pre-computed interaction features.

        Returns:
            :class:`Score` with numpy agent_scores array and scene_score.
        """
        features = scenario_features.interaction_features
        if not features:
            warning_message = f"Invalid interaction_features for {scenario.metadata.scenario_id}."
            warn(warning_message, UserWarning, stacklevel=3)
            return Score(agent_scores=None, agent_scores_valid=None, scene_score=None)

        scores = np.zeros(shape=(scenario.agent_data.num_agents,), dtype=np.float32)
        valid = np.zeros(shape=(scenario.agent_data.num_agents,), dtype=bool)
        weights = self.get_weights(scenario, scenario_features)

        # Filter to ego-only pairs if configured
        active_features = features
        if self.config.ego_pairs_only or self.config.score_weighting_method == ScoreWeightingMethod.DISTANCE_TO_EGO:
            ego = scenario.metadata.ego_agent_index
            active_features = [f for f in features if ego in (f.agent_id_a, f.agent_id_b)]

        for pair_feat in active_features:
            i, j = pair_feat.agent_id_a, pair_feat.agent_id_b
            agent_pair_score = self._score_fn(  # pyright: ignore[reportCallIssue]
                collision=pair_feat.collision or 0.0,
                collision_weight=self.config.weights.collision,
                collision_detection=self.config.detections.collision,
                mttcp=pair_feat.mttcp if pair_feat.mttcp is not None else float("inf"),
                mttcp_weight=self.config.weights.mttcp,
                mttcp_detection=1.0 / (self.config.detections.mttcp + 1e-6),
                thw=pair_feat.thw if pair_feat.thw is not None else float("inf"),
                thw_weight=self.config.weights.thw,
                thw_detection=1.0 / (self.config.detections.thw + 1e-6),
                ttc=pair_feat.ttc if pair_feat.ttc is not None else float("inf"),
                ttc_weight=self.config.weights.ttc,
                ttc_detection=1.0 / (self.config.detections.ttc + 1e-6),
                drac=pair_feat.drac or 0.0,
                drac_weight=self.config.weights.drac,
                drac_detection=self.config.detections.drac,
            )
            scores[i] += weights[i] * agent_pair_score
            scores[j] += weights[j] * agent_pair_score
            valid[i] = True
            valid[j] = True

        if self.config.categorize_scores:
            for idx in range(scores.shape[0]):
                scores[idx] = self.categorize(scores[idx])

        scores = np.nan_to_num(scores, nan=0.0)
        denom = max(int(valid.sum()), 1)
        scene_score = float(
            np.clip(scores.sum() / denom, a_min=self.config.score_clip.min, a_max=self.config.score_clip.max)
        )
        return Score(agent_scores=scores, agent_scores_valid=valid, scene_score=scene_score)

    def compute(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> ScenarioScores:
        """Compute interaction scores and a scene-level score.

        Args:
            scenario: Scenario with agent metadata.
            scenario_features: Pre-computed interaction features.

        Returns:
            :class:`ScenarioScores` with ``interaction_scores`` and ``interaction_scene_score`` populated.
        """
        internal = self._compute_interaction_score(scenario, scenario_features)
        agent_scores_list: list[AgentScore] = []
        if internal.agent_scores is not None:
            agent_scores_list = [
                AgentScore(agent_id=int(i), score=float(s)) for i, s in enumerate(internal.agent_scores)
            ]
        return ScenarioScores(
            scenario_id=scenario.metadata.scenario_id,
            interaction_scores=agent_scores_list,
            interaction_scene_score=internal.scene_score,
        )
