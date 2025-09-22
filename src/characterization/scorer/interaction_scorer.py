import numpy as np
from omegaconf import DictConfig

from characterization.features.interaction_features import InteractionStatus
from characterization.schemas import Scenario, ScenarioFeatures, ScenarioScores, Score
from characterization.scorer.base_scorer import BaseScorer
from characterization.utils.io_utils import get_logger

from .score_utils import INTERACTION_SCORE_FUNCTIONS

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

    def compute_interaction_score(self, scenario: Scenario, scenario_features: ScenarioFeatures) -> Score:
        """Computes interaction scores for agent pairs and a scene-level score from scenario features.

        Args:
            scenario (Scenario): Scenario object containing scenario information.
            scenario_features (ScenarioFeatures): ScenarioFeatures object containing computed features.

        Returns:
            ScenarioScores: An object containing computed interaction agent-pair scores and the scene-level score.
        """
        scores = np.zeros(shape=(scenario.agent_data.num_agents,), dtype=np.float32)
        features = scenario_features.interaction_features
        if features is None or features.interaction_agent_indices is None or features.interaction_status is None:
            warning_message = "interaction_features is None. Cannot compute interaction scores."
            logger.warning(warning_message)
            return Score(agent_scores=scores, scene_score=0.0)

        # Get the agent weights
        weights = self.get_weights(scenario, scenario_features)

        interaction_agent_indices = features.interaction_agent_indices
        if self.score_wrt_ego_only:
            interaction_agent_indices = [
                (i, j) for i, j in interaction_agent_indices if scenario.metadata.ego_vehicle_index in (i, j)
            ]
        for n, (i, j) in enumerate(interaction_agent_indices):
            status = features.interaction_status[n]
            if status != InteractionStatus.COMPUTED_OK:
                continue

            # Compute the agent-pair scores
            agent_pair_score = self.score_function(
                collision=features.collision[n] if features.collision is not None else 0.0,
                collision_weight=self.weights.collision,
                mttcp=features.mttcp[n] if features.mttcp is not None else np.inf,
                mttcp_weight=self.weights.mttcp,
                mttcp_detection=self.detections.mttcp,
                thw=features.thw[n] if features.thw is not None else np.inf,
                thw_weight=self.weights.thw,
                thw_detection=self.detections.thw,
                ttc=features.ttc[n] if features.ttc is not None else np.inf,
                ttc_weight=self.weights.ttc,
                ttc_detection=self.detections.ttc,
                drac=features.drac[n] if features.drac is not None else 0.0,
                drac_weight=self.weights.drac,
                drac_detection=self.detections.drac,
            )
            scores[i] += weights[i] * agent_pair_score
            scores[j] += weights[j] * agent_pair_score

        # Replace NaNs with zeros as a safeguard
        scores = np.nan_to_num(scores, nan=0.0)

        # Normalize the scores
        denom = max(np.where(scores > 0.0)[0].shape[0], 1)
        scene_score = np.clip(scores.sum() / denom, a_min=self.score_clip.min, a_max=self.score_clip.max)
        return Score(agent_scores=scores, scene_score=scene_score)

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
