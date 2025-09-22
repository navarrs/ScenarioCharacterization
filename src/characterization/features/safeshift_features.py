from omegaconf import DictConfig

from characterization.features.base_feature import BaseFeature
from characterization.features.individual_features import IndividualFeatures
from characterization.features.interaction_features import InteractionFeatures
from characterization.schemas import Scenario, ScenarioFeatures
from characterization.utils.common import AgentTrajectoryMasker
from characterization.utils.geometric_utils import compute_agent_to_agent_closest_dists
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


class SafeShiftFeatures(BaseFeature):
    """Comprehensive scenario feature extractor combining individual and interaction features.

    Attributes:
        config (DictConfig): Configuration parameters for comprehensive feature computation.
        features (Any): Feature-specific configuration extracted from config.
        characterizer_type (str): Type identifier, always "feature".
        return_criterion (ReturnCriterion): Criterion for returning results (CRITICAL or AVERAGE).
        agent_i (InteractionAgent): Utility object for interaction computations (agent i).
        agent_j (InteractionAgent): Utility object for interaction computations (agent j).
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the SafeShiftFeatures comprehensive extractor.

        Args:
            config (DictConfig): Configuration dictionary containing feature parameters.
                Expected keys:
                - return_criterion (str, optional): Determines whether to return 'critical'
                  (max/min/sum values) or 'average' statistics for all features. Defaults to 'critical'.
                - features (optional): Feature-specific configuration parameters for both
                  individual and interaction feature computation.
        """
        super().__init__(config)

    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Compute comprehensive scenario features combining individual and interaction analysis.

        Args:
            scenario (Scenario): Complete scenario data containing:
                - agent_data: Agent trajectories, positions, velocities, dimensions, headings,
                  validity masks, and type classifications
                - metadata: Scenario parameters including timestamps, speed thresholds,
                  distance limits, and interaction configuration values
                - static_map_data: Map conflict points and precomputed distances for
                  comprehensive spatial analysis

        Returns:
            ScenarioFeatures: Comprehensive feature object containing:
                - metadata: Original scenario metadata for reference and traceability
                - individual_features: Individual agent motion analysis including:
                  * Speed profiles and speed limit compliance
                  * Acceleration and deceleration patterns
                  * Jerk calculations for comfort assessment
                  * Waiting behaviors near conflict points (periods, intervals, distances)
                - interaction_features: Pairwise agent interaction analysis including:
                  * Spatial relationships (separation, intersection, collision detection)
                  * Temporal conflict metrics (mTTCP, TTC, THW)
                  * Safety indicators (DRAC - deceleration rate to avoid collision)
                  * Interaction status and agent pair metadata
                - agent_to_agent_closest_dists: Minimum pairwise distances between all
                  agent combinations across all timesteps for proximity analysis

        Raises:
            ValueError: If an unknown return criterion is provided or insufficient scenario data.
        """
        # Unpack senario fields
        agent_data = scenario.agent_data
        agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
        agent_positions = agent_trajectories.agent_xyz_pos
        return ScenarioFeatures(
            metadata=scenario.metadata,
            individual_features=IndividualFeatures.compute_individual_features(scenario, self.return_criterion),
            interaction_features=InteractionFeatures.compute_interaction_features(scenario, self.return_criterion),
            agent_to_agent_closest_dists=compute_agent_to_agent_closest_dists(agent_positions),
        )
