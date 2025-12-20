import json
from itertools import combinations
from pathlib import Path
from warnings import warn

import numpy as np
from omegaconf import DictConfig

import characterization.features.interaction_utils as interaction
from characterization.features.base_feature import BaseFeature
from characterization.schemas import Interaction, Scenario, ScenarioFeatures
from characterization.utils.common import (
    MIN_VALID_POINTS,
    SMALL_EPS,
    AgentTrajectoryMasker,
    FeatureType,
    InteractionStatus,
    ReturnCriterion,
    categorize_from_thresholds,
)
from characterization.utils.geometric_utils import compute_agent_to_agent_closest_dists
from characterization.utils.io_utils import get_logger
from characterization.utils.scenario_types import AgentPairType, get_agent_pair_type

logger = get_logger(__name__)


class InteractionFeatures(BaseFeature):
    """Computes pairwise interaction features between agents in a scenario.

    Attributes:
        config (DictConfig): Configuration parameters for interaction feature computation.
        features (Any): Feature-specific configuration extracted from config.
        characterizer_type (str): Type identifier, always "feature".
        return_criterion (ReturnCriterion): Criterion for returning results.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the InteractionFeatures extractor.

        Args:
            config (DictConfig): Configuration dictionary containing interaction feature parameters.
        """
        super().__init__(config)

        self.categorize_features = FeatureType(self.config.get("feature_type", "continuous")) == FeatureType.CATEGORICAL
        if self.categorize_features:
            vehicle_vehicle_file = Path(self.config.get("vehicle_vehicle_categorization_file", ""))
            assert vehicle_vehicle_file.is_file(), f"Categorization file {vehicle_vehicle_file} does not exist."
            with vehicle_vehicle_file.open("r") as f:
                self.vehicle_vehicle_categories = json.load(f)

            vehicle_pedestrian_file = Path(self.config.get("vehicle_pedestrian_categorization_file", ""))
            assert vehicle_pedestrian_file.is_file(), f"Categorization file {vehicle_pedestrian_file} does not exist."
            with vehicle_pedestrian_file.open("r") as f:
                self.vehicle_pedestrian_categories = json.load(f)

            vehicle_cyclist_file = Path(self.config.get("vehicle_cyclist_categorization_file", ""))
            assert vehicle_cyclist_file.is_file(), f"Categorization file {vehicle_cyclist_file} does not exist."
            with vehicle_cyclist_file.open("r") as f:
                self.vehicle_cyclist_categories = json.load(f)

            pedestrian_pedestrian_file = Path(self.config.get("pedestrian_pedestrian_categorization_file", ""))
            assert pedestrian_pedestrian_file.is_file(), (
                f"Categorization file {pedestrian_pedestrian_file} does not exist."
            )
            with pedestrian_pedestrian_file.open("r") as f:
                self.pedestrian_pedestrian_categories = json.load(f)

            pedestrian_cyclist_file = Path(self.config.get("pedestrian_cyclist_categorization_file", ""))
            assert pedestrian_cyclist_file.is_file(), f"Categorization file {pedestrian_cyclist_file} does not exist."
            with pedestrian_cyclist_file.open("r") as f:
                self.pedestrian_cyclist_categories = json.load(f)

            cyclist_cyclist_file = Path(self.config.get("cyclist_cyclist_categorization_file", ""))
            assert cyclist_cyclist_file.is_file(), f"Categorization file {cyclist_cyclist_file} does not exist."
            with cyclist_cyclist_file.open("r") as f:
                self.cyclist_cyclist_categories = json.load(f)

    def categorize(self, value: float, agent_pair_type: AgentPairType, feature_name: str) -> float:
        """Categorize a feature value based on agent type and predefined percentiles.

        Args:
            value (float): The feature value to categorize.
            agent_pair_type (AgentPairType): The type of the agent pair (VEHICLE_VEHICLE, VEHICLE_CYCLIST, etc.).
            feature_name (str): The name of the feature being categorized.

        Returns:
            float: The categorized feature value based on percentiles.
        """
        match agent_pair_type:
            case AgentPairType.TYPE_VEHICLE_VEHICLE:
                categories = self.vehicle_vehicle_categories.get(feature_name, None)
            case AgentPairType.TYPE_VEHICLE_PEDESTRIAN:
                categories = self.vehicle_pedestrian_categories.get(feature_name, None)
            case AgentPairType.TYPE_VEHICLE_CYCLIST:
                categories = self.vehicle_cyclist_categories.get(feature_name, None)
            case AgentPairType.TYPE_PEDESTRIAN_PEDESTRIAN:
                categories = self.pedestrian_pedestrian_categories.get(feature_name, None)
            case AgentPairType.TYPE_PEDESTRIAN_CYCLIST:
                categories = self.pedestrian_cyclist_categories.get(feature_name, None)
            case AgentPairType.TYPE_CYCLIST_CYCLIST:
                categories = self.cyclist_cyclist_categories.get(feature_name, None)
            case _:
                logger.warning("Unknown agent type: %s", agent_pair_type)
                return -1.0

        if categories is None:
            logger.warning("No categories found for feature %s and agent type %s.", feature_name, agent_pair_type)
            return -1.0

        threshold_values = list(categories.values())
        return float(categorize_from_thresholds(value, threshold_values))

    def compute_interaction_features(self, scenario: Scenario) -> Interaction | None:
        """Compute comprehensive pairwise interaction features for all agent combinations.

        Args:
            scenario (Scenario): Complete scenario data containing:
                - agent_data: Agent positions, velocities, headings, dimensions, validity masks, and types
                - metadata: Timestamps, distance thresholds, speed limits, and interaction parameters
                - static_map_data: Map conflict points and agent distances to conflict points

        Returns:
            Interaction: Structured object containing computed interaction features:
                - separation: Minimum/average spatial distances between agent pairs
                - intersection: Sum/average of geometric intersection events
                - collision: Sum/average of collision detection events (separation <= breach threshold)
                - mttcp: Minimum/average time to conflict point for each agent pair
                - thw: Minimum/average time headway for leader-follower interactions
                - ttc: Minimum/average time to collision for leader-follower interactions
                - drac: Maximum/average deceleration rate to avoid collision
                - interaction_status: Processing status for each agent pair (computed/invalid/stationary)
                - interaction_agent_indices: Agent pair indices (i, j) for each interaction
                - interaction_agent_types: Agent type pairs for each interaction
                Returns None if scenario has fewer than 2 agents.

        Note:
            - Agent pairs must have overlapping valid timesteps to be processed
            - Stationary agents (both below stationary_speed threshold) are marked as STATIONARY
            - Agents beyond agent_to_agent_max_distance are marked as DISTANCE_TOO_FAR
            - Leader-follower metrics (THW, TTC, DRAC) require agents with similar headings
            - All feature arrays use dtype np.float32, with np.nan for invalid interactions
        """
        metadata = scenario.metadata
        agent_data = scenario.agent_data
        map_data = scenario.static_map_data

        # TODO: Refactor method to use AgentTrajectoryMasker instead of InteractionAgent
        agent_i = interaction.InteractionAgent()
        agent_j = interaction.InteractionAgent()

        agent_combinations = list(combinations(range(agent_data.num_agents), 2))
        if len(agent_combinations) == 0:
            warning_message = "No agent combinations found. Ensure that the scenario has at least two agents."
            warn(warning_message, UserWarning, stacklevel=2)
            return None

        agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
        agent_types = agent_data.agent_types
        agent_masks = agent_trajectories.agent_valid.squeeze(-1).astype(bool)
        agent_positions = agent_trajectories.agent_xyz_pos
        agent_lengths = agent_trajectories.agent_lengths.squeeze(-1)
        agent_widths = agent_trajectories.agent_widths.squeeze(-1)
        agent_heights = agent_trajectories.agent_heights.squeeze(-1)

        # NOTE: this is also computed as a feature in the individual features.
        agent_velocities = np.linalg.norm(agent_trajectories.agent_xy_vel, axis=-1) + SMALL_EPS
        agent_headings = np.rad2deg(agent_trajectories.agent_headings)
        conflict_points = map_data.map_conflict_points if map_data is not None else None
        dists_to_conflict_points = map_data.agent_distances_to_conflict_points if map_data is not None else None

        # Meta information
        stationary_speed = metadata.max_stationary_speed
        agent_to_agent_max_distance = metadata.agent_to_agent_max_distance
        agent_to_conflict_point_max_distance = metadata.agent_to_conflict_point_max_distance
        agent_to_agent_distance_breach = metadata.agent_to_agent_distance_breach
        heading_threshold = metadata.heading_threshold
        agent_max_deceleration = metadata.agent_max_deceleration

        # Meta information to be included in ScenarioFeatures Valid interactions will be added 'agent_pair_indeces' and
        # 'interaction_status'
        scenario_interaction_statuses = [InteractionStatus.UNKNOWN for _ in agent_combinations]
        scenario_agent_pair_indeces = [(i, j) for i, j in agent_combinations]
        scenario_agents_pair_types = [(agent_types[i], agent_types[j]) for i, j in agent_combinations]

        num_interactions = len(agent_combinations)
        # Features to be included in ScenarioFeatures
        scenario_separations = np.full(num_interactions, np.nan, dtype=np.float32)
        scenario_intersections = np.full(num_interactions, np.nan, dtype=np.float32)
        scenario_collisions = np.full(num_interactions, np.nan, dtype=np.float32)
        scenario_mttcps = np.full(num_interactions, np.nan, dtype=np.float32)
        scenario_inv_mttcps = np.full(num_interactions, np.nan, dtype=np.float32)
        scenario_thws = np.full(num_interactions, np.nan, dtype=np.float32)
        scenario_inv_thws = np.full(num_interactions, np.nan, dtype=np.float32)
        scenario_ttcs = np.full(num_interactions, np.nan, dtype=np.float32)
        scenario_inv_ttcs = np.full(num_interactions, np.nan, dtype=np.float32)
        scenario_dracs = np.full(num_interactions, np.nan, dtype=np.float32)

        # Compute distance to conflict points
        for n, (i, j) in enumerate(agent_combinations):
            agent_i.reset()
            agent_j.reset()

            # There should be at least two valid timestamps for the combined agents masks
            mask_i, mask_j = agent_masks[i], agent_masks[j]
            mask = np.where(mask_i & mask_j)[0]
            if not mask.sum():
                # No valid data for this pair of agents
                scenario_interaction_statuses[n] = InteractionStatus.MASK_NOT_VALID
                continue

            # TODO: Refactor to use AgentMasker since this is doing redundant stuff that the masker already does.
            agent_i.position, agent_j.position = agent_positions[i][mask], agent_positions[j][mask]
            agent_i.speed, agent_j.speed = agent_velocities[i][mask], agent_velocities[j][mask]
            agent_i.heading, agent_j.heading = agent_headings[i][mask], agent_headings[j][mask]
            agent_i.length, agent_j.length = agent_lengths[i][mask], agent_lengths[j][mask]
            agent_i.width, agent_j.width = agent_widths[i][mask], agent_widths[j][mask]
            agent_i.height, agent_j.height = agent_heights[i][mask], agent_heights[j][mask]

            agent_i.agent_type, agent_j.agent_type = agent_types[i], agent_types[j]
            agent_i.lane, agent_j.lane = None, None  # TODO: Add lane information if available

            if conflict_points is not None and dists_to_conflict_points is not None:
                agent_j.dists_to_conflict = dists_to_conflict_points[i][mask]
                agent_j.dists_to_conflict = dists_to_conflict_points[j][mask]

            # Check if agents are within a valid distance threshold to compute interactions
            separations = interaction.compute_separation(agent_i, agent_j)
            if not np.any(separations <= agent_to_agent_max_distance):
                scenario_interaction_statuses[n] = InteractionStatus.DISTANCE_TOO_FAR
                continue

            # Check if agents are stationary
            agent_i.stationary_speed = stationary_speed
            agent_j.stationary_speed = stationary_speed
            if agent_i.is_stationary and agent_j.is_stationary:
                scenario_interaction_statuses[n] = InteractionStatus.STATIONARY
                continue

            # Compute interaction features
            # separations = interaction.compute_separation(agent_i, agent_j)
            intersections = interaction.compute_intersections(agent_i, agent_j)
            collisions = (separations <= agent_to_agent_distance_breach) | intersections
            intersections = intersections.astype(np.float32)
            collisions = collisions.astype(np.float32)

            # Minimum time to conflict point (mTTCP) is calculated from t=0 to t=first time on of the agents cross that
            # point, aligned to what's done in ExiD: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9827305)
            mttcps = interaction.compute_mttcp(agent_i, agent_j, agent_to_conflict_point_max_distance)

            # To compute Time Headway (THW), Time to Collision (TTC), and Deceleration Rate to Avoid Collision (DRAC),
            # we currently assume that agents are sharing the same lane.
            valid_headings = interaction.find_valid_headings(agent_i, agent_j, heading_threshold)
            if valid_headings.shape[0] < MIN_VALID_POINTS:
                thws = np.full(1, np.inf, dtype=np.float32)
                ttcs = np.full(1, np.inf, dtype=np.float32)
                dracs = np.full(1, 0.0, dtype=np.float32)
                scenario_interaction_statuses[n] = InteractionStatus.PARTIAL_INVALID_HEADING
            else:
                # At this point agents are sharing a lane and have at least two steps with headings within the defined
                # threshold. TODO: check if steps are consecutive
                # Now we need to check if who is the leading agent within the interaction.
                leading_agent = interaction.find_leading_agent(agent_i, agent_j, valid_headings)

                # Now compute leader-follower interaction state
                thws = interaction.compute_thw(agent_i, agent_j, leading_agent, valid_headings)
                ttcs = interaction.compute_ttc(agent_i, agent_j, leading_agent, valid_headings)
                dracs = interaction.compute_drac(
                    agent_i, agent_j, leading_agent, valid_headings, agent_max_deceleration
                )
                scenario_interaction_statuses[n] = InteractionStatus.COMPUTED_OK

            match self.return_criterion:
                case ReturnCriterion.CRITICAL:
                    separation = separations.min()
                    intersection = intersections.sum()
                    collision = collisions.sum()
                    mttcp = mttcps.min()
                    ttc = ttcs.min()
                    thw = thws.min()
                    drac = dracs.max()
                case ReturnCriterion.AVERAGE:
                    # NOTE: whenever there are valid values within a trajectory, this return the mean over those values
                    # and not the entire trajectory.
                    separation = separations.mean()
                    intersection = intersections.mean()
                    collision = collisions.mean()
                    mttcp = mttcps.mean()
                    ttc = ttcs.mean()
                    thw = thws.mean()
                    drac = dracs.mean()
                case _:
                    error_message = f"{self.return_criterion} not supported. Expected 'critical' or 'average'."
                    raise ValueError(error_message)

            # TODO: add the stability cap to configuration
            inv_mttcp = min(1.0 / (mttcp + SMALL_EPS), 10.0)
            inv_ttc = min(1.0 / (ttc + SMALL_EPS), 10.0)
            inv_thw = min(1.0 / (thw + SMALL_EPS), 10.0)

            if self.categorize_features:
                agent_pair_type = get_agent_pair_type(agent_i.agent_type, agent_j.agent_type)
                separation = self.categorize(separation, agent_pair_type, "separation")
                intersection = self.categorize(intersection, agent_pair_type, "intersection")
                collision = self.categorize(collision, agent_pair_type, "collision")
                mttcp = self.categorize(mttcp, agent_pair_type, "mttcp")
                inv_mttcp = self.categorize(inv_mttcp, agent_pair_type, "inv_mttcp")
                thw = self.categorize(thw, agent_pair_type, "thw")
                inv_thw = self.categorize(inv_thw, agent_pair_type, "inv_thw")
                ttc = self.categorize(ttc, agent_pair_type, "ttc")
                inv_ttc = self.categorize(inv_ttc, agent_pair_type, "inv_ttc")
                drac = self.categorize(drac, agent_pair_type, "drac")

            # Store computed features in the state dictionary
            scenario_separations[n] = separation
            scenario_intersections[n] = intersection
            scenario_collisions[n] = collision
            scenario_mttcps[n] = mttcp
            scenario_inv_mttcps[n] = inv_mttcp
            scenario_thws[n] = thw
            scenario_inv_thws[n] = inv_thw
            scenario_ttcs[n] = ttc
            scenario_inv_ttcs[n] = inv_ttc
            scenario_dracs[n] = drac

        return Interaction(
            separation=scenario_separations,
            intersection=scenario_intersections,
            collision=scenario_collisions,
            mttcp=scenario_mttcps,
            inv_mttcp=scenario_inv_mttcps,
            thw=scenario_thws,
            inv_thw=scenario_inv_thws,
            ttc=scenario_ttcs,
            inv_ttc=scenario_inv_ttcs,
            drac=scenario_dracs,
            interaction_status=scenario_interaction_statuses,
            interaction_agent_indices=scenario_agent_pair_indeces,
            interaction_agent_types=scenario_agents_pair_types,
        )

    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Compute scenario features focused on agent-to-agent interactions.

        Args:
            scenario (Scenario): Complete scenario data containing:
                - agent_data: Agent trajectories, dimensions, headings, and validity information
                - metadata: Scenario parameters including distance thresholds, speed limits,
                  and interaction-specific configuration values
                - static_map_data: Map conflict points and precomputed distances for mTTCP analysis

        Returns:
            ScenarioFeatures: Feature object containing:
                - metadata: Original scenario metadata for reference and traceability
                - interaction_features: Comprehensive pairwise interaction analysis including:
                  * Spatial relationships (separation, intersection, collision detection)
                  * Temporal conflict metrics (mTTCP, TTC, THW)
                  * Safety indicators (DRAC - deceleration rate to avoid collision)
                  * Interaction status and agent pair metadata

        Raises:
            ValueError: If the scenario contains fewer than 2 agents.
        """
        # Unpack senario fields
        agent_to_agent_closest_dists = None
        if self.compute_agent_to_agent_closest_dists:
            agent_data = scenario.agent_data
            agent_trajectories = AgentTrajectoryMasker(agent_data.agent_trajectories)
            agent_positions = agent_trajectories.agent_xyz_pos
            agent_to_agent_closest_dists = compute_agent_to_agent_closest_dists(agent_positions)

        return ScenarioFeatures(
            metadata=scenario.metadata,
            interaction_features=self.compute_interaction_features(scenario),
            agent_to_agent_closest_dists=agent_to_agent_closest_dists,
        )
