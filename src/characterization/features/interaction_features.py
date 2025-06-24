import itertools

import numpy as np
from omegaconf import DictConfig

import characterization.features.interaction_utils as interaction
from characterization.features.base_feature import BaseFeature
from characterization.utils.common import EPS, InteractionStatus, get_logger
from characterization.utils.schemas import Scenario, ScenarioFeatures

logger = get_logger(__name__)


class InteractionFeatures(BaseFeature):
    def __init__(self, config: DictConfig) -> None:
        """Initializes the InteractionFeatures class.

        Args:
            config (DictConfig): Configuration for the feature. Expected to contain key-value pairs
                relevant to feature computation, such as thresholds or parameters. Must include
                'return_criterion' (str), which determines whether to return 'critical' or 'average'
                statistics for each feature.
        """
        super(InteractionFeatures, self).__init__(config)

        self.return_criterion = config.get("return_criterion", "critical")
        self.agent_i = interaction.InteractionAgent()
        self.agent_j = interaction.InteractionAgent()

    def compute(self, scenario: Scenario) -> ScenarioFeatures:
        """Computes pairwise interaction features for each agent pair in the scenario.

        Args:
            scenario (Scenario): Scenario object containing agent positions, velocities, headings,
                validity masks, timestamps, map conflict points, distances to conflict points,
                and other scenario-level information.

        Returns:
            ScenarioFeatures: An object containing computed interaction features for each valid agent pair,
                including separation, intersection, collision, minimum time to conflict point (mTTCP),
                interaction status, agent pair indices, and agent pair types.

        Raises:
            ValueError: If no agent combinations are found (i.e., less than two agents in the scenario).
        """
        agent_combinations = list(itertools.combinations(range(scenario.num_agents), 2))
        if len(agent_combinations) == 0:
            raise ValueError("No agent combinations found. Ensure that the scenario has at least two agents.")

        agent_types = scenario.agent_types
        agent_masks = scenario.agent_valid
        agent_positions = scenario.agent_positions
        agent_lengths = scenario.agent_lengths
        agent_widths = scenario.agent_widths
        agent_heights = scenario.agent_heights
        # NOTE: this is also computed as a feature in the individual features.
        agent_velocities = np.linalg.norm(scenario.agent_velocities, axis=-1) + EPS
        agent_headings = np.rad2deg(scenario.agent_headings)
        conflict_points = scenario.map_conflict_points
        dists_to_conflict_points = scenario.agent_distances_to_conflict_points

        # TODO: Figure out where's best place to get these from
        stationary_speed = scenario.stationary_speed
        agent_to_agent_max_distance = scenario.agent_to_agent_max_distance
        agent_to_conflict_point_max_distance = scenario.agent_to_conflict_point_max_distance
        agent_to_agent_distance_breach = scenario.agent_to_agent_distance_breach
        heading_threshold = scenario.heading_threshold

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
        scenario_thws = np.full(num_interactions, np.nan, dtype=np.float32)
        scenario_ttcs = np.full(num_interactions, np.nan, dtype=np.float32)
        scenario_dracs = np.full(num_interactions, np.nan, dtype=np.float32)

        # Compute distance to conflict points
        for n, (i, j) in enumerate(agent_combinations):
            self.agent_i.reset()
            self.agent_j.reset()

            # There should be at leat two valid timestaps for the combined agents masks
            mask_i, mask_j = agent_masks[i], agent_masks[j]
            mask = np.where(mask_i & mask_j)[0]
            if not mask.sum():
                # No valid data for this pair of agents
                scenario_interaction_statuses[n] = InteractionStatus.MASK_NOT_VALID
                continue

            self.agent_i.position, self.agent_j.position = agent_positions[i][mask], agent_positions[j][mask]
            self.agent_i.speed, self.agent_j.speed = agent_velocities[i][mask], agent_velocities[j][mask]
            self.agent_i.heading, self.agent_j.heading = agent_headings[i][mask], agent_headings[j][mask]
            self.agent_i.length, self.agent_j.length = agent_lengths[i][mask], agent_lengths[j][mask]
            self.agent_i.width, self.agent_j.width = agent_widths[i][mask], agent_widths[j][mask]
            self.agent_i.height, self.agent_j.height = agent_heights[i][mask], agent_heights[j][mask]

            self.agent_i.agent_type, self.agent_j.agent_type = agent_types[i], agent_types[j]
            self.agent_i.lane, self.agent_j.lane = None, None  # TODO: Add lane information if available

            # type: ignore
            if conflict_points is not None and dists_to_conflict_points is not None:
                self.agent_j.dists_to_conflict = dists_to_conflict_points[i][mask]
                self.agent_j.dists_to_conflict = dists_to_conflict_points[j][mask]

            # Check if agents are within a valid distance threshold to compute interactions
            separations = interaction.compute_separation(self.agent_i, self.agent_j)
            if not np.any(separations <= agent_to_agent_max_distance):
                scenario_interaction_statuses[n] = InteractionStatus.DISTANCE_TOO_FAR
                continue

            # Check if agents are stationary
            self.agent_i.stationary_speed = stationary_speed
            self.agent_j.stationary_speed = stationary_speed
            if self.agent_i.is_stationary and self.agent_i.is_stationary:
                scenario_interaction_statuses[n] = InteractionStatus.STATIONARY
                continue

            # Compute interaction features
            # separations = interaction.compute_separation(self.agent_i, self.agent_j)
            intersections = interaction.compute_intersections(self.agent_i, self.agent_j)
            collisions = (separations <= agent_to_agent_distance_breach) | intersections
            intersections = intersections.astype(np.float32)
            collisions = collisions.astype(np.float32)

            # Minimum time to conflict point (mTTCP) is calculated from t=0 to t=first time on of the agents cross that
            # point, aligned to what's done in ExiD: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9827305)
            mttcps = interaction.compute_mttcp(self.agent_i, self.agent_j, agent_to_conflict_point_max_distance)

            # To compute Time Headway (THW), Time to Collision (TTC), and Deceleration Rate to Avoid Collision (DRAC),
            # we currently assume that agents are sharing the same lane.
            valid_headings = interaction.find_valid_headings(self.agent_i, self.agent_j, heading_threshold)
            if valid_headings.shape[0] < 2:
                thw = np.full(1, np.inf, dtype=np.float32)
                ttc = np.full(1, np.inf, dtype=np.float32)
                drac = np.full(1, np.inf, dtype=np.float32)
                scenario_interaction_statuses[n] = InteractionStatus.PARTIAL_INVALID_HEADING
            else:
                # At this point agents are sharing a lane and have at least two steps with headings within the defined
                # threshold. TODO: check if steps are consecutive
                # Now we need to check if who is the leading agent within the interaction.
                leading_agent = interaction.find_leading_agent(self.agent_i, self.agent_j, valid_headings)

                # Now compute leader-follower interaction state
                thw = interaction.compute_thw(self.agent_i, self.agent_j, leading_agent, valid_headings)
                ttc = interaction.compute_ttc(self.agent_i, self.agent_j, leading_agent, valid_headings)
                drac = interaction.compute_drac(self.agent_i, self.agent_j, leading_agent, valid_headings)

                scenario_interaction_statuses[n] = InteractionStatus.COMPUTED_OK

            if self.return_criterion == "critical":
                separation = separations.min()
                intersection = intersections.sum()
                collision = collisions.sum()
                mttcp = mttcps.min()
                ttc = ttc.min()
                thw = thw.min()
                drac = drac.max()
            elif self.return_criterion == "average":
                # NOTE: whenever there are valid values within a trajectory, this return the mean over those values and
                # not the entire trajectory.
                separation = separations.mean()
                intersection = intersections.mean()
                collision = collisions.mean()
                mttcp = mttcps.mean()
                ttc = ttc.mean()
                thw = thw.mean()
                drac = drac.mean()
            else:
                raise ValueError(f"Criterion: {self.return_criterion} not supported. Expected 'critical' or 'average'.")

            # Store computed features in the state dictionary
            scenario_separations[n] = separation
            scenario_intersections[n] = intersection
            scenario_collisions[n] = collision
            scenario_mttcps[n] = mttcp
            scenario_thws[n] = thw
            scenario_ttcs[n] = ttc
            scenario_dracs[n] = drac

        return ScenarioFeatures(
            num_agents=scenario.num_agents,
            scenario_id=scenario.scenario_id,
            agent_types=scenario.agent_types,
            separation=scenario_separations,
            intersection=scenario_intersections,
            collision=scenario_collisions,
            mttcp=scenario_mttcps,
            thw=scenario_thws,
            ttc=scenario_ttcs,
            drac=scenario_dracs,
            interaction_status=scenario_interaction_statuses,
            interaction_agent_indices=scenario_agent_pair_indeces,
            interaction_agent_types=scenario_agents_pair_types,
        )
