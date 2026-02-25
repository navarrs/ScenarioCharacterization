import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from tqdm import tqdm

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
from characterization.utils.scenario_types import AgentPairType, AgentType, get_agent_pair_type

logger = get_logger(__name__)

# Global context for multiprocessing workers (populated via initializer)
_WORKER_CONTEXT: dict[str, Any] = {}


def _init_worker_context(
    agent_masks: NDArray[np.bool_],
    agent_positions: NDArray[np.float32],
    agent_velocities: NDArray[np.float32],
    agent_headings: NDArray[np.float32],
    agent_lengths: NDArray[np.float32],
    agent_widths: NDArray[np.float32],
    agent_heights: NDArray[np.float32],
    agent_types: list[AgentType],
    conflict_points: NDArray[np.float32] | None,
    dists_to_conflict_points: NDArray[np.float32] | None,
    stationary_speed: float,
    agent_to_agent_max_distance: float,
    agent_to_conflict_point_max_distance: float,
    agent_to_agent_distance_breach: float,
    heading_threshold: float,
    agent_max_deceleration: float,
    return_criterion: ReturnCriterion,
    categorize_features: bool,  # noqa: FBT001
    categorization_dicts: dict[str, dict[str, dict[str, Any]]] | None,
) -> None:
    """Initialize worker process with shared context data."""
    _WORKER_CONTEXT.update(
        {
            "agent_masks": agent_masks,
            "agent_positions": agent_positions,
            "agent_velocities": agent_velocities,
            "agent_headings": agent_headings,
            "agent_lengths": agent_lengths,
            "agent_widths": agent_widths,
            "agent_heights": agent_heights,
            "agent_types": agent_types,
            "conflict_points": conflict_points,
            "dists_to_conflict_points": dists_to_conflict_points,
            "stationary_speed": stationary_speed,
            "agent_to_agent_max_distance": agent_to_agent_max_distance,
            "agent_to_conflict_point_max_distance": agent_to_conflict_point_max_distance,
            "agent_to_agent_distance_breach": agent_to_agent_distance_breach,
            "heading_threshold": heading_threshold,
            "agent_max_deceleration": agent_max_deceleration,
            "return_criterion": return_criterion,
            "categorize_features": categorize_features,
            "categorization_dicts": categorization_dicts,
        }
    )


def _categorize_feature_value(
    value: float,
    feature_name: str,
    agent_pair_type: AgentPairType,
    categorization_dicts: dict[str, dict[str, dict[str, Any]]],
) -> float:
    """Categorize a feature value based on agent pair type and predefined percentiles.

    Args:
        value: The feature value to categorize.
        feature_name: The name of the feature being categorized.
        agent_pair_type: The type of the agent pair (VEHICLE_VEHICLE, VEHICLE_CYCLIST, etc.).
        categorization_dicts: Dictionary mapping agent pair types to their categorization thresholds.

    Returns:
        The categorized feature value based on percentiles, or -1.0 if categorization fails.
    """
    match agent_pair_type:
        case AgentPairType.TYPE_VEHICLE_VEHICLE:
            categories = categorization_dicts["vehicle_vehicle"].get(feature_name, None)
        case AgentPairType.TYPE_VEHICLE_PEDESTRIAN:
            categories = categorization_dicts["vehicle_pedestrian"].get(feature_name, None)
        case AgentPairType.TYPE_VEHICLE_CYCLIST:
            categories = categorization_dicts["vehicle_cyclist"].get(feature_name, None)
        case AgentPairType.TYPE_PEDESTRIAN_PEDESTRIAN:
            categories = categorization_dicts["pedestrian_pedestrian"].get(feature_name, None)
        case AgentPairType.TYPE_PEDESTRIAN_CYCLIST:
            categories = categorization_dicts["pedestrian_cyclist"].get(feature_name, None)
        case AgentPairType.TYPE_CYCLIST_CYCLIST:
            categories = categorization_dicts["cyclist_cyclist"].get(feature_name, None)
        case _:
            return -1.0

    if categories is None:
        return -1.0

    threshold_values = list(categories.values())
    return float(categorize_from_thresholds(value, threshold_values))


def _process_agent_pair_worker(n: int, i: int, j: int) -> tuple[int, InteractionStatus, dict[str, float] | None]:
    """Worker function to process a single agent pair using global context."""
    # Extract context (with fork, this is copy-on-write from parent)
    agent_masks = _WORKER_CONTEXT["agent_masks"]
    agent_positions = _WORKER_CONTEXT["agent_positions"]
    agent_velocities = _WORKER_CONTEXT["agent_velocities"]
    agent_headings = _WORKER_CONTEXT["agent_headings"]
    agent_lengths = _WORKER_CONTEXT["agent_lengths"]
    agent_widths = _WORKER_CONTEXT["agent_widths"]
    agent_heights = _WORKER_CONTEXT["agent_heights"]
    agent_types = _WORKER_CONTEXT["agent_types"]
    conflict_points = _WORKER_CONTEXT["conflict_points"]
    dists_to_conflict_points = _WORKER_CONTEXT["dists_to_conflict_points"]
    stationary_speed = _WORKER_CONTEXT["stationary_speed"]
    agent_to_agent_max_distance = _WORKER_CONTEXT["agent_to_agent_max_distance"]
    agent_to_conflict_point_max_distance = _WORKER_CONTEXT["agent_to_conflict_point_max_distance"]
    agent_to_agent_distance_breach = _WORKER_CONTEXT["agent_to_agent_distance_breach"]
    heading_threshold = _WORKER_CONTEXT["heading_threshold"]
    agent_max_deceleration = _WORKER_CONTEXT["agent_max_deceleration"]
    return_criterion = _WORKER_CONTEXT["return_criterion"]
    categorize_features = _WORKER_CONTEXT["categorize_features"]
    categorization_dicts = _WORKER_CONTEXT["categorization_dicts"]

    agent_i = interaction.InteractionAgent()
    agent_j = interaction.InteractionAgent()

    # There should be at least two valid timestamps for the combined agents masks
    mask_i, mask_j = agent_masks[i], agent_masks[j]
    mask = np.where(mask_i & mask_j)[0]
    if not mask.sum():
        # No valid data for this pair of agents
        return (n, InteractionStatus.MASK_NOT_VALID, None)

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
        agent_i.dists_to_conflict = dists_to_conflict_points[i][mask]
        agent_j.dists_to_conflict = dists_to_conflict_points[j][mask]

    # Check if agents are within a valid distance threshold to compute interactions
    separations = interaction.compute_separation(agent_i, agent_j)
    if not np.any(separations <= agent_to_agent_max_distance):
        return (n, InteractionStatus.DISTANCE_TOO_FAR, None)

    # Check if agents are stationary
    agent_i.stationary_speed = stationary_speed
    agent_j.stationary_speed = stationary_speed
    if agent_i.is_stationary and agent_j.is_stationary:
        return (n, InteractionStatus.STATIONARY, None)

    # Compute interaction features
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
        status = InteractionStatus.PARTIAL_INVALID_HEADING
    else:
        # At this point agents are sharing a lane and have at least two steps with headings within the defined
        # threshold. TODO: check if steps are consecutive
        # Now we need to check if who is the leading agent within the interaction.
        leading_agent = interaction.find_leading_agent(agent_i, agent_j, valid_headings)

        # Now compute leader-follower interaction state
        thws = interaction.compute_thw(agent_i, agent_j, leading_agent, valid_headings)
        ttcs = interaction.compute_ttc(agent_i, agent_j, leading_agent, valid_headings)
        dracs = interaction.compute_drac(agent_i, agent_j, leading_agent, valid_headings, agent_max_deceleration)
        status = InteractionStatus.COMPUTED_OK

    match return_criterion:
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
            error_message = f"{return_criterion} not supported. Expected 'critical' or 'average'."
            raise ValueError(error_message)

    # TODO: add the stability cap to configuration
    inv_mttcp = min(1.0 / (mttcp + SMALL_EPS), 10.0)
    inv_ttc = min(1.0 / (ttc + SMALL_EPS), 10.0)
    inv_thw = min(1.0 / (thw + SMALL_EPS), 10.0)

    if categorize_features and categorization_dicts is not None:
        agent_pair_type = get_agent_pair_type(agent_i.agent_type, agent_j.agent_type)

        separation = _categorize_feature_value(separation, "separation", agent_pair_type, categorization_dicts)
        intersection = _categorize_feature_value(intersection, "intersection", agent_pair_type, categorization_dicts)
        collision = _categorize_feature_value(collision, "collision", agent_pair_type, categorization_dicts)
        mttcp = _categorize_feature_value(mttcp, "mttcp", agent_pair_type, categorization_dicts)
        inv_mttcp = _categorize_feature_value(inv_mttcp, "inv_mttcp", agent_pair_type, categorization_dicts)
        thw = _categorize_feature_value(thw, "thw", agent_pair_type, categorization_dicts)
        inv_thw = _categorize_feature_value(inv_thw, "inv_thw", agent_pair_type, categorization_dicts)
        ttc = _categorize_feature_value(ttc, "ttc", agent_pair_type, categorization_dicts)
        inv_ttc = _categorize_feature_value(inv_ttc, "inv_ttc", agent_pair_type, categorization_dicts)
        drac = _categorize_feature_value(drac, "drac", agent_pair_type, categorization_dicts)

    # use a simple dict for speed
    return (
        n,
        status,
        {
            "separation": separation,
            "intersection": intersection,
            "collision": collision,
            "mttcp": mttcp,
            "inv_mttcp": inv_mttcp,
            "thw": thw,
            "inv_thw": inv_thw,
            "ttc": ttc,
            "inv_ttc": inv_ttc,
            "drac": drac,
        },
    )


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

        # Prepare categorization dicts for worker processes
        categorization_dicts = None
        if self.categorize_features:
            categorization_dicts = {
                "vehicle_vehicle": self.vehicle_vehicle_categories,
                "vehicle_pedestrian": self.vehicle_pedestrian_categories,
                "vehicle_cyclist": self.vehicle_cyclist_categories,
                "pedestrian_pedestrian": self.pedestrian_pedestrian_categories,
                "pedestrian_cyclist": self.pedestrian_cyclist_categories,
                "cyclist_cyclist": self.cyclist_cyclist_categories,
            }

        # Process agent combinations in parallel with fork context for zero-copy data sharing
        with ProcessPoolExecutor(
            mp_context=mp.get_context("fork"),  # faster than spawn
            initializer=_init_worker_context,  # read-only
            initargs=(
                agent_masks,
                agent_positions,
                agent_velocities,
                agent_headings,
                agent_lengths,
                agent_widths,
                agent_heights,
                agent_types,
                conflict_points,
                dists_to_conflict_points,
                stationary_speed,
                agent_to_agent_max_distance,
                agent_to_conflict_point_max_distance,
                agent_to_agent_distance_breach,
                heading_threshold,
                agent_max_deceleration,
                self.return_criterion,
                self.categorize_features,
                categorization_dicts,
            ),
        ) as executor:
            # Submit all tasks (only passing indices now)
            futures = [
                executor.submit(_process_agent_pair_worker, n, i, j) for n, (i, j) in enumerate(agent_combinations)
            ]

            # Process results as they complete with tqdm progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing pairwise interactions"):
                n, status, results = future.result()
                scenario_interaction_statuses[n] = status

                if results is not None:
                    scenario_separations[n] = results["separation"]
                    scenario_intersections[n] = results["intersection"]
                    scenario_collisions[n] = results["collision"]
                    scenario_mttcps[n] = results["mttcp"]
                    scenario_inv_mttcps[n] = results["inv_mttcp"]
                    scenario_thws[n] = results["thw"]
                    scenario_inv_thws[n] = results["inv_thw"]
                    scenario_ttcs[n] = results["ttc"]
                    scenario_inv_ttcs[n] = results["inv_ttc"]
                    scenario_dracs[n] = results["drac"]

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
