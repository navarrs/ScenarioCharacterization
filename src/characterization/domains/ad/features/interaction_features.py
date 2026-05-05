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
from characterization.domains.ad.features.base_feature import ADBaseFeature
from characterization.domains.ad.features.interaction_utils import compute_intersections_per_timestep
from characterization.domains.ad.scenario_types import (
    AgentPairType,
    AgentTrajectoryMasker,
    AgentType,
    get_agent_pair_type,
)
from characterization.domains.ad.schemas import InteractionPairFeatures, Scenario, ScenarioFeatures
from characterization.utils.common import (
    FeatureType,
    InteractionStatus,
    ReturnCriterion,
    categorize_from_thresholds,
    return_by_criterion,
)
from characterization.utils.constants import EPSILON, MIN_VALID_POINTS
from characterization.utils.geometric_utils import compute_agent_to_agent_closest_dists
from characterization.utils.logging_utils import get_pylogger

logger = get_pylogger(__name__)


def _chunked_prefilter_candidates(
    agent_masks: NDArray[np.bool_],
    agent_positions: NDArray[np.float32],
    agent_to_agent_max_distance: float,
    chunk_size: int = 256,
) -> NDArray[np.bool_]:
    """Return a boolean mask of agent pairs that are candidate interactions.

    This is the exact equivalent of the per-pair distance check inside the worker:
    ``np.any(||pos_i(t) - pos_j(t)|| <= threshold for shared valid t)``.
    A pair is marked ``True`` (candidate) when its minimum pairwise distance
    across all shared valid timesteps is within the threshold.  Pairs that are
    always beyond the threshold would receive ``InteractionStatus.DISTANCE_TOO_FAR``
    and are marked ``False``.

    The naive broadcast ``(n, n, T, 2)`` would require more memory.
    To keep the memory footprint bounded, timesteps are processed in chunks.

    Why not AABB?
    An Axis-Aligned Bounding Box filter checks whether
    the trajectory extents of two agents overlap spatially,
    but is blind to when each agent occupies a region.
    This filter is the exact temporal equivalent of the existing check
    and produces zero false positives.

    Why not a KD-tree?
    A per-timestep KD-tree approach would need a for loop over all timesteps,
    each building or querying a spatial index.
    The chunked NumPy approach replaces that with
    `T // chunk_size` fully vectorised iterations,
    which should be faster.

    Args:
        agent_masks: Boolean validity mask of shape ``(n_agents, T)``.
        agent_positions: Agent XYZ positions of shape ``(n_agents, T, 3)``.
            Only the XY components are used.
        agent_to_agent_max_distance: Distance threshold in metres.
        chunk_size: Number of timesteps to process per iteration.  Controls the
            memory/iteration-count trade-off.  Defaults to 256.

    Returns:
        Boolean array of shape ``(num_pairs,)`` where ``num_pairs = n*(n-1)/2``.
        Element ``k`` is ``True`` when pair ``k``
        (ordered as ``np.triu_indices(n, k=1)``,
        equivalently ``itertools.combinations(range(n), 2)``)
        comes within the distance threshold at some shared valid timestep.

    """
    n_agents: int = agent_positions.shape[0]
    num_timesteps: int = agent_positions.shape[1]
    pair_rows: NDArray[np.intp]
    pair_cols: NDArray[np.intp]
    pair_rows, pair_cols = np.triu_indices(n_agents, k=1)
    num_pairs: int = pair_rows.shape[0]

    # Replace invalid timesteps with inf so they never contribute to a minimum distance.
    pos_xy: NDArray[np.float32] = np.where(
        agent_masks[:, :, None],
        agent_positions[:, :, :2],
        np.inf,
    )  # (n_agents, T, 2)

    # Running minimum pairwise distance across all timesteps, initialised to inf.
    min_dists: NDArray[np.float32] = np.full(num_pairs, np.inf, dtype=np.float32)  # (num_pairs,)

    for t0 in tqdm(range(0, num_timesteps, chunk_size), desc="Distance pre-filter", unit="chunk"):
        chunk: NDArray[np.float32] = pos_xy[:, t0 : t0 + chunk_size, :]  # (n_agents, chunk, 2)
        # inf - inf = nan when both agents are invalid at the same timestep.
        # This is expected and intentional:
        # the nan is replaced with inf immediately afterwards
        # so those timesteps never contribute to the running minimum.
        # Suppress the numpy RuntimeWarning that would otherwise fire here.
        with np.errstate(invalid="ignore"):
            diff: NDArray[np.float32] = chunk[pair_rows] - chunk[pair_cols]  # (num_pairs, chunk, 2)
        dists: NDArray[np.float32] = np.linalg.norm(diff, axis=-1)  # (num_pairs, chunk)
        np.nan_to_num(dists, copy=False, nan=np.inf)
        min_dists = np.minimum(min_dists, dists.min(axis=1))  # (num_pairs,)

    return min_dists <= agent_to_agent_max_distance  # (num_pairs,)


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
    stationary_speed: float,
    agent_to_agent_max_distance: float,
    agent_to_conflict_point_max_distance: float,
    agent_to_agent_distance_breach: float,
    heading_threshold: float,
    agent_max_deceleration: float,
    return_criterion: ReturnCriterion,
    categorize_features: bool,  # noqa: FBT001
    categorization_dicts: dict[str, dict[str, dict[str, Any]]] | None,
    inv_stability_cap: float,
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
            "stationary_speed": stationary_speed,
            "agent_to_agent_max_distance": agent_to_agent_max_distance,
            "agent_to_conflict_point_max_distance": agent_to_conflict_point_max_distance,
            "agent_to_agent_distance_breach": agent_to_agent_distance_breach,
            "heading_threshold": heading_threshold,
            "agent_max_deceleration": agent_max_deceleration,
            "return_criterion": return_criterion,
            "categorize_features": categorize_features,
            "categorization_dicts": categorization_dicts,
            "inv_stability_cap": inv_stability_cap,
        },
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
    agent_types = _WORKER_CONTEXT["agent_types"]
    conflict_points = _WORKER_CONTEXT["conflict_points"]
    stationary_speed = _WORKER_CONTEXT["stationary_speed"]
    agent_to_agent_max_distance = _WORKER_CONTEXT["agent_to_agent_max_distance"]
    agent_to_conflict_point_max_distance = _WORKER_CONTEXT["agent_to_conflict_point_max_distance"]
    agent_to_agent_distance_breach = _WORKER_CONTEXT["agent_to_agent_distance_breach"]
    heading_threshold = _WORKER_CONTEXT["heading_threshold"]
    agent_max_deceleration = _WORKER_CONTEXT["agent_max_deceleration"]
    return_criterion = _WORKER_CONTEXT["return_criterion"]
    categorize_features = _WORKER_CONTEXT["categorize_features"]
    categorization_dicts = _WORKER_CONTEXT["categorization_dicts"]
    inv_stability_cap = _WORKER_CONTEXT["inv_stability_cap"]

    mask_i, mask_j = agent_masks[i], agent_masks[j]
    joint_valid = mask_i & mask_j  # (T,) bool
    if joint_valid.sum() < MIN_VALID_POINTS:
        return (n, InteractionStatus.MASK_NOT_VALID, None)

    pos_i = agent_positions[i]  # (T, 3)
    pos_j = agent_positions[j]  # (T, 3)
    spd_i = agent_velocities[i]  # (T,)
    spd_j = agent_velocities[j]  # (T,)
    hdg_i = agent_headings[i]  # (T,) radians
    hdg_j = agent_headings[j]  # (T,) radians

    # Distance check
    separations = interaction.compute_separation_1d(pos_i, pos_j, joint_valid)  # (N_valid,)
    if not np.any(separations <= agent_to_agent_max_distance):
        return (n, InteractionStatus.DISTANCE_TOO_FAR, None)

    # Stationarity check
    if spd_i[joint_valid].mean() < stationary_speed and spd_j[joint_valid].mean() < stationary_speed:
        return (n, InteractionStatus.STATIONARY, None)

    # Per-timestep intersection and collision (N_valid,)
    intersections_t = compute_intersections_per_timestep(pos_i, pos_j, joint_valid)  # (T,) bool
    valid_intersect = intersections_t[joint_valid]  # (N_valid,) bool
    collisions_arr = ((separations <= agent_to_agent_distance_breach) | valid_intersect).astype(np.float32)
    intersections_arr = valid_intersect.astype(np.float32)

    # mTTCP using map-defined conflict points; nan when none available
    conflict_pts = conflict_points if conflict_points is not None else np.empty((0, 3), dtype=np.float32)
    mttcp = float(
        interaction.compute_mttcp(
            pos_i,
            pos_j,
            spd_i,
            spd_j,
            conflict_pts,
            joint_valid,
            agent_to_conflict_point_max_distance,
            return_criterion,
        )
    )

    # THW — heading check is internal; returns inf when no co-directional pair found
    thw = float(
        interaction.compute_thw(
            pos_i,
            pos_j,
            spd_i,
            spd_j,
            hdg_i,
            hdg_j,
            joint_valid,
            heading_threshold,
            return_criterion,
        )
    )

    # CPA geometry gate for TTC/DRAC; no vertical gate needed for ground-plane agents
    joint_valid_gated = interaction.compute_pair_gate(
        pos_i,
        pos_j,
        spd_i,
        spd_j,
        hdg_i,
        hdg_j,
        joint_valid,
        max_cpa_dist=agent_to_agent_distance_breach,
        max_vertical_separation=float("inf"),
    )
    ttc = float(
        interaction.compute_ttc(
            pos_i,
            pos_j,
            spd_i,
            spd_j,
            hdg_i,
            hdg_j,
            joint_valid_gated,
            return_criterion,
        )
    )
    drac = float(
        interaction.compute_drac(
            pos_i,
            pos_j,
            spd_i,
            spd_j,
            hdg_i,
            hdg_j,
            joint_valid_gated,
            agent_max_deceleration,
            return_criterion,
        )
    )

    # Aggregate per-timestep arrays; scalar metrics are already aggregated by criterion
    separation = float(return_by_criterion(separations, return_criterion, critical_is_min=True))
    intersection = float(return_by_criterion(intersections_arr, return_criterion))
    collision = float(return_by_criterion(collisions_arr, return_criterion))

    # Inverse metrics — guard against nan/inf
    inv_mttcp = min(1.0 / (mttcp + EPSILON), inv_stability_cap) if np.isfinite(mttcp) else float("nan")
    inv_ttc = min(1.0 / (ttc + EPSILON), inv_stability_cap) if np.isfinite(ttc) else float("nan")
    inv_thw = min(1.0 / (thw + EPSILON), inv_stability_cap) if np.isfinite(thw) else float("nan")

    status = InteractionStatus.COMPUTED_OK

    if categorize_features and categorization_dicts is not None:
        agent_pair_type = get_agent_pair_type(agent_types[i], agent_types[j])

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


class InteractionFeatures(ADBaseFeature):
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
        self.inv_stability_cap: float = self.config.get("inv_stability_cap", 10.0)
        if self.categorize_features:
            vehicle_vehicle_file = Path(self.config.get("vehicle_vehicle_categorization_file", ""))
            if not vehicle_vehicle_file.is_file():
                msg = f"Categorization file {vehicle_vehicle_file} does not exist."
                raise FileNotFoundError(msg)
            with vehicle_vehicle_file.open("r") as f:
                self.vehicle_vehicle_categories = json.load(f)

            vehicle_pedestrian_file = Path(self.config.get("vehicle_pedestrian_categorization_file", ""))
            if not vehicle_pedestrian_file.is_file():
                msg = f"Categorization file {vehicle_pedestrian_file} does not exist."
                raise FileNotFoundError(msg)
            with vehicle_pedestrian_file.open("r") as f:
                self.vehicle_pedestrian_categories = json.load(f)

            vehicle_cyclist_file = Path(self.config.get("vehicle_cyclist_categorization_file", ""))
            if not vehicle_cyclist_file.is_file():
                msg = f"Categorization file {vehicle_cyclist_file} does not exist."
                raise FileNotFoundError(msg)
            with vehicle_cyclist_file.open("r") as f:
                self.vehicle_cyclist_categories = json.load(f)

            pedestrian_pedestrian_file = Path(self.config.get("pedestrian_pedestrian_categorization_file", ""))
            if not pedestrian_pedestrian_file.is_file():
                msg = f"Categorization file {pedestrian_pedestrian_file} does not exist."
                raise FileNotFoundError(msg)
            with pedestrian_pedestrian_file.open("r") as f:
                self.pedestrian_pedestrian_categories = json.load(f)

            pedestrian_cyclist_file = Path(self.config.get("pedestrian_cyclist_categorization_file", ""))
            if not pedestrian_cyclist_file.is_file():
                msg = f"Categorization file {pedestrian_cyclist_file} does not exist."
                raise FileNotFoundError(msg)
            with pedestrian_cyclist_file.open("r") as f:
                self.pedestrian_cyclist_categories = json.load(f)

            cyclist_cyclist_file = Path(self.config.get("cyclist_cyclist_categorization_file", ""))
            if not cyclist_cyclist_file.is_file():
                msg = f"Categorization file {cyclist_cyclist_file} does not exist."
                raise FileNotFoundError(msg)
            with cyclist_cyclist_file.open("r") as f:
                self.cyclist_cyclist_categories = json.load(f)

    def compute_interaction_features(
        self,
        scenario: Scenario,
        *,
        max_workers: int | None = None,
    ) -> list[InteractionPairFeatures] | None:
        """Compute pairwise interaction features for all candidate agent pairs.

        Args:
            scenario (Scenario): Complete scenario data containing:
                - agent_data: Agent positions, velocities, headings, dimensions, validity masks, and types
                - metadata: Timestamps, distance thresholds, speed limits, and interaction parameters
                - static_map_data: Map conflict points and agent distances to conflict points
            max_workers: Maximum number of worker processes for parallel computation.
                Defaults to None, which uses the number of processors on the machine.

        Returns:
            List of :class:`InteractionPairFeatures`, one entry per successfully computed pair
            (pairs with ``COMPUTED_OK`` or ``PARTIAL_INVALID_HEADING`` status). Returns ``None``
            if the scenario has fewer than 2 agents.

        Note:
            - Agent pairs must have overlapping valid timesteps to be processed.
            - Stationary pairs and pairs beyond the distance threshold are omitted from the list.
            - Leader-follower metrics (THW, TTC, DRAC) require agents with similar headings.
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
        agent_velocities = np.linalg.norm(agent_trajectories.agent_xy_vel, axis=-1) + EPSILON
        agent_headings = agent_trajectories.agent_headings.squeeze(-1)  # (N, T) radians
        conflict_points = map_data.map_conflict_points if map_data is not None else None

        # Meta information
        stationary_speed = metadata.max_stationary_speed
        agent_to_agent_max_distance = metadata.agent_to_agent_max_distance
        agent_to_conflict_point_max_distance = metadata.agent_to_conflict_point_max_distance
        agent_to_agent_distance_breach = metadata.agent_to_agent_distance_breach
        heading_threshold = metadata.heading_threshold
        agent_max_deceleration = metadata.agent_max_deceleration

        # Per-pair status tracking (for the distance pre-filter and worker results)
        scenario_interaction_statuses = [InteractionStatus.UNKNOWN for _ in agent_combinations]
        # Collected results for pairs that completed successfully
        pair_results: dict[int, dict[str, float]] = {}

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

        # Pre-filter: compute exact pairwise minimum distances
        # before invoking the pool.
        # Pairs that are never within range are marked DISTANCE_TOO_FAR here
        # and never submitted to a worker,
        # avoiding their IPC and scheduling overhead.
        is_candidate: NDArray[np.bool_] = _chunked_prefilter_candidates(
            agent_masks,
            agent_positions,
            agent_to_agent_max_distance,
        )  # (num_pairs,)

        candidate_ns: NDArray[np.intp] = np.where(is_candidate)[0]
        for i in np.where(~is_candidate)[0]:
            scenario_interaction_statuses[int(i)] = InteractionStatus.DISTANCE_TOO_FAR

        num_combinations = len(agent_combinations)
        logger.info(
            "Distance pre-filter: %d / %d pairs left (%.1f%% skipped)",
            len(candidate_ns),
            num_combinations,
            100.0 * (~is_candidate).mean(),
        )

        def _run_pairwise_full_loop(pairs: list[tuple[int, int, int]]) -> None:
            for n, i, j in tqdm(pairs, desc="Computing pairwise interactions (full loop)"):
                pair_n, status, results = _process_agent_pair_worker(n, i, j)
                scenario_interaction_statuses[pair_n] = status
                if results is not None:
                    pair_results[pair_n] = results

        candidate_pairs = [(int(n), agent_combinations[n][0], agent_combinations[n][1]) for n in candidate_ns]

        # max_workers=None means "use all CPUs" and goes to the pool path.
        # When max_workers=1 (or 0), skip subprocess creation entirely: run in-process to
        # avoid fork overhead and the CoW page faults caused by Python refcount updates.
        if max_workers is None or max_workers > 1:
            # Process agent combinations in parallel with fork context for zero-copy data sharing
            with ProcessPoolExecutor(
                max_workers=max_workers,
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
                    stationary_speed,
                    agent_to_agent_max_distance,
                    agent_to_conflict_point_max_distance,
                    agent_to_agent_distance_breach,
                    heading_threshold,
                    agent_max_deceleration,
                    self.return_criterion,
                    self.categorize_features,
                    categorization_dicts,
                    self.inv_stability_cap,
                ),
            ) as executor:
                futures = [executor.submit(_process_agent_pair_worker, n, i, j) for n, i, j in candidate_pairs]

                # Process results as they complete with tqdm progress bar
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Computing pairwise interactions (pool with {max_workers=})",
                ):
                    n, status, results = future.result()
                    scenario_interaction_statuses[n] = status

                    if results is not None:
                        pair_results[n] = results
        else:
            _init_worker_context(
                agent_masks,
                agent_positions,
                agent_velocities,
                agent_headings,
                agent_lengths,
                agent_widths,
                agent_heights,
                agent_types,
                conflict_points,
                stationary_speed,
                agent_to_agent_max_distance,
                agent_to_conflict_point_max_distance,
                agent_to_agent_distance_breach,
                heading_threshold,
                agent_max_deceleration,
                self.return_criterion,
                self.categorize_features,
                categorization_dicts,
                self.inv_stability_cap,
            )
            _run_pairwise_full_loop(candidate_pairs)

        _ok_statuses = {InteractionStatus.COMPUTED_OK, InteractionStatus.PARTIAL_INVALID_HEADING}
        output: list[InteractionPairFeatures] = []
        for pair_n, (i, j) in enumerate(agent_combinations):
            if scenario_interaction_statuses[pair_n] not in _ok_statuses:
                continue
            r = pair_results.get(pair_n)
            if r is None:
                continue
            type_a, type_b = agent_types[i], agent_types[j]
            output.append(
                InteractionPairFeatures(
                    agent_id_a=i,
                    agent_id_b=j,
                    pair_type=f"{type_a.name}_{type_b.name}",
                    separation=r["separation"] if np.isfinite(r["separation"]) else None,
                    intersection=r["intersection"] if np.isfinite(r["intersection"]) else None,
                    collision=r["collision"] if np.isfinite(r["collision"]) else None,
                    mttcp=r["mttcp"] if np.isfinite(r["mttcp"]) else None,
                    thw=r["thw"] if np.isfinite(r["thw"]) else None,
                    ttc=r["ttc"] if np.isfinite(r["ttc"]) else None,
                    drac=r["drac"] if np.isfinite(r["drac"]) else None,
                )
            )
        return output

    def compute(
        self,
        scenario: Scenario,
        *,
        max_workers: int | None = None,
    ) -> ScenarioFeatures:
        """Compute scenario features focused on agent-to-agent interactions.

        Args:
            scenario (Scenario): Complete scenario data containing:
                - agent_data: Agent trajectories, dimensions, headings, and validity information
                - metadata: Scenario parameters including distance thresholds, speed limits,
                  and interaction-specific configuration values
                - static_map_data: Map conflict points and precomputed distances for mTTCP analysis
            max_workers (int | None): Maximum number of worker processes for parallel computation.
                Defaults to None, which uses the number of processors on the machine.

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
            interaction_features=self.compute_interaction_features(scenario, max_workers=max_workers),
            agent_to_agent_closest_dists=agent_to_agent_closest_dists,
        )
