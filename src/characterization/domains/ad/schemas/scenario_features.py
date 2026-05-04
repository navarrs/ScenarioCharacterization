import numpy as np
from pydantic import BaseModel

from characterization.domains.ad.scenario_types import AgentType
from characterization.schemas.scenario_features import (
    BaseFeatureDetections,
    BaseFeatureWeights,
    BaseIndividualAgentFeatures,
    BaseInteractionPairFeatures,
)
from characterization.schemas.types import Float32NDArray1D, Float32NDArray2D, Int32NDArray1D
from characterization.utils.common import InteractionStatus, TrajectoryType
from characterization.utils.constants import EPSILON

from .scenario import ScenarioMetadata


class FeatureDetections(BaseFeatureDetections):
    """Detection thresholds for autonomous driving feature computation.

    Extends :class:`BaseFeatureDetections` with AD-specific thresholds.

    Attributes:
        speed: Speed threshold in m/s.
        speed_limit_diff: Difference between agent speed and speed limit (m/s).
        acceleration: Acceleration threshold (m/s²).
        deceleration: Deceleration threshold (m/s²).
        jerk: Jerk threshold (m/s³).
        waiting_period: Waiting period at traffic signals (s).
        kalman_difficulty: Kalman-filter prediction difficulty threshold.
        mttcp: Minimum time to conflict point (s).
        thw: Time headway threshold (s).
        ttc: Time-to-collision threshold (s).
        drac: Deceleration rate to avoid collision (m/s²).
        collision: Collision distance threshold (m).
    """

    # NOTE: speed threshold is from Proven Safety Countermeasures:
    # https://highways.dot.gov/sites/fhwa.dot.gov/files/App%20Speed%20Limits_508.pdf
    speed: float = 13.0  # m/s (~47 km/h / ~30 mph)
    speed_limit_diff: float = 5.0  # m/s (~10 mph over limit)
    # NOTE: from https://arxiv.org/pdf/2202.07438 (Table 3)
    acceleration: float = 10.0  # m/s²
    # NOTE: from https://copradar.com/chapts/references/acceleration.html
    deceleration: float = 5.0  # m/s²
    # NOTE: from https://mpmanser.com/wp-content/uploads/2020/04/Feng-Manser_2017.pdf
    jerk: float = 1.5  # m/s³
    # NOTE: reaction time beyond 4 s considered slow (countdown timer study)
    waiting_period: float = 4.0  # s
    # NOTE: from UniTraj https://arxiv.org/pdf/2403.15098
    kalman_difficulty: float = 50.0
    # NOTE: from https://arxiv.org/pdf/2202.07438 (Table 3, k=1)
    mttcp: float = 4.0  # s
    thw: float = 2.0  # s
    ttc: float = 2.0  # s
    drac: float = 3.0  # m/s²
    collision: float = 1.0


class FeatureWeights(BaseFeatureWeights):
    """Feature weights for autonomous driving score computation.

    Extends :class:`BaseFeatureWeights` with AD-specific weight fields.
    """

    speed: float = 1.0
    speed_limit_diff: float = 1.0
    acceleration: float = 1.0
    deceleration: float = 1.0
    jerk: float = 0.1
    waiting_period: float = 1.0
    kalman_difficulty: float = 1.0
    trajectory_type: float = 1.0
    collision: float = 1.0
    mttcp: float = 1.0
    thw: float = 1.0
    ttc: float = 1.0
    drac: float = 1.0


class Individual(BaseModel):
    """Individual kinematic features per agent in an AD scenario.

    Attributes:
        valid_idxs: Indices of agents with valid data.
        agent_types: Type of each agent.
        agent_trajectory_types: Trajectory classification for each agent.
        speed: Per-agent speed values (m/s).
        speed_limit_diff: Per-agent speed-limit difference (m/s).
        acceleration: Per-agent acceleration (m/s²).
        deceleration: Per-agent deceleration (m/s²).
        jerk: Per-agent jerk (m/s³).
        waiting_period: Per-agent waiting period at signals (s).
        kalman_difficulty: Per-agent Kalman-filter prediction difficulty.
    """

    valid_idxs: Int32NDArray1D | None = None
    agent_types: list[AgentType] | None = None
    agent_trajectory_types: list[TrajectoryType]

    speed: Float32NDArray1D | None = None
    speed_limit_diff: Float32NDArray1D | None = None
    acceleration: Float32NDArray1D | None = None
    deceleration: Float32NDArray1D | None = None
    jerk: Float32NDArray1D | None = None
    waiting_period: Float32NDArray1D | None = None
    kalman_difficulty: Float32NDArray1D | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}


class Interaction(BaseModel):
    """Pairwise interaction features for agent pairs in an AD scenario.

    Attributes:
        separation: Separation distance between agent pairs (m).
        intersection: Intersection distance between agent pairs (m).
        collision: Collision indicator per pair.
        mttcp: Minimum time to conflict point per pair (s).
        inv_mttcp: Inverse of mttcp.
        thw: Time headway per pair (s).
        inv_thw: Inverse of thw.
        ttc: Time to collision per pair (s).
        inv_ttc: Inverse of ttc.
        drac: Deceleration rate to avoid collision per pair (m/s²).
        interaction_status: Status of each interaction.
        interaction_agent_indices: Pairs of interacting agent indices.
        interaction_agent_types: Agent type pairs.
    """

    separation: Float32NDArray1D | None = None
    intersection: Float32NDArray1D | None = None
    collision: Float32NDArray1D | None = None
    mttcp: Float32NDArray1D | None = None
    inv_mttcp: Float32NDArray1D | None = None
    thw: Float32NDArray1D | None = None
    inv_thw: Float32NDArray1D | None = None
    ttc: Float32NDArray1D | None = None
    inv_ttc: Float32NDArray1D | None = None
    drac: Float32NDArray1D | None = None

    interaction_status: list[InteractionStatus] | None = None
    interaction_agent_indices: list[tuple[int, int]] | None = None
    interaction_agent_types: list[tuple[AgentType, AgentType]] | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}


class IndividualAgentFeatures(BaseIndividualAgentFeatures):
    """Kinematic features for a single agent in an AD scenario.

    Inherits all common fields from :class:`BaseIndividualAgentFeatures`, including the
    AD-specific ``speed_limit_diff`` and ``jerk`` fields.
    """


class InteractionPairFeatures(BaseInteractionPairFeatures):
    """Pairwise interaction features between two agents in an AD scenario.

    Inherits all common fields from :class:`BaseInteractionPairFeatures`, including the
    AD-specific ``separation``, ``intersection``, and ``collision`` fields.
    """


class ScenarioFeatures(BaseModel):
    """All extracted features for an AD scenario.

    Attributes:
        metadata: Scenario metadata.
        individual_features: Per-agent kinematic features, one entry per valid agent.
        interaction_features: Pairwise interaction features for candidate agent pairs.
        agent_to_agent_closest_dists: Shape (N, N) closest distances between agent pairs.
    """

    metadata: ScenarioMetadata
    individual_features: list[IndividualAgentFeatures] | None = None
    interaction_features: list[InteractionPairFeatures] | None = None
    agent_to_agent_closest_dists: Float32NDArray2D | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}


def to_individual_array(features: list[IndividualAgentFeatures]) -> Individual:
    """Convert list-based individual features to the legacy array format.

    Args:
        features: List of per-agent feature objects produced by the AD feature extractor.

    Returns:
        :class:`Individual` with all fields populated as NumPy arrays.
    """
    valid_idxs = np.array([f.agent_id for f in features], dtype=np.int32)
    agent_types: list[AgentType] = [AgentType[f.agent_type] for f in features]
    traj_types: list[TrajectoryType] = [
        TrajectoryType[f.trajectory_type] if f.trajectory_type else TrajectoryType.STATIONARY for f in features
    ]
    n = len(features)

    def _arr(values: list[float | None]) -> Float32NDArray1D | None:
        if not values or all(v is None for v in values):
            return None
        return np.array([v if v is not None else np.nan for v in values], dtype=np.float32)

    return Individual(
        valid_idxs=valid_idxs if n > 0 else None,
        agent_types=agent_types if n > 0 else None,
        agent_trajectory_types=traj_types,
        speed=_arr([f.speed for f in features]),
        speed_limit_diff=_arr([f.speed_limit_diff for f in features]),
        acceleration=_arr([f.acceleration for f in features]),
        deceleration=_arr([f.deceleration for f in features]),
        jerk=_arr([f.jerk for f in features]),
        waiting_period=_arr([f.waiting_period for f in features]),
        kalman_difficulty=_arr([f.kalman_difficulty for f in features]),
    )


def to_interaction_array(features: list[InteractionPairFeatures]) -> Interaction:
    """Convert list-based interaction features to the legacy array format.

    Only pairs present in the list (i.e. those with a computable result) are represented;
    the returned :class:`Interaction` does not include non-candidate pairs.

    Args:
        features: List of per-pair feature objects produced by the AD feature extractor.

    Returns:
        :class:`Interaction` with all fields populated as NumPy arrays.
    """
    n = len(features)

    def _arr(values: list[float | None]) -> Float32NDArray1D | None:
        if not values:
            return None
        return np.array([v if v is not None else np.nan for v in values], dtype=np.float32)

    inv_cap = 10.0

    def _inv(val: float | None) -> float:
        if val is None or not np.isfinite(val):
            return np.nan
        return min(1.0 / (val + EPSILON), inv_cap)

    statuses = [InteractionStatus.COMPUTED_OK] * n
    indices = [(f.agent_id_a, f.agent_id_b) for f in features]
    pair_types = [(AgentType[t] for t in f.pair_type.split("_", 1)) for f in features]
    agent_type_pairs: list[tuple[AgentType, AgentType]] = []
    for f in features:
        a_str, b_str = f.pair_type.split("_", 1)
        agent_type_pairs.append((AgentType[a_str], AgentType[b_str]))

    return Interaction(
        separation=_arr([f.separation for f in features]),
        intersection=_arr([f.intersection for f in features]),
        collision=_arr([f.collision for f in features]),
        mttcp=_arr([f.mttcp for f in features]),
        inv_mttcp=_arr([_inv(f.mttcp) for f in features]),
        thw=_arr([f.thw for f in features]),
        inv_thw=_arr([_inv(f.thw) for f in features]),
        ttc=_arr([f.ttc for f in features]),
        inv_ttc=_arr([_inv(f.ttc) for f in features]),
        drac=_arr([f.drac for f in features]),
        interaction_status=statuses,
        interaction_agent_indices=indices,
        interaction_agent_types=agent_type_pairs,
    )
