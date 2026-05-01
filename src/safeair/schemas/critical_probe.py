"""Pydantic schemas for counterfactual probe results."""

from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel

from safeair.schemas.types import Float32NDArray2D


class ProbeType(StrEnum):
    """Counterfactual probe function type."""

    CONSTANT_VELOCITY = "constant_velocity"


class CriticalityMetric(StrEnum):
    """Safety metric that identified the criticality timestamp.

    Indicates which measure was used to locate the most dangerous moment in the probed trajectory.
    """

    TTC = "ttc"
    DRAC = "drac"


class CriticalityResult(NamedTuple):
    """Timestamp and metric for the most critical moment in a probed trajectory pair.

    Attributes:
        timestamp: Absolute frame index of peak criticality.
        metric: Which safety metric (TTC or DRAC) identified this moment.
    """

    timestamp: int
    metric: CriticalityMetric


class CriticalProbe(BaseModel):
    """Result of a counterfactual trajectory probe that increased scenario criticality.

    Attributes:
        probed_agent_id: ID of the agent whose future trajectory was replaced by a counterfactual.
        probed_agent_trajectory: Shape ``(T, 10)`` counterfactual trajectory used during probing.
        is_ego_agent: Whether the probed agent is the ego agent.
        probe_type: The type of counterfactual probe function used.
        affected_agent_ids: IDs of agents whose interaction criticality increased due to the probe.
        criticality_results: Maps ``str(agent_id)`` to the :class:`CriticalityResult` for that agent — the frame
            index and safety metric of its most critical moment in the probed future. One entry per affected agent.
            Restricted to timesteps after ``current_time_index``.
        score_before: Interaction scene score before probing.
        score_after: Interaction scene score after probing.
        affected_pair_scores_before: Raw pair scores before probing for each affected pair, keyed ``"min_id:max_id"``.
        affected_pair_scores_after: Raw pair scores after probing for each affected pair, keyed ``"min_id:max_id"``.
    """

    probed_agent_id: int
    probed_agent_trajectory: Float32NDArray2D
    is_ego_agent: bool
    probe_type: ProbeType
    affected_agent_ids: list[int]
    criticality_results: dict[str, CriticalityResult]
    score_before: float
    score_after: float
    affected_pair_scores_before: dict[str, float]
    affected_pair_scores_after: dict[str, float]

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}
