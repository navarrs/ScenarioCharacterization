"""Schema for counterfactual probing results."""

from pydantic import BaseModel

from characterization.probing.common import CriticalityMetric, ProbeType
from characterization.utils.common import Float32NDArray2D


class CriticalityResult(BaseModel):
    """Criticality timestamp and metric for a single affected agent.

    Attributes:
        timestamp: Absolute frame index where peak criticality occurs in the counterfactual scenario.
        metric: Safety metric used to identify the critical frame (TTC or DRAC).
    """

    timestamp: int
    metric: CriticalityMetric

    model_config = {"frozen": True}


class CriticalProbe(BaseModel):
    """Result of applying a counterfactual probe to a scenario.

    Attributes:
        probed_agent_id: ID of the agent whose trajectory was replaced by the counterfactual.
        probed_agent_trajectory: Full counterfactual trajectory for the probed agent, shape (T, 10).
        is_ego_agent: True if the probed agent is the ego vehicle.
        probe_type: Type of counterfactual applied (e.g. CONSTANT_VELOCITY).
        affected_agent_ids: IDs of agents that become critically close to the probed agent under
            the counterfactual. Filtered to the single most-critical agent when
            ``single_affected_agent=True`` in the probing config.
        criticality_results: Per-affected-agent criticality; keyed by ``str(agent_id)``.
        score_before: Interaction scene score of the unperturbed scenario.
        score_after: Interaction scene score with the counterfactual applied.
        affected_pair_scores_before: Baseline pair scores keyed by ``"id_a:id_b"`` (canonical order).
        affected_pair_scores_after: Post-probe pair scores, same keying as above.
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

    model_config = {"arbitrary_types_allowed": True}
