"""Shared enumerations and data contracts for counterfactual scenario probing."""

from enum import Enum, StrEnum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class ProbeValidity(Enum):
    """Validity of a counterfactual probe trajectory."""

    VALID = "VALID"
    INVALID = "INVALID"


class ProbeType(StrEnum):
    """Type of counterfactual probe applied to an agent."""

    CONSTANT_VELOCITY = "CONSTANT_VELOCITY"


class ValidatorType(StrEnum):
    """Strategy used to select the best probe candidate from all valid per-agent probes."""

    MAX_SCORE_DELTA = "MAX_SCORE_DELTA"


class CriticalityMetric(StrEnum):
    """Safety metric used to identify the critical frame in a probed interaction."""

    TTC = "TTC"
    DRAC = "DRAC"


class CandidateProbeResult(NamedTuple):
    """Per-agent probe data passed to the validator for selection.

    Attributes:
        agent_id: ID of the probed agent.
        is_ego: Whether the probed agent is the ego vehicle.
        perturbed_traj: Counterfactual trajectory for the probed agent.
        pair_delta: Maximum pair-score increase across all affected pairs (pre-computed by the prober).
        affected_ids: IDs of agents whose pair score exceeded ``min_score_delta``.
        scores_before: Baseline pair scores keyed as ``"id_a:id_b"``.
        scores_after: Post-probe pair scores keyed as ``"id_a:id_b"``.
    """

    agent_id: int
    is_ego: bool
    perturbed_traj: NDArray[np.float32]
    pair_delta: float
    affected_ids: list[int]
    scores_before: dict[str, float]
    scores_after: dict[str, float]
