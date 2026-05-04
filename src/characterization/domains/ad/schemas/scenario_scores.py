"""AD scenario scoring schemas.

``Score`` is an internal numpy-based helper used by AD scorers for intermediate
computations. ``AgentScore`` and ``ScenarioScores`` are the canonical public
output types shared with all domains.
"""

from pydantic import BaseModel

from characterization.schemas.scenario_scores import AgentScore, ScenarioScores
from characterization.schemas.types import BooleanNDArray1D, Float32NDArray1D

__all__ = ["AgentScore", "ScenarioScores", "Score"]


class Score(BaseModel):
    """Internal numpy-based score container used by AD scorers during computation.

    Use ``ScenarioScores`` with ``list[AgentScore]`` for the public output.

    Attributes:
        agent_scores: Per-agent scores as a numpy array.
        agent_scores_valid: Boolean mask indicating valid agent scores.
        scene_score: Aggregate scene-level score.
    """

    agent_scores: Float32NDArray1D | None = None
    agent_scores_valid: BooleanNDArray1D | None = None
    scene_score: float | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}
