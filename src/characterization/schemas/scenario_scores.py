"""General scenario scoring output schemas shared across domains."""

from typing import Any

from pydantic import BaseModel


class AgentScore(BaseModel):
    """Score for a single agent.

    Attributes:
        agent_id: Agent identifier.
        score: Computed score value.
    """

    agent_id: int
    score: float

    model_config = {"validate_assignment": True}


class ScenarioScores(BaseModel):
    """Scoring outputs for a scenario.

    Attributes:
        scenario_id: Scenario identifier.
        individual_scores: Per-agent individual (kinematic) scores.
        interaction_scores: Per-agent accumulated interaction scores.
        individual_scene_score: Scene-level aggregate of individual scores. ``None`` if not computed.
        interaction_scene_score: Scene-level aggregate of interaction scores. ``None`` if not computed.
        scene_score: Combined scene-level score (weighted average of individual and interaction). ``None`` if not
            computed.
    """

    scenario_id: str
    individual_scores: list[AgentScore] = []
    interaction_scores: list[AgentScore] = []
    individual_scene_score: float | None = None
    interaction_scene_score: float | None = None
    scene_score: float | None = None

    model_config = {"validate_assignment": True}

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get a score by attribute name.

        Args:
            key: Attribute name to look up.

        Returns:
            The value of the named attribute.

        Raises:
            KeyError: If the key is not found.
        """
        if hasattr(self, key):
            return getattr(self, key)
        error_message = f"Key '{key}' not found in ScenarioScores."
        raise KeyError(error_message)
