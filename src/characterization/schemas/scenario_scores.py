from typing import Any

from pydantic import BaseModel

from characterization.utils.common import Float32NDArray1D

from .scenario import ScenarioMetadata


class Score(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Represents a score for a scenario or agent, including individual and scene scores.

    Attributes:
        agent_scores (Float32NDArray1D | None): Individual scores for each agent in the scenario.
        scene_score (float | None): Overall score for the scene based on individual agent scores.
    """

    agent_scores: Float32NDArray1D | None = None
    scene_score: float | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}


class ScenarioScores(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Represents the scores for a scenario, including individual agent scores, interaction scores, and combined.

    This class is used to encapsulate the results of scoring a scenario based on various criteria, such as safety and
    interaction quality.

    Attributes:
        metadata (ScenarioMetadata): Metadata about the scenario being scored.
        individual (Score | None): Individual scores for agents in the scenario.
        interaction (Score | None): Interaction scores for the scenario, capturing the quality of interactions
            between agents.
        safeshift (Score | None): Combined score that reflects the overall safety and interaction quality of the
            scenario.
    """

    metadata: ScenarioMetadata

    # Individual Scores
    individual_scores: Score | None = None

    # Interaction Scores
    interaction_scores: Score | None = None

    # Combined Scores
    safeshift_scores: Score | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get the value of a key in the ScenarioScores object.

        Args:
            key (str): The key to retrieve.

        Returns:
            Any: The value associated with the key.
        """
        if hasattr(self, key):
            return getattr(self, key)
        error_message = f"Key '{key}' not found in ScenarioScores."
        raise KeyError(error_message)
