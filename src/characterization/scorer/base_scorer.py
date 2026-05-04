"""General scorer abstractions shared across all domains."""

import json
import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from characterization.schemas.scenario_scores import ScenarioScores
from characterization.utils.common import ValueClipper, categorize_from_thresholds


class ScoreWeightingMethod(Enum):
    """Enumeration of score weighting methods."""

    UNIFORM = "uniform"
    DISTANCE_TO_EGO = "distance_to_ego"
    DISTANCE_TO_RELEVANT_AGENTS = "distance_to_relevant_agents"


class BaseScorerConfig(BaseModel):
    """Base configuration shared by all domain scorer configs.

    Attributes:
        score_weighting_method: How per-agent contributions are weighted when aggregating to a scene score.
        max_critical_distance: Minimum denominator (in meters) when computing distance-based weights, preventing
            extreme weights for agents very close to the focal agent.
        aggregated_score_weight: Weight ``w`` given to the individual scene score in the combined score
            ``w * individual + (1 - w) * interaction``.
        ego_pairs_only: If True, only score interaction pairs where one of the agents is the ego agent.
        categorize_scores: If True, map raw scores to percentile-based categories.
        score_clip: Min/max bounds for the final scene-level score.
    """

    score_weighting_method: ScoreWeightingMethod = ScoreWeightingMethod.UNIFORM
    max_critical_distance: float = 0.5
    aggregated_score_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    ego_pairs_only: bool = False
    categorize_scores: bool = False
    score_clip: ValueClipper = Field(default_factory=ValueClipper)

    model_config = {"validate_assignment": True}

    @classmethod
    def from_dict_config(cls, cfg: DictConfig) -> "Self":
        """Construct from an OmegaConf DictConfig, mapping common keys.

        Subclasses should override to also map domain-specific keys (weights, detections, etc.).

        Args:
            cfg: Hydra/OmegaConf configuration dict.

        Returns:
            Configured instance.
        """
        score_clip_raw = cfg.get("score_clip", {})
        score_clip = ValueClipper(
            min=score_clip_raw.get("min", 0.0) if score_clip_raw else 0.0,
            max=score_clip_raw.get("max", 200.0) if score_clip_raw else 200.0,
        )
        raw_method = cfg.get("score_weighting_method", ScoreWeightingMethod.UNIFORM.value)
        return cls(
            score_weighting_method=ScoreWeightingMethod(raw_method),
            max_critical_distance=cfg.get("max_critical_distance", 0.5),
            aggregated_score_weight=cfg.get("aggregated_score_weight", 0.5),
            ego_pairs_only=cfg.get("ego_pairs_only", False),
            categorize_scores=cfg.get("categorize_scores", False),
            score_clip=score_clip,
        )


class BaseScorer(ABC):
    """Abstract base class for scenario scorers."""

    def __init__(self) -> None:
        """Initialize base scorer state. Subclasses initialise domain-specific configuration."""
        self.characterizer_type = "score"
        self.categories: dict[str, float] | None = None

    @property
    def name(self) -> str:
        """Returns the class name formatted as a lowercase string with underscores."""
        return re.sub(r"(?<!^)([A-Z])", r"_\1", self.__class__.__name__).lower()

    @staticmethod
    def _get_weights_uniform(num_agents: int) -> NDArray[np.float32]:
        """Returns uniform weights (1.0) for all agents.

        Args:
            num_agents: Number of agents in the scenario.

        Returns:
            NDArray[np.float32]: Uniform weights.
        """
        return np.ones(num_agents, dtype=np.float32)

    def categorize(self, score: float) -> float:
        """Categorizes a score based on predefined percentile thresholds.

        Args:
            score: The score to categorize.

        Returns:
            The categorized score.
        """
        if self.categories is None:
            error_message = "Categories not loaded. Cannot categorize scores."
            raise ValueError(error_message)
        threshold_values = list(self.categories.values())
        return float(categorize_from_thresholds(score, threshold_values))

    def _load_categorization_file(self, path_str: str) -> dict[str, float]:
        """Load a JSON categorization (percentile) file.

        Args:
            path_str: Path to the JSON file.

        Returns:
            dict mapping percentile labels to threshold values.
        """
        categorization_file = Path(path_str)
        if not categorization_file.is_file():
            msg = f"Categorization file {categorization_file} does not exist."
            raise FileNotFoundError(msg)
        with categorization_file.open("r") as f:
            return json.load(f)

    @abstractmethod
    def compute(self, scenario: object, scenario_features: object) -> ScenarioScores:
        """Computes scenario-level scores from features.

        Args:
            scenario: Scenario object containing scenario information.
            scenario_features: ScenarioFeatures object containing computed features.

        Returns:
            ScenarioScores: An object containing computed scenario scores.
        """
