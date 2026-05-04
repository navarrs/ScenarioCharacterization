from typing import Self

from pydantic import BaseModel, ConfigDict


class BaseFeatureDetections(BaseModel):
    """Base detection thresholds shared across domains.

    Each domain subclass overrides these defaults with domain-appropriate values and may add domain-specific fields
    (e.g. ``speed_limit_diff`` for AD, ``loss_of_separation`` for aviation).

    All values are in SI units (m/s, m/s², s).
    """

    speed: float = 0.0
    acceleration: float = 0.0
    deceleration: float = 0.0
    waiting_period: float = 0.0
    kalman_difficulty: float = 0.0
    mttcp: float = 0.0
    thw: float = 0.0
    ttc: float = 0.0
    drac: float = 0.0

    model_config = {"validate_default": True, "validate_assignment": True, "extra": "forbid", "frozen": True}

    @classmethod
    def from_dict(cls, data: dict[str, float] | None) -> "Self":
        """Create an instance from a dict, ignoring unknown keys.

        Args:
            data: Threshold values. Keys not present in the model are silently ignored.

        Returns:
            Instance with values from ``data``, falling back to field defaults for missing keys.
        """
        if not data:
            return cls()
        allowed_keys = set(cls.model_fields.keys())
        filtered_data = {k: v for k, v in data.items() if k in allowed_keys}
        return cls(**filtered_data)


class BaseFeatureWeights(BaseModel):
    """Base feature weights shared across domains.

    Each domain subclass overrides these defaults and may add domain-specific weight fields.
    """

    speed: float = 1.0
    acceleration: float = 1.0
    deceleration: float = 1.0
    waiting_period: float = 1.0
    kalman_difficulty: float = 1.0
    trajectory_type: float = 1.0
    mttcp: float = 1.0
    thw: float = 1.0
    ttc: float = 1.0
    drac: float = 1.0

    model_config = {"validate_default": True, "validate_assignment": True, "extra": "forbid", "frozen": True}

    @classmethod
    def from_dict(cls, data: dict[str, float] | None) -> "Self":
        """Create an instance from a dict, ignoring unknown keys.

        Args:
            data: Weight values. Keys not present in the model are silently ignored.

        Returns:
            Instance with values from ``data``, falling back to field defaults for missing keys.
        """
        if not data:
            return cls()
        allowed_keys = set(cls.model_fields.keys())
        filtered_data = {k: v for k, v in data.items() if k in allowed_keys}
        return cls(**filtered_data)


class BaseIndividualAgentFeatures(BaseModel):
    """Kinematic features for a single agent, shared across domains.

    All numeric fields default to ``None`` (not computable). Domain-specific fields that are
    unused in a given domain simply remain ``None``.

    Attributes:
        agent_id: Index of the agent within the scenario.
        agent_type: Agent type name (e.g. ``"VEHICLE"`` or ``"AIRCRAFT"``).
        speed: Aggregated speed in m/s. ``None`` if not computable.
        acceleration: Aggregated positive acceleration in m/s². ``None`` if not computable.
        deceleration: Aggregated deceleration magnitude in m/s². ``None`` if not computable.
        waiting_period: Total time in seconds the agent was stationary near a conflict point.
            ``None`` if no conflict point data is available.
        trajectory_type: Trajectory classification string (e.g. ``"STRAIGHT"``). ``None`` if not computable.
        kalman_difficulty: Normalised Kalman-filter prediction error. ``None`` if insufficient history.
        speed_limit_diff: Difference between agent speed and the posted speed limit (m/s).
            AD-specific; ``None`` for other domains.
        jerk: Aggregated jerk in m/s³. AD-specific; ``None`` for other domains.
    """

    model_config = ConfigDict(validate_assignment=True)

    agent_id: int
    agent_type: str
    speed: float | None = None
    acceleration: float | None = None
    deceleration: float | None = None
    waiting_period: float | None = None
    trajectory_type: str | None = None
    kalman_difficulty: float | None = None
    speed_limit_diff: float | None = None
    jerk: float | None = None


class BaseInteractionPairFeatures(BaseModel):
    """Pairwise interaction features between two agents, shared across domains.

    All numeric fields default to ``None`` (not computable or not applicable). Domain-specific
    fields that are unused in a given domain simply remain ``None``.

    Attributes:
        agent_id_a: Index of the first agent within the scenario.
        agent_id_b: Index of the second agent within the scenario.
        pair_type: Agent pair type string (e.g. ``"VEHICLE_VEHICLE"`` or ``"AIRCRAFT_AIRCRAFT"``).
        mttcp: Minimum time-to-conflict-point in seconds. ``None`` if no conflict points available.
        thw: Minimum time headway in seconds. ``None`` if no co-directional follower pair found.
        ttc: Minimum time-to-collision in seconds. ``None`` if agents are not closing.
        drac: Maximum deceleration rate to avoid collision in m/s². ``None`` if not computable.
        separation: Minimum or average separation distance in metres between the pair.
            AD-specific; ``None`` for other domains.
        intersection: Fraction or sum of timesteps with geometric path intersection.
            AD-specific; ``None`` for other domains.
        collision: Collision indicator — fraction of timesteps within breach distance or with
            intersection. AD-specific; ``None`` for other domains.
    """

    model_config = ConfigDict(validate_assignment=True)

    agent_id_a: int
    agent_id_b: int
    pair_type: str
    mttcp: float | None = None
    thw: float | None = None
    ttc: float | None = None
    drac: float | None = None
    separation: float | None = None
    intersection: float | None = None
    collision: float | None = None
