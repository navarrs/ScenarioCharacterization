from pydantic import BaseModel

from characterization.utils.ad_types import AgentType
from characterization.utils.common import (
    Float32NDArray1D,
    Float32NDArray2D,
    Int32NDArray1D,
    InteractionStatus,
    TrajectoryType,
)

from .scenario import ScenarioMetadata


class Individual(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Represents the individual features extracted from a scenario for each agent.

    This class is used to encapsulate all relevant individual features for agents in a scenario, including their states
    and characteristics that can be used for analysis or modeling.

    Attributes:
        speed (Float32NDArray1D | None): Speed of each agent at each timestep.
        speed_limit_diff (Float32NDArray1D | None): Difference between agent speed and speed limit at each timestep.
        acceleration (Float32NDArray1D | None): Acceleration of each agent at each timestep.
        deceleration (Float32NDArray1D | None): Deceleration of each agent at each timestep.
        jerk (Float32NDArray1D | None): Jerk (rate of change of acceleration) of each agent at each timestep.
        waiting_period (Float32NDArray1D | None): Waiting period of each agent at each timestep.
        waiting_interval (Float32NDArray1D | None): Waiting interval of each agent at each timestep.
        waiting_distance (Float32NDArray1D | None): Waiting distance of each agent at each timestep.
        kalman_difficulty (Float32NDArray1D | None): Kalman filter difficulty value for each agent.
    """

    # Agent meta
    valid_idxs: Int32NDArray1D | None = None
    agent_types: list[AgentType] | None = None
    agent_trajectory_types: list[TrajectoryType]

    # Individual Features
    speed: Float32NDArray1D | None = None
    speed_limit_diff: Float32NDArray1D | None = None
    acceleration: Float32NDArray1D | None = None
    deceleration: Float32NDArray1D | None = None
    jerk: Float32NDArray1D | None = None
    waiting_period: Float32NDArray1D | None = None
    waiting_interval: Float32NDArray1D | None = None
    waiting_distance: Float32NDArray1D | None = None
    kalman_difficulty: Float32NDArray1D | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}


class Interaction(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Represents the features extracted from a scenario, including individual agent features and interaction features.

    This class is used to encapsulate all relevant features for a scenario, including agent states, interaction metrics,
    and other characteristics that can be used for analysis or modeling.

    Attributes:
        metadata (ScenarioMetadata): Metadata for the scenario, including IDs, timestamps, and thresholds.
        valid_idxs (Int32NDArray1D | None): Indices of valid agents in the scenario.
        agent_types (list[str] | None): list of types for each agent, e.g., "vehicle", "pedestrian", etc.
        agent_to_agent_closest_dists (Float32NDArray2D | None):
            2D array of shape(num_agents, num_agents) representing the closest distances between agents.
        separation (Float32NDArray1D | None): Separation distance between agents at each timestep.
        intersection (Float32NDArray1D | None): Intersection distance between agents at each timestep.
        collision (Float32NDArray1D | None): Collision distance between agents at each timestep.
        mttcp (Float32NDArray1D | None): Minimum time to conflict point (mTTCP) for each agent at each timestep.
        interaction_status (list[InteractionStatus] | None):
            list of interaction statuses for each agent pair in the scenario.
        interaction_agent_indices (list[tuple[int, int]] | None):
            list of tuples representing the indices of interacting agents in the scenario.
        interaction_agent_types (list[tuple[str, str]] | None):
            list of tuples representing the types of interacting agents in the scenario.
    """

    # Interaction Features
    separation: Float32NDArray1D | None = None
    intersection: Float32NDArray1D | None = None
    collision: Float32NDArray1D | None = None
    mttcp: Float32NDArray1D | None = None
    thw: Float32NDArray1D | None = None
    ttc: Float32NDArray1D | None = None
    drac: Float32NDArray1D | None = None

    # leader_follower: Float32NDArray1D | None = None
    # valid_headings: Float32NDArray1D | None = None
    interaction_status: list[InteractionStatus] | None = None
    interaction_agent_indices: list[tuple[int, int]] | None = None
    interaction_agent_types: list[tuple[AgentType, AgentType]] | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}


class ScenarioFeatures(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Represents the features extracted from a scenario, including individual agent features and interaction features.

    This class is used to encapsulate all relevant features for a scenario, including agent states, interaction metrics,
    and other characteristics that can be used for analysis or modeling.

    Attributes:
        metadata (ScenarioMetadata): Metadata for the scenario, including IDs, timestamps, and thresholds.
        individual_features (IndividualFeatures | None): Individual features for each agent in the scenario.
        interaction_features (InteractionFeatures | None): Interaction features between agents in the scenario.
    """

    metadata: ScenarioMetadata

    # Agent meta
    individual_features: Individual | None = None

    # Interaction Features
    interaction_features: Interaction | None = None

    agent_to_agent_closest_dists: Float32NDArray2D | None = None
    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}
