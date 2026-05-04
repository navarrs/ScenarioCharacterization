from pydantic import BaseModel

from characterization.schemas.scenario_features import (
    BaseFeatureDetections,
    BaseFeatureWeights,
    BaseIndividualAgentFeatures,
    BaseInteractionPairFeatures,
)


class CharacterizationParameters(BaseModel):
    """Thresholds governing feature characterization — always in SI units (m, m/s, m/s², degrees).

    Obtained from: https://github.com/AmeliaCMU/AmeliaScenes/blob/main/amelia_scenes

    Attributes:
        max_stationary_speed: Speed threshold in m/s below which the trajectory is considered stationary.
        max_stationary_displacement: Max total displacement in meters for a trajectory to be considered stationary.
        max_straight_lateral_displacement: Max lateral deviation in meters for a trajectory to be considered straight.
        min_uturn_longitudinal_displacement: Min forward/backward displacement in meters for a U-turn.
        min_takeoff_altitude_gain: Min net altitude gain in meters to classify as takeoff.
        min_takeoff_speed_gain: Min net speed gain in m/s to classify as takeoff (ground roll before liftoff).
        min_landing_altitude_loss: Min net altitude loss in meters to classify as landing or landed.
        max_landed_altitude: Max altitude at end of trajectory in meters to classify as landed (vs landing).
        max_straight_absolute_heading_diff: Max heading change in degrees for a trajectory to be considered straight.
        agent_to_agent_max_distance: Max distance in meters between agents to be considered interacting.
        agent_to_agent_fov_deg: Half-angle of the FOV cone (in degrees) centred on each agent's forward heading used to
            pre-filter candidate interaction pairs beyond ``horizontal_separation_breach``. A pair is included only if
            either agent's heading points toward the other within this angle at any joint-valid timestep. 45° means ±45°
            around the forward direction (a 90° total cone). Pairs within ``horizontal_separation_breach`` are always
            included regardless of heading.
        agent_to_conflict_point_max_distance: Max distance in meters from a conflict point to be considered near it.
        horizontal_separation_breach: Lateral (XY) distance in meters below which agents are considered in breach of
            separation.
        vertical_separation_breach: Altitude difference in meters below which agents are considered in breach of
            vertical separation.
        heading_threshold: Max heading difference in degrees to consider two agents co-directional.
        agent_max_deceleration: Maximum feasible deceleration in m/s² (used as clip upper bound for DRAC).
    """

    max_stationary_speed: float = 2.0  # m/s, ~3 knots
    agent_max_deceleration: float = 5.0  # m/s²

    max_stationary_displacement: float = 5.0  # m
    max_straight_lateral_displacement: float = 5.0  # m
    min_uturn_longitudinal_displacement: float = -10.0  # m

    max_straight_absolute_heading_diff: float = 30.0  # degrees
    heading_threshold: float = 30.0  # degrees

    min_takeoff_speed_gain: float = 15.0  # m/s, ~30 knots
    min_takeoff_altitude_gain: float = 10.0  # m, ~35 ft
    min_landing_altitude_loss: float = 30.0  # m
    max_landed_altitude: float = 10.0  # m

    agent_to_agent_max_distance: float = 4000.0  # m, runway extent
    agent_to_agent_fov_deg: float = 45.0  # degrees, half-angle FOV for candidate pair selection
    agent_to_conflict_point_max_distance: float = 50.0  # m, to hold-line

    horizontal_separation_breach: float = 150.0  # m, lateral separation standard
    vertical_separation_breach: float = 300.0  # m, 1000 ft, vertical separation standard


class FeatureDetections(BaseFeatureDetections):
    """Detection thresholds for aviation feature computation.

    Extends :class:`BaseFeatureDetections` with aviation-specific thresholds.

    Attributes:
        speed: Cap for taxiing speed (m/s).
        acceleration: Cap for ground acceleration (m/s²).
        deceleration: Cap for autobrake deceleration (m/s²).
        waiting_period: Cap for hold-short waiting time (s).
        kalman_difficulty: Cap for Kalman prediction error.
        loss_of_separation: Binary/fractional loss-of-separation indicator cap.
        mttcp: Cap for inverse time to conflict point (1/s).
        thw: Cap for inverse time headway (1/s).
        ttc: Cap for inverse time to collision (1/s).
        drac: Cap for deceleration rate to avoid collision (m/s²).
    """

    # Typical fast taxiing speed for commercial aircraft is 30 knots (~15 m/s)
    # https://aviation.stackexchange.com/questions/426/what-is-the-maximum-taxi-speed-and-who-defines-it
    speed: float = 20.0  # m/s

    # Typical max ground acceleration is around 0.1-0.3g (0.98-2.94 m/s²)
    acceleration: float = 3.0  # m/s²

    # Autobrake systems provide selectable rates: https://skybrary.aero/articles/deceleration-runway
    deceleration: float = 3.0  # m/s²

    # Scenario length is 60 s in amelia
    waiting_period: float = 60.0  # s

    # Empirically derived; unnormalized values range 0-4000, downscaled by 100
    kalman_difficulty: float = 30.0

    # Binary/fractional loss of separation indicator
    loss_of_separation: float = 1.0

    # Inverse time metrics: 10 means we cap at 0.1 s to conflict
    mttcp: float = 10.0  # 1/s
    thw: float = 10.0  # 1/s
    ttc: float = 10.0  # 1/s

    drac: float = 5.0  # m/s²


class FeatureWeights(BaseFeatureWeights):
    """Feature weights for aviation score computation.

    Extends :class:`BaseFeatureWeights` with an aviation-specific ``loss_of_separation`` weight.
    """

    speed: float = 1.0
    acceleration: float = 1.0
    deceleration: float = 1.0
    waiting_period: float = 1.0
    trajectory_type: float = 0.1  # Lower weight than AD default
    kalman_difficulty: float = 1.0
    loss_of_separation: float = 1.0
    mttcp: float = 1.0
    thw: float = 1.0
    ttc: float = 1.0
    drac: float = 1.0


class IndividualAgentFeatures(BaseIndividualAgentFeatures):
    """Kinematic features for a single agent in an aviation scenario.

    Inherits all common fields from :class:`BaseIndividualAgentFeatures`.
    Aviation-specific fields (``speed_limit_diff``, ``jerk``) are inherited but always ``None``.
    """


class InteractionPairFeatures(BaseInteractionPairFeatures):
    """Pairwise interaction features between two agents in an aviation scenario.

    Extends :class:`BaseInteractionPairFeatures` with the aviation-specific
    ``loss_of_separation`` field. AD-specific fields (``separation``, ``intersection``,
    ``collision``) are inherited but always ``None`` for aviation scenarios.

    Attributes:
        loss_of_separation: 1.0 if trajectory paths intersect (with vertical separation check);
            otherwise the fraction of joint-valid timesteps where both lateral and vertical
            separation are below their respective breach thresholds. ``None`` if not computable.
    """

    loss_of_separation: float | None = None


class ScenarioFeatures(BaseModel):
    """All extracted features for an aviation scenario.

    Attributes:
        scenario_id: Scenario identifier.
        individual_features: Per-agent kinematic features, one entry per agent.
        interaction_features: Pairwise interaction features for candidate agent pairs.
    """

    scenario_id: str
    individual_features: list[IndividualAgentFeatures]
    interaction_features: list[InteractionPairFeatures]

    model_config = {"validate_assignment": True}
