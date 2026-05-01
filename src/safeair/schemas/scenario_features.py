from pydantic import BaseModel


class CharacterizationParameters(BaseModel):
    """Thresholds that govern feature characterization computations — always in SI units (m, m/s, m/s², degrees).

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

    max_stationary_speed: float = 2.0  # m/s, ~3knots
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

    agent_to_agent_max_distance: float = 4000.0  # m, runway extent (10k-13kft) airplanning.com/post/airport-runway
    agent_to_agent_fov_deg: float = 45.0  # degrees, half-angle FOV for candidate pair selection
    agent_to_conflict_point_max_distance: float = 50.0  # m, to hold-line [Arbitrary]

    horizontal_separation_breach: float = 150.0  # m, lateral separation standard: airservicesaustralia.com
    vertical_separation_breach: float = 300.0  # m, 1000 ft, vertical separation standard: airservicesaustralia.com


class FeatureWeights(BaseModel):
    """Per-feature multipliers for score computation.

    All weights default to 1.0 except ``trajectory_type`` (0.1), matching the relative importance used in the
    reference SafeShift scorer (https://arxiv.org/abs/2202.07438).
    """

    speed: float = 1.0
    acceleration: float = 1.0
    deceleration: float = 1.0
    waiting_period: float = 1.0
    trajectory_type: float = 0.1
    kalman_difficulty: float = 1.0
    loss_of_separation: float = 1.0
    mttcp: float = 1.0
    thw: float = 1.0
    ttc: float = 1.0
    drac: float = 1.0


class FeatureDetections(BaseModel):
    """Per-feature detection (capping) thresholds for score computation.

    Each value caps the contribution of its feature before weighting. Defaults represent reasonable upper bounds for
    aviation ground operations.

    # NOTE: The metrics are in SI units.
    """

    # Typical fast taxiing speed for commercial aircraft is 30 knots (~15 m/s), normal speed is 10-20 knots (~5-10 m/s).
    # https://aviation.stackexchange.com/questions/426/what-is-the-maximum-taxi-speed-and-who-defines-it
    speed: float = 20.0  # m/s

    # Typical max ground acceleration for commercial aircraft is around 0.1-0.3g (0.98-2.94 m/s²), with 3 m/s²
    # being a reasonable upper bound for aggressive maneuvers.
    acceleration: float = 3.0  # m/s²

    # Autobrake systems provide selectable rates of deceleration which usually vary between 3 -6 knots per second
    # constant deceleration rate. https://skybrary.aero/articles/deceleration-runway
    deceleration: float = 3.0  # m/s²

    # Stationary waiting periods at hold-short points can vary widely, but we set the cap at 60 seconds since that is
    # the scenario length defined in amelia.
    waiting_period: float = 60.0  # seconds

    # Empirically derived. Unnormalized values typically range from 0 to 4000, but we use a downscaling factor of 100
    # and a cap from 0 to 50 to keep the score contributions in a reasonable range.
    kalman_difficulty: float = 30.0

    # Binary/fractional loss of separation indicator.
    loss_of_separation: float = 1.0

    # Empirically derived and use to detect the inverse of time (e.g., 1/mttcp), 10 will mean that we cap the value at
    # 0.1s to conflict.
    mttcp: float = 10.0  # 1/s
    thw: float = 10.0  # 1/s
    ttc: float = 10.0  # 1/s

    # Empirically derived
    drac: float = 5.0  # m/s²


class IndividualAgentFeatures(BaseModel):
    """Kinematic features for a single agent in a scenario.

    Attributes:
        agent_id: Agent identifier.
        agent_type: Agent type name (e.g. ``"AIRCRAFT"``).
        speed: Mean speed in m/s after moving-average smoothing. ``None`` if not computable.
        acceleration: Mean positive acceleration in m/s². ``None`` if not computable.
        deceleration: Mean magnitude of deceleration in m/s². ``None`` if not computable.
        waiting_period: Total time in seconds that the agent was stationary near a conflict point.
            ``None`` if no conflict point data is available.
        trajectory_type: Trajectory classification (e.g. ``"STRAIGHT"``). ``None`` if not computable.
        kalman_difficulty: Normalized Kalman-filter prediction error. Higher values mean harder to predict.
            ``None`` if insufficient history.
    """

    agent_id: int
    agent_type: str
    speed: float | None = None
    acceleration: float | None = None
    deceleration: float | None = None
    waiting_period: float | None = None
    trajectory_type: str | None = None
    kalman_difficulty: float | None = None

    model_config = {"validate_assignment": True}


class InteractionPairFeatures(BaseModel):
    """Pairwise interaction features between two agents in a scenario.

    Attributes:
        agent_id_a: First agent identifier.
        agent_id_b: Second agent identifier.
        pair_type: Agent pair type name (e.g. ``"AIRCRAFT_AIRCRAFT"``).
        loss_of_separation: 1.0 if trajectory paths intersect (with vertical separation check); otherwise the fraction
            of joint-valid timesteps where both lateral and vertical separation are below their respective breach
            thresholds. ``None`` if not computable.
        mttcp: Minimum time-to-conflict-point difference in seconds. ``None`` if no conflict points available.
        thw: Minimum time headway in seconds. ``None`` if no co-directional follower pair found.
        ttc: Minimum time-to-collision in seconds. ``None`` if agents are not closing.
        drac: Maximum deceleration rate to avoid collision in m/s². ``None`` if not computable.
    """

    agent_id_a: int
    agent_id_b: int
    pair_type: str
    loss_of_separation: float | None = None
    mttcp: float | None = None
    thw: float | None = None
    ttc: float | None = None
    drac: float | None = None

    model_config = {"validate_assignment": True}


class ScenarioFeatures(BaseModel):
    """All extracted features for a scenario.

    Attributes:
        scenario_id: Scenario identifier.
        individual_features: Per-agent kinematic features, one entry per agent.
        interaction_features: Pairwise interaction features for candidate agent pairs.
    """

    scenario_id: str
    individual_features: list[IndividualAgentFeatures]
    interaction_features: list[InteractionPairFeatures]

    model_config = {"validate_assignment": True}
