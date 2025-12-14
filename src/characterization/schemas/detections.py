from pydantic import BaseModel


class FeatureDetections(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the parameters for feature detections.

    Attributes:
        speed (float): Speed threshold in m/s.
        speed_limit_diff (float): Speed limit difference threshold in m/s.
        acceleration (float): Acceleration threshold in m/s^2.
        deceleration (float): Deceleration threshold in m/s^2.
        jerk (float): Jerk threshold in m/s^3.
        waiting_period (float): Waiting period threshold in seconds / meters.
        kalman_difficulty (float): Kalman filter difficulty threshold.
        mttcp (float): Minimum time to collision with a pedestrian threshold in seconds.
        thw (float): Time headway threshold in seconds.
        ttc (float): Time to collision threshold in seconds.
        drac (float): Deceleration rate to avoid collision threshold in m/s^2.
        collision (float): Collision threshold.
    """

    # NOTE: While speed detection values are largely dependent on the context (ie.., urban vs rural vs highways, etc),
    # our default values are set according to Proven Safety Countermeasures stating "A driver traveling at 30 mph who
    # hits a pedestrian has a 45% chance of killing or seriously injuring them..."
    # Reference: https://highways.dot.gov/sites/fhwa.dot.gov/files/App%20Speed%20Limits_508.pdf
    speed: float = 13.0  # in m/s, i.e. ~47 km/h or ~30 mph
    speed_limit_diff: float = 5.0  # in m/s, i.e. ~10 mph over the speed limit
    # NOTE: Acceleration detection value is derived from https://arxiv.org/pdf/2202.07438 (Table 3), which focuses on
    # harsh acceleration events in urban driving scenarios.
    acceleration: float = 10.0  # in m/s^2
    # NOTE: Braking detection is derived from "Reasonable skilled driver max (0.62g, ~4m/s^2)" obtained from:
    # https://copradar.com/chapts/references/acceleration.html and
    # https://shunauto.com/article/what-is-the-proper-acceleration-for-a-car
    deceleration: float = 5  # in m/s^2
    # NOTE: Jerk detection value is derived from https://mpmanser.com/wp-content/uploads/2020/04/Feng-Manser_2017.pdf,
    # which analyzes longitudinal jerk values to identify aggressive drivers.
    jerk: float = 1.5  # in m/s^3
    # NOTE: Waiting period at traffic signals detection value is derived form: "Measurement and comparative analysis of
    # driver's perception-reaction time to green phase at the intersections with and without a countdown timer", which
    # suggests reaction time beyond 4 seconds as slow.
    waiting_period: float = 4.0  # in s / m
    # NOTE: Kalman difficulty value derived from UniTraj: https://arxiv.org/pdf/2403.15098, which considers difficulty
    # levels above 50 as challenging for trajectory prediction tasks.
    kalman_difficulty: float = 50.0
    # NOTE: Time-based measures, which accounts for other agent vehicle are set from https://arxiv.org/pdf/2202.07438
    # (Table 3) with k=1
    mttcp: float = 4.0  # in seconds
    thw: float = 2.0  # in seconds
    ttc: float = 2.0  # in seconds
    # NOTE: Detection value for deceleration rate to avoid a collision, which accounts for other agent vehicle is set
    # from https://arxiv.org/pdf/2202.07438 (Table 3) with k=1
    drac: float = 3.0  # in m/s^2
    # NOTE: Collision detection threshold
    collision: float = 1.0

    model_config = {"validate_default": True, "validate_assignment": True, "extra": "forbid", "frozen": True}

    @classmethod
    def from_dict(cls, data: dict[str, float] | None) -> "FeatureDetections":
        """Creates an instance of FeatureDetections from a dictionary.

        Ignores any keys that are not defined in the model.
        If the input dictionary is empty, returns an instance with default values.

        Args:
            data (dict[str, float]): Dictionary containing the parameters.

        Returns:
            FeatureDetections: An instance of FeatureDetections.
        """
        if not data:
            return cls()

        allowed_keys = set(cls.model_fields.keys())
        filtered_data = {k: v for k, v in data.items() if k in allowed_keys}
        return cls(**filtered_data)


class FeatureWeights(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the weights for features.

    Attributes:
        speed (float): Speed threshold in m/s.
        speed_limit_diff (float): Speed limit difference threshold in m/s.
        acceleration (float): Acceleration threshold in m/s^2.
        deceleration (float): Deceleration threshold in m/s^2.
        jerk (float): Jerk threshold in m/s^3.
        waiting_period (float): Waiting period threshold in s/m.
        kalman_difficulty (float): Kalman filter difficulty threshold.
        trajectory_type (float): Trajectory type threshold.
        collision (float): Collision threshold.
        mttcp (float): Minimum time to collision with a pedestrian threshold in seconds.
        thw (float): Time headway threshold in seconds.
        ttc (float): Time to collision threshold in seconds.
        drac (float): Deceleration rate to avoid collision threshold in m/s^2.
    """

    speed: float = 1.0
    speed_limit_diff: float = 1.0
    acceleration: float = 1.0
    deceleration: float = 1.0
    jerk: float = 0.1
    waiting_period: float = 1.0
    kalman_difficulty: float = 1.0
    trajectory_type: float = 1.0
    collision: float = 1.0
    mttcp: float = 1.0
    thw: float = 1.0
    ttc: float = 1.0
    drac: float = 1.0

    model_config = {"validate_default": True, "validate_assignment": True, "extra": "forbid", "frozen": True}

    @classmethod
    def from_dict(cls, data: dict[str, float] | None) -> "FeatureWeights":
        """Creates an instance of FeatureWeights from a dictionary.

        Ignores any keys that are not defined in the model.
        If the input dictionary is empty, returns an instance with default values.

        Args:
            data (dict[str, float]): Dictionary containing the parameters.

        Returns:
            FeatureWeights: An instance of FeatureWeights.
        """
        if not data:
            return cls()

        allowed_keys = set(cls.model_fields.keys())
        filtered_data = {k: v for k, v in data.items() if k in allowed_keys}
        return cls(**filtered_data)
