from pydantic import BaseModel


class FeatureDetections(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the parameters for individual feature detections.

    Attributes:
        speed (float): Speed threshold in m/s.
        speed_limit_diff (float): Speed limit difference threshold in m/s.
        acceleration (float): Acceleration threshold in m/s^2.
        deceleration (float): Deceleration threshold in m/s^2.
        jerk (float): Jerk threshold in m/s^3.
        waiting_period (float): Waiting period threshold in seconds.
        waiting_intervals (float): Waiting intervals threshold in number of intervals.
        waiting_distances (float): Waiting distances threshold in meters.
        kalman_difficulty (float): Kalman filter difficulty threshold.

        mttcp (float): Minimum time to collision with a pedestrian threshold in seconds.
        thw (float): Time headway threshold in seconds.
        ttc (float): Time to collision threshold in seconds.
        drac (float): Deceleration rate to avoid collision threshold in m/s^2.
    """

    speed: float = 30.0  # in m/s, i.e. ~108 km/h ~70 mph
    speed_limit_diff: float = 8.0  # in m/s, i.e. ~30 km/h over the speed limit
    acceleration: float = 10.0  # in m/s^2
    deceleration: float = 10.0  # in m/s^2
    jerk: float = 10.0  # in m/s^3
    waiting_period: float = 8.0  # in seconds
    waiting_intervals: float = 8.0  # in number of intervals
    waiting_distances: float = 8.0  # in meters
    kalman_difficulty: float = 80.0

    mttcp: float = 4.0  # in seconds
    thw: float = 4.0  # in seconds
    ttc: float = 4.0  # in seconds
    drac: float = 3.0  # in m/s^2

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
