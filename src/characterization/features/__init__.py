from .base_feature import BaseFeature

SUPPORTED_FEATURES = [
    "random_feature",
    # AD features
    "speed",
    "speed_limit_diff",
    "acceleration",
    "deceleration",
    "jerk",
    "waiting_period",
    "trajectory_type",
    "kalman_difficulty",
    "separation",
    "intersection",
    "collision",
    "mttcp",
    "thw",
    "ttc",
    "drac",
    # Aviation features
    "loss_of_separation",
]

__all__ = [
    "SUPPORTED_FEATURES",
    "BaseFeature",
]
