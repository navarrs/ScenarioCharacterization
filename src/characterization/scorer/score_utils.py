import numpy as np

from characterization.utils.common import EPS


def simple_individual_score(
    speed: float = 0.0,
    speed_weight: float = 1.0,
    speed_detection: float = 1.0,
    acceleration: float = 0.0,
    acceleration_weight: float = 1.0,
    acceleration_detection: float = 1.0,
    deceleration: float = 0.0,
    deceleration_weight: float = 1.0,
    deceleration_detection: float = 1.0,
    jerk: float = 0.0,
    jerk_weight: float = 1.0,
    jerk_detection: float = 1.0,
    waiting_period: float = 0.0,
    waiting_period_weight: float = 1.0,
    waiting_period_detection: float = 1.0,
) -> np.ndarray:
    """Aggregates a simple score for an agent using weighted feature values.

    Args:
        **kwargs: Feature values for the agent, including speed, acceleration, deceleration,
            jerk, and waiting_period.

    Returns:
        np.ndarray: The aggregated score for the agent.
    """
    # Detection values are roughly obtained from: https://arxiv.org/abs/2202.07438
    return (
        speed_weight * min(speed_detection, speed)
        + acceleration_weight * min(acceleration_detection, acceleration)
        + deceleration_weight * min(deceleration_detection, deceleration)
        + jerk_weight * min(jerk_detection, jerk)
        + waiting_period_weight * min(waiting_period_detection, waiting_period)
    )


def simple_interaction_score(
    collision: float = 0.0,
    collision_weight: float = 1.0,
    mttcp: float = 0.0,
    mttcp_weight: float = 1.0,
    mttcp_detection: float = 1.0,
) -> np.ndarray:
    """Aggregates a simple interaction score for an agent pair using weighted feature values.

    Args:
        **kwargs: Feature values for the agent pair, including collision and mttcp.

    Returns:
        np.ndarray: The aggregated score for the agent pair.
    """
    inv_mttcp = 1.0 / (mttcp + EPS)
    return collision_weight * collision + mttcp_weight * min(mttcp_detection, inv_mttcp)
