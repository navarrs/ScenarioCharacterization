import numpy as np

from characterization.utils.common import EPS

SUPPORTED_SCORERS = ["individual", "interaction", "safeshift"]


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
) -> float:
    """Aggregates a simple score for an agent using weighted feature values.

    Args:
        **kwargs: Feature values for the agent, including speed, acceleration, deceleration,
            jerk, and waiting_period.

    Returns:
        float: The aggregated score for the agent.
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
    mttcp: float = np.inf,
    mttcp_weight: float = 1.0,
    mttcp_detection: float = 1.0,
    thw: float = np.inf,
    thw_weight: float = 1.0,
    thw_detection: float = 1.0,
    ttc: float = np.inf,
    ttc_weight: float = 1.0,
    ttc_detection: float = 1.0,
    drac: float = 0.0,
    drac_weight: float = 1.0,
    drac_detection: float = 1.0,
) -> float:
    """Aggregates a simple interaction score for an agent pair using weighted feature values.

    Args:
        **kwargs: Feature values for the agent pair, including collision and mttcp.

    Returns:
        float: The aggregated score for the agent pair.
    """
    inv_mttcp = 1.0 / (mttcp + EPS)
    inv_thw = 1.0 / (thw + EPS)
    inv_ttc = 1.0 / (ttc + EPS)
    return (
        collision_weight * collision
        + mttcp_weight * min(mttcp_detection, inv_mttcp)
        + thw_weight * min(thw_detection, inv_thw)
        + ttc_weight * min(ttc_detection, inv_ttc)
        + min(drac_detection, drac_weight * drac)
    )


INDIVIDUAL_SCORE_FUNCTIONS = {
    "simple": simple_individual_score,
}
INTERACTION_SCORE_FUNCTIONS = {
    "simple": simple_interaction_score,
}
