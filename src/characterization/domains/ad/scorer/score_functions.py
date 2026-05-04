"""AD-domain scoring functions for individual agents and interaction pairs."""

from characterization.utils.common import TrajectoryType
from characterization.utils.constants import EPSILON, TRAJECTORY_TYPE_WEIGHTS


def simple_individual_score(
    speed: float = 0.0,
    speed_weight: float = 1.0,
    speed_detection: float = 1.0,
    speed_limit_diff: float = 0.0,
    speed_limit_diff_weight: float = 1.0,
    speed_limit_diff_detection: float = 1.0,
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
    trajectory_type: TrajectoryType = TrajectoryType.TYPE_UNSET,
    trajectory_type_weight: float = 0.1,
    kalman_difficulty: float = 0.0,
    kalman_difficulty_weight: float = 1.0,
    kalman_difficulty_detection: float = 1.0,
) -> float:
    """Aggregate a simple score for an AD agent from weighted feature values.

    Each feature's contribution is capped at its detection threshold before being multiplied by its weight. Detection
    thresholds are loosely inspired by https://arxiv.org/abs/2202.07438.

    Args:
        speed: Speed of the agent (m/s).
        speed_weight: Weight for the speed feature.
        speed_detection: Detection cap for the speed feature.
        speed_limit_diff: Speed difference above the speed limit (m/s).
        speed_limit_diff_weight: Weight for the speed limit difference feature.
        speed_limit_diff_detection: Detection cap for the speed limit difference feature.
        acceleration: Acceleration of the agent (m/s²).
        acceleration_weight: Weight for the acceleration feature.
        acceleration_detection: Detection cap for the acceleration feature.
        deceleration: Deceleration of the agent (m/s²).
        deceleration_weight: Weight for the deceleration feature.
        deceleration_detection: Detection cap for the deceleration feature.
        jerk: Jerk of the agent (m/s³).
        jerk_weight: Weight for the jerk feature.
        jerk_detection: Detection cap for the jerk feature.
        waiting_period: Waiting period at a conflict point (s).
        waiting_period_weight: Weight for the waiting period feature.
        waiting_period_detection: Detection cap for the waiting period feature.
        trajectory_type: Trajectory type of the agent.
        trajectory_type_weight: Weight for the trajectory type feature.
        kalman_difficulty: Kalman prediction difficulty of the agent.
        kalman_difficulty_weight: Weight for the Kalman difficulty feature.
        kalman_difficulty_detection: Detection cap for the Kalman difficulty feature.

    Returns:
        The aggregated score for the agent.
    """
    return (
        speed_weight * min(speed_detection, speed)
        + speed_limit_diff_weight * min(speed_limit_diff_detection, speed_limit_diff)
        + acceleration_weight * min(acceleration_detection, acceleration)
        + deceleration_weight * min(deceleration_detection, deceleration)
        + jerk_weight * min(jerk_detection, jerk)
        + waiting_period_weight * min(waiting_period_detection, waiting_period)
        + trajectory_type_weight * TRAJECTORY_TYPE_WEIGHTS[trajectory_type]
        + kalman_difficulty_weight * min(kalman_difficulty_detection, max(kalman_difficulty, 0.0))
    )


INDIVIDUAL_SCORE_FUNCTIONS: dict[str, object] = {
    "simple": simple_individual_score,
}


def simple_interaction_score(
    collision: float = 0.0,
    collision_weight: float = 1.0,
    collision_detection: float = 1.0,
    mttcp: float = float("inf"),
    mttcp_weight: float = 1.0,
    mttcp_detection: float = 1.0,
    thw: float = float("inf"),
    thw_weight: float = 1.0,
    thw_detection: float = 1.0,
    ttc: float = float("inf"),
    ttc_weight: float = 1.0,
    ttc_detection: float = 1.0,
    drac: float = 0.0,
    drac_weight: float = 1.0,
    drac_detection: float = 1.0,
) -> float:
    """Aggregate a simple interaction score for an AD agent pair from weighted feature values.

    Time-based metrics (mttcp, thw, ttc) are inverted so that smaller values (more dangerous) produce larger scores.

    Args:
        collision: Collision indicator (1 if collision occurred, else 0).
        collision_weight: Weight for the collision feature.
        collision_detection: Detection cap for the collision feature.
        mttcp: Minimum time to closest point (s); infinite if no conflict.
        mttcp_weight: Weight for the mttcp feature.
        mttcp_detection: Detection cap applied to ``1 / mttcp``.
        thw: Time headway (s); infinite if not applicable.
        thw_weight: Weight for the thw feature.
        thw_detection: Detection cap applied to ``1 / thw``.
        ttc: Time to collision (s); infinite if agents not closing.
        ttc_weight: Weight for the ttc feature.
        ttc_detection: Detection cap applied to ``1 / ttc``.
        drac: Deceleration rate to avoid collision (m/s²).
        drac_weight: Weight for the drac feature.
        drac_detection: Detection cap for the drac feature.

    Returns:
        The aggregated score for the agent pair.
    """
    inv_mttcp = 1.0 / (mttcp + EPSILON)
    inv_thw = 1.0 / (thw + EPSILON)
    inv_ttc = 1.0 / (ttc + EPSILON)
    return (
        collision_weight * min(collision_detection, collision)
        + mttcp_weight * min(mttcp_detection, inv_mttcp)
        + thw_weight * min(thw_detection, inv_thw)
        + ttc_weight * min(ttc_detection, inv_ttc)
        + drac_weight * min(drac_detection, drac)
    )


INTERACTION_SCORE_FUNCTIONS: dict[str, object] = {
    "simple": simple_interaction_score,
}
