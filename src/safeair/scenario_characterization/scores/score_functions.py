"""Scoring functions for individual agents and interaction pairs."""

import numpy as np

from characterization.domains.aviation.scenario_types import INVALID_STATE_VALUE, RawAgentTrajectory
from characterization.utils.constants import EPSILON
from safeair.scenario_characterization.common import TrajectoryType
from safeair.scenario_characterization.features.individual_utils import estimate_kalman_filter

TRAJECTORY_TYPE_WEIGHTS: dict[str, float] = {
    TrajectoryType.STATIONARY: 8.0,
    TrajectoryType.STRAIGHT: 0.5,
    TrajectoryType.TURNING: 1.0,
    TrajectoryType.U_TURN: 1.0,
}


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
    waiting_period: float = 0.0,
    waiting_period_weight: float = 1.0,
    waiting_period_detection: float = 1.0,
    trajectory_type: str = TrajectoryType.STATIONARY,
    trajectory_type_weight: float = 0.1,
    kalman_difficulty: float = 0.0,
    kalman_difficulty_weight: float = 1.0,
    kalman_difficulty_detection: float = 1.0,
) -> float:
    """Aggregate a simple score for an individual agent from weighted feature values.

    Each feature's contribution is capped at its detection threshold before being multiplied by its weight. Detection
    thresholds are loosely inspired by https://arxiv.org/abs/2202.07438.

    Args:
        speed: Agent speed in m/s.
        speed_weight: Weight for the speed feature.
        speed_detection: Detection cap for the speed feature.
        acceleration: Positive acceleration in m/s².
        acceleration_weight: Weight for the acceleration feature.
        acceleration_detection: Detection cap for the acceleration feature.
        deceleration: Deceleration magnitude in m/s².
        deceleration_weight: Weight for the deceleration feature.
        deceleration_detection: Detection cap for the deceleration feature.
        waiting_period: Time spent stationary near a conflict point in seconds.
        waiting_period_weight: Weight for the waiting period feature.
        waiting_period_detection: Detection cap for the waiting period feature.
        trajectory_type: Trajectory type string (STATIONARY / STRAIGHT / TURNING / U_TURN).
        trajectory_type_weight: Weight for the trajectory type feature.
        kalman_difficulty: Normalized Kalman prediction error.
        kalman_difficulty_weight: Weight for the Kalman difficulty feature.
        kalman_difficulty_detection: Detection cap for the Kalman difficulty feature.

    Returns:
        Weighted aggregate score for the agent.
    """
    return (
        speed_weight * min(speed_detection, speed)
        + acceleration_weight * min(acceleration_detection, acceleration)
        + deceleration_weight * min(deceleration_detection, deceleration)
        + waiting_period_weight * min(waiting_period_detection, waiting_period)
        + trajectory_type_weight * TRAJECTORY_TYPE_WEIGHTS.get(trajectory_type, 0.0)
        + kalman_difficulty_weight * min(kalman_difficulty_detection, max(kalman_difficulty, 0.0))
    )


INDIVIDUAL_SCORE_REGISTRY: dict[str, object] = {
    "simple": simple_individual_score,
}


def simple_interaction_score(
    loss_of_separation: float = 0.0,
    loss_of_separation_weight: float = 1.0,
    loss_of_separation_detection: float = 1.0,
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
    """Aggregate a simple score for an agent pair from weighted interaction feature values.

    Time-based metrics (mttcp, thw, ttc) are transformed to reciprocals so that smaller values (more dangerous)
    produce larger scores. An infinite value contributes approximately zero (1 / (inf + eps) ≈ 0).

    Args:
        loss_of_separation: Loss of separation indicator or fraction in [0, 1].
        loss_of_separation_weight: Weight for the loss of separation feature.
        loss_of_separation_detection: Detection cap for the loss of separation feature.
        mttcp: Minimum time to conflict point in seconds (inf = no conflict detected).
        mttcp_weight: Weight for the mttcp feature.
        mttcp_detection: Detection cap applied to ``1 / mttcp``.
        thw: Time headway in seconds (inf = not applicable).
        thw_weight: Weight for the thw feature.
        thw_detection: Detection cap applied to ``1 / thw``.
        ttc: Time to collision in seconds (inf = agents not closing).
        ttc_weight: Weight for the ttc feature.
        ttc_detection: Detection cap applied to ``1 / ttc``.
        drac: Deceleration rate to avoid collision in m/s².
        drac_weight: Weight for the drac feature.
        drac_detection: Detection cap for the drac feature.

    Returns:
        Weighted aggregate score for the agent pair.
    """
    inv_mttcp = 1.0 / (mttcp + EPSILON)
    inv_thw = 1.0 / (thw + EPSILON)
    inv_ttc = 1.0 / (ttc + EPSILON)
    return (
        loss_of_separation_weight * min(loss_of_separation_detection, loss_of_separation)
        + mttcp_weight * min(mttcp_detection, inv_mttcp)
        + thw_weight * min(thw_detection, inv_thw)
        + ttc_weight * min(ttc_detection, inv_ttc)
        + drac_weight * min(drac_detection, drac)
    )


INTERACTION_SCORE_REGISTRY: dict[str, object] = {
    "simple": simple_interaction_score,
}


def compute_imputation_score(agent_trajectory: RawAgentTrajectory) -> float:
    """Compute an imputation score for the given agent sequence.

    Args:
        agent_trajectory: Agent sequence data.

    Returns:
        Imputation score.
    """
    if agent_trajectory.num_frames == 0:
        return 0.0
    return np.sum(agent_trajectory.valid == INVALID_STATE_VALUE)


def compute_kalman_score(
    agent_trajectory: RawAgentTrajectory,
    current_time_index: int,
    min_past_valid_points: int = 2,
) -> float:
    """Compute trajectory prediction difficulty using Kalman filter error.

    This function measures how difficult it is to predict an agent's future trajectory by comparing
    Kalman filter predictions against ground truth future positions. Higher scores indicate more
    unpredictable/difficult trajectories.

    Args:
        agent_trajectory: Trajectory data of the agent.
        current_time_index: Time index separating past and future data.
        min_past_valid_points: Minimum number of valid past points required for prediction.

    Returns:
        Normalized difficulty score. Higher values indicate more unpredictable trajectories.
        Returns 0.0 if insufficient data for prediction.
    """
    positions = agent_trajectory.xyz_position

    # Split data into past (observed) and future (ground truth) segments
    mask = agent_trajectory.valid.astype(bool).squeeze(-1)
    past_mask = mask[:current_time_index]
    future_mask = mask[current_time_index:]

    # Extract valid positions from each segment
    past_indices = np.where(past_mask)[0]  # original frame indices of valid past positions
    past_positions = positions[:current_time_index][past_mask]
    # Check if we have sufficient data for prediction
    if future_mask.sum() == 0 or past_positions.shape[0] < min_past_valid_points:
        return 0.0

    # Get the prediction target (last valid future position)
    last_valid_future_index = np.where(future_mask)[0][-1]
    total_future_steps = int(future_mask.shape[0])

    # Generate Kalman filter prediction
    predicted_position = estimate_kalman_filter(
        past_positions, last_valid_future_index + 1, min_valid_points=min_past_valid_points, frame_indices=past_indices
    )

    # Compute prediction error normalized by the total future window size
    prediction_target = positions[current_time_index:][future_mask][-1]
    prediction_error = np.linalg.norm(predicted_position - prediction_target).item()
    return prediction_error * (last_valid_future_index + 1) / total_future_steps


def compute_simple_agent_score(
    agent_trajectory: RawAgentTrajectory,
    current_time_index: int,
    *,
    is_valid: bool,
    min_past_valid_points: int = 2,
) -> float:
    """Compute a simple score based on the number of safe frames.

    Args:
        agent_trajectory: Trajectory data of the agent.
        current_time_index: Time index separating past and future data.
        is_valid: Indicates if the agent's data is valid.
        min_past_valid_points: Minimum number of valid past points required for scoring.

    Returns:
        Computed score for the agent.
    """
    if not is_valid:
        return -1.0
    imputation_score = compute_imputation_score(agent_trajectory)
    kalman_score = compute_kalman_score(agent_trajectory, current_time_index, min_past_valid_points)

    return imputation_score + kalman_score


SCORE_FUNCTIONS_REGISTRY = {"simple": compute_simple_agent_score}
