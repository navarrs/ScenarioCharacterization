import numpy as np
from shapely import LineString

from characterization.utils.common import MIN_VALID_POINTS, SMALL_EPS, InteractionAgent
from characterization.utils.io_utils import get_logger

logger = get_logger(__name__)


def is_sharing_lane(lane_i: np.ndarray | None, lane_j: np.ndarray | None) -> bool:  # noqa: ARG001
    """Checks if two agents are sharing the same lane.

    Args:
        lane_i (np.ndarray or None): The lane of the first agent.
        lane_j (np.ndarray or None): The lane of the second agent.

    Returns:
        bool: True if both agents are sharing the same lane, False otherwise.
    """
    # if lane_i is None or lane_j is None:
    #     return False
    # lane_i = lane_i[np.isfinite(lane_i)]
    # lane_j = lane_j[np.isfinite(lane_j)]
    # return np.isin(lane_i, lane_j).any()
    return True


def find_valid_headings(
    agent_i: InteractionAgent,
    agent_j: InteractionAgent,
    heading_threshold: float = 0.1,
) -> np.ndarray:
    """Checks if the headings of two agents are within a specified threshold.

    Args:
        agent_i (InteractionAgent): The first agent.
        agent_j (InteractionAgent): The second agent.
        heading_threshold (float): Threshold for considering headings as valid. Heading are assumed to be in degrees.

    Returns:
        bool: True if the headings of both agents are within the threshold, False otherwise.
    """
    valid_headings = np.empty(shape=(0,), dtype=bool)
    if agent_i.heading is None or agent_j.heading is None:
        return valid_headings
    if not is_sharing_lane(agent_i.lane, agent_j.lane):
        return valid_headings

    heading_diff = np.abs((agent_j.heading - agent_i.heading + 540) % 360 - 180)
    # return valid headings
    return np.where(heading_diff <= heading_threshold)[0]


def find_leading_agent(
    agent_i: InteractionAgent, agent_j: InteractionAgent, mask: np.ndarray | None = None, angle_threshold: float = 90
) -> np.ndarray:
    """Determines which agent is leading based on their positions and headings.

    Args:
        agent_i (InteractionAgent): The first agent.
        agent_j (InteractionAgent): The second agent.
        mask (np.ndarray or None): Optional mask to filter positions.
        angle_threshold (float): Angle threshold in degrees to determine if one agent is behind the other.

    Returns:
        int: 0 if agent_i is leading, 1 if agent_j is leading.
    """
    position_i, position_j = agent_i.position, agent_j.position
    heading_i = agent_i.heading
    if mask is not None:
        position_i = position_i[mask]
        position_j = position_j[mask]
        heading_i = heading_i[mask]

    # Compute i-to-j angles based on positions and headings
    x_i, y_i = position_i[:, 0], position_i[:, 1]
    x_j, y_j = position_j[:, 0], position_j[:, 1]
    heading_i = np.deg2rad(heading_i)

    # Vector i to j
    vector_to_j = np.array([x_j - x_i, y_j - y_i])
    angle_to_j = np.arctan2(vector_to_j[1], vector_to_j[0])  # Angle in radians

    # Adjust the angle difference to be between -π and π
    angle_ij = angle_to_j - heading_i
    # Wrap the angle difference to be between -π and π
    angle_ij = np.rad2deg((angle_ij + np.pi) % (2 * np.pi) - np.pi)

    # Check if position_j is "behind" position_i
    leading_agent = np.abs(angle_ij) > angle_threshold
    # 0 -> i is leading, 1 -> j is leading
    return (~leading_agent).astype(int)


def compute_separation(agent_i: InteractionAgent, agent_j: InteractionAgent) -> np.ndarray:
    """Computes the separation distance between two agents at each timestep.

    Args:
        agent_i (InteractionAgent): The first agent.
        agent_j (InteractionAgent): The second agent.

    Returns:
        np.ndarray: Array of separation distances between agent_i and agent_j at each timestep (shape: [T,]).
    """
    position_i, position_j = agent_i.position, agent_j.position
    return np.linalg.norm(position_i - position_j, axis=-1)


def compute_intersections(agent_i: InteractionAgent, agent_j: InteractionAgent) -> np.ndarray:
    """Computes whether two agents' trajectory segments intersect at each timestep.

    Args:
        agent_i (InteractionAgent): The first agent.
        agent_j (InteractionAgent): The second agent.

    Returns:
        np.ndarray: Boolean array indicating whether each segment of agent_i intersects with the corresponding segment
            of agent_j (shape: [T,]).
    """
    position_i, position_j = agent_i.position, agent_j.position
    if position_i.shape[0] < MIN_VALID_POINTS or position_j.shape[0] < MIN_VALID_POINTS:
        return np.zeros((position_i.shape[0],), dtype=bool)

    segments_i = np.stack([position_i[:-1], position_i[1:]], axis=1)
    segments_j = np.stack([position_j[:-1], position_j[1:]], axis=1)
    segments_i = [LineString(x) for x in segments_i]
    segments_j = [LineString(x) for x in segments_j]

    intersections = [x.intersects(y) for x, y in zip(segments_i, segments_j, strict=False)]
    # Make it consistent with the number of timesteps
    return np.array([intersections[0]] + intersections, dtype=bool)  # noqa: RUF005


def compute_mttcp(
    agent_i: InteractionAgent,
    agent_j: InteractionAgent,
    agent_to_agent_max_distance: float = 0.5,
) -> np.ndarray:
    """Computes the minimum time to conflict point (mTTCP) between two agents.

                                 |  𝚫xi(t)     𝚫xj(t) |
        𝚫TTCP  =       min       | ------  -  ------  |
                  t in {0, tcp}  |  𝚫vi(t)     𝚫vj(t) |

    The mTTCP is defined as the minimum absolute difference in time for each agent to reach a conflict point,
    for all timesteps where the agents are within a specified distance threshold.

    Args:
        agent_i (InteractionAgent): The first agent.
        agent_j (InteractionAgent): The second agent.
        agent_to_agent_max_distance (float): The maximum distance between agents to consider for mTTCP.

    Returns:
        np.ndarray: An array of mTTCP values for each timestep (shape: [N,]), or [np.inf] if no valid pairs are found.
    """
    position_i, position_j = agent_i.position, agent_j.position
    vel_i, vel_j = agent_i.speed, agent_j.speed

    # T, 2 -> T, T
    dists = np.linalg.norm(position_i[:, None, :] - position_j, axis=-1)
    i_idx, _ = np.where(dists <= agent_to_agent_max_distance)

    _, i_unique = np.unique(i_idx, return_index=True)
    ti = i_idx[i_unique]
    if len(ti) == 0:
        return np.array([np.inf], dtype=np.float32)

    conflict_points = position_i[ti]
    mttcp = np.inf * np.ones(conflict_points.shape[0])

    cp_to_position_i = np.linalg.norm(position_i - conflict_points[:, None], axis=-1)
    cp_to_position_j = np.linalg.norm(position_j - conflict_points[:, None], axis=-1)
    tj = cp_to_position_j.argmin(axis=-1)

    t_min = np.minimum(ti, tj) + 1
    for n, t in enumerate(t_min):
        # Compute the time to conflict point for each agent
        ttcp_i = cp_to_position_i[n, :t] / vel_i[:t]  # Shape: (num. conflict points, 0 to t)
        ttcp_j = cp_to_position_j[n, :t] / vel_j[:t]

        # Calculate the absolute difference in time to conflict point
        ttcp = np.abs(ttcp_i - ttcp_j)

        # Update the minimum mTTCP
        mttcp[n] = ttcp.min()

    return mttcp


def compute_thw(
    agent_i: InteractionAgent,
    agent_j: InteractionAgent,
    leading_agent: np.ndarray,
    valid_headings: np.ndarray | None = None,
    *,
    return_only_finite: bool = True,
) -> np.ndarray:
    """Computes the following leader-follower interaction measurements.

        Time Headway (THW):
        -------------------
            TWH = d / v_f

        where d is the gap between the leader and the follower, and v_f is the speed of the follower.

    Args:
        agent_i (InteractionAgent): The first agent.
        agent_j (InteractionAgent): The second agent.
        leading_agent (np.ndarray): Array indicating which agent is leading (0 for agent_i, 1 for agent_j).
        valid_headings (np.ndarray | None): Optional mask to filter valid headings.
        return_only_finite (bool): Whether to return only finite THW values.

    Returns:
        np.ndarray: Array of time headway values for each timestep (shape: [T,]).
    """
    position_i, position_j = np.linalg.norm(agent_i.position, axis=-1), np.linalg.norm(agent_j.position, axis=-1)
    speed_i, speed_j = agent_i.speed, agent_j.speed
    length_i, length_j = agent_i.length, agent_j.length
    if valid_headings is not None:
        position_i = position_i[valid_headings]
        speed_i = speed_i[valid_headings]
        length_i = length_i[valid_headings]
        position_j = position_j[valid_headings]
        speed_j = speed_j[valid_headings]
        length_j = length_j[valid_headings]

    thw = np.full(position_i.shape[0], np.inf, dtype=np.float32)

    # NOTE: this assumes the leader value is correctly computed. Need to still verify this.
    # ...where i is the agent ahead
    i_idx = np.where(leading_agent == 0)[0]
    if len(i_idx) > 0:
        d_i = position_i[i_idx] - position_j[i_idx] - length_i[i_idx]
        thw[i_idx] = d_i / (speed_j[i_idx] + SMALL_EPS)

    # ...where j is the agent ahead
    j_idx = np.where(leading_agent == 1)[0]
    if len(j_idx) > 0:
        d_j = position_j[j_idx] - position_i[j_idx] - length_j[j_idx]
        thw[j_idx] = d_j / (speed_i[j_idx] + SMALL_EPS)

    if return_only_finite:
        thw = thw[np.isfinite(thw)]
        if thw.size == 0:
            thw = np.array([np.inf], dtype=np.float32)
    return np.abs(thw)


def compute_ttc(
    agent_i: InteractionAgent,
    agent_j: InteractionAgent,
    leading_agent: np.ndarray,
    valid_headings: np.ndarray | None = None,
    *,
    return_only_finite: bool = True,
) -> np.ndarray:
    """Computes the following leader-follower interaction measurement.

        Time-to-Collision (TTC):
        ------------------------
                        d
            TTC = ---------------  forall v_i > v_j
                     v_i - v_j

        where d is the gap between the leader and the follower, and v_i and v_j are the speeds of the follower and the
        leader, and where the follower's speed is higher than the leader's speed.

    Args:
        agent_i (InteractionAgent): The first agent.
        agent_j (InteractionAgent): The second agent.
        leading_agent (np.ndarray): Array indicating which agent is leading (0 for agent_i, 1 for agent_j).
        valid_headings (np.ndarray | None): Optional mask to filter valid headings.
        return_only_finite (bool): Whether to return only finite TTC values.

    Returns:
        np.ndarray: Array of time-to-collision values for each timestep (shape: [T,]).
    """
    position_i, position_j = np.linalg.norm(agent_i.position, axis=-1), np.linalg.norm(agent_j.position, axis=-1)
    speed_i, speed_j = agent_i.speed, agent_j.speed
    length_i, length_j = agent_i.length, agent_j.length
    if valid_headings is not None:
        position_i = position_i[valid_headings]
        speed_i = speed_i[valid_headings]
        length_i = length_i[valid_headings]
        position_j = position_j[valid_headings]
        speed_j = speed_j[valid_headings]
        length_j = length_j[valid_headings]

    ttc = np.full(position_i.shape[0], np.inf, dtype=np.float32)
    # ...where i is the agent ahead and j's speed is higher
    i_leads = np.where(leading_agent == 0)[0]
    j_faster = np.where(speed_j > speed_i)[0]
    i_idx = np.intersect1d(i_leads, j_faster)
    if len(i_idx) > 0:
        d_ij = position_j[i_idx] - position_i[i_idx] - length_i[i_idx]
        ttc[i_idx] = d_ij / (speed_j[i_idx] - speed_i[i_idx] + SMALL_EPS)

    # ...where j is the agent ahead and i's speed is higher
    j_leads = np.where(leading_agent == 1)[0]
    i_faster = np.where(speed_i > speed_j)[0]
    j_idx = np.intersect1d(j_leads, i_faster)
    if len(j_idx) > 0:
        d_ji = position_i[j_idx] - position_j[j_idx] - length_j[j_idx]
        ttc[j_idx] = d_ji / (speed_i[j_idx] - speed_j[j_idx] + SMALL_EPS)

    if return_only_finite:
        ttc = ttc[np.isfinite(ttc)]
        if ttc.size == 0:
            ttc = np.array([np.inf], dtype=np.float32)
    return np.abs(ttc)


def compute_drac(
    agent_i: InteractionAgent,
    agent_j: InteractionAgent,
    leading_agent: np.ndarray,
    valid_headings: np.ndarray | None = None,
    *,
    return_nonzero_only: bool = True,
) -> np.ndarray:
    """Computes the following leader-follower interaction measurement.

        Deceleration Rate to Avoid a Crash (DRAC):
        -----------------------------------------
                (v_j - v_i) ** 2
        DRAC = ------------------
                      2 d
        the average delay of a road user to avoid an accident at given velocities and distance between vehicles,
        where i is the leader and j is the follower.

    Args:
        agent_i (InteractionAgent): The first agent.
        agent_j (InteractionAgent): The second agent.
        leading_agent (np.ndarray): Array indicating which agent is leading (0 for agent_i, 1 for agent_j).
        valid_headings (np.ndarray | None): Optional mask to filter valid headings.
        return_nonzero_only (bool): Whether to return only non-zero DRAC values.

    Returns:
        np.ndarray: Array of time-to-collision values for each timestep (shape: [T,]).
    """
    position_i, position_j = np.linalg.norm(agent_i.position, axis=-1), np.linalg.norm(agent_j.position, axis=-1)
    speed_i, speed_j = agent_i.speed, agent_j.speed
    length_i, length_j = agent_i.length, agent_j.length
    if valid_headings is not None:
        position_i = position_i[valid_headings]
        speed_i = speed_i[valid_headings]
        length_i = length_i[valid_headings]
        position_j = position_j[valid_headings]
        speed_j = speed_j[valid_headings]
        length_j = length_j[valid_headings]

    drac = np.full(position_i.shape[0], 0.0, dtype=np.float32)

    # ...where i is the agent ahead and j's speed is higher
    i_leads = np.where(leading_agent == 0)[0]
    j_faster = np.where(speed_j > speed_i)[0]
    i_idx = np.intersect1d(i_leads, j_faster)
    if len(i_idx) > 0:
        d_ij = np.abs(position_j[i_idx] - position_i[i_idx] - length_i[i_idx])
        v_ji = speed_j[i_idx] - speed_i[i_idx]
        drac[i_idx] = (v_ji**2) / (2 * d_ij + SMALL_EPS)

    # ...where j is the agent ahead and i's speed is higher
    j_leads = np.where(leading_agent == 1)[0]
    i_faster = np.where(speed_i > speed_j)[0]
    j_idx = np.intersect1d(j_leads, i_faster)
    if len(j_idx) > 0:
        d_ji = np.abs(position_i[j_idx] - position_j[j_idx] - length_j[j_idx])
        v_ij = speed_i[j_idx] - speed_j[j_idx]
        drac[j_idx] = (v_ij**2) / (2 * d_ji + SMALL_EPS)

    # TODO: account for the leader deceleration rate
    if return_nonzero_only:
        drac = drac[np.isfinite(drac) & (drac > 0)]
        if drac.size == 0:
            drac = np.array([0.0], dtype=np.float32)
    return drac
