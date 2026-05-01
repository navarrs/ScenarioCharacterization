from characterization.utils.common import SpeedUnits, TrajectoryType, XYZScale

MILLION = 1e6
EPSILON = 1e-6
LARGE_VALUE = 1e6
LARGE_FLOAT = 1e10
POSITION_DIMS = [2, 3]
MIN_VALID_POINTS = 2

KM_TO_HM = 10.0
KM_TO_M = 1000.0
HM_TO_M = 100.0
FT_TO_KM = 0.0003048
FT_TO_HM = 0.003048
FT_TO_M = 0.3048

KNOTS_TO_MS: float = 0.514444  # 1 knot = 0.514444 m/s
MPH_TO_MS: float = 0.44704  # 1 mph = 0.44704 m/s

AGENT_DIFFICULTY = {"easy": [0, 500], "medium": [500, 1250], "hard": [1250, LARGE_FLOAT]}

SCALE_FACTOR_FROM_KM = {
    XYZScale.KM: 1.0,
    XYZScale.HM: KM_TO_HM,
    XYZScale.M: KM_TO_M,
}

SCALE_FACTOR_TO_M = {
    XYZScale.KM: KM_TO_M,
    XYZScale.HM: HM_TO_M,
    XYZScale.M: 1.0,
}

SPEED_TO_MS = {
    SpeedUnits.KNOTS: KNOTS_TO_MS,
    SpeedUnits.MS: 1.0,
    SpeedUnits.MPH: MPH_TO_MS,
}

# Weights for different trajectory types are loosely set based on Figure 3 (a) of https://arxiv.org/pdf/2403.15098
# Weight per class is set as (100% - class frequency %) * 0.10
TRAJECTORY_TYPE_WEIGHTS = {
    TrajectoryType.TYPE_UNSET: 0.0,
    # Stationary agents correspond to less than 10% of the data
    TrajectoryType.TYPE_STATIONARY: 9.0,
    # Straight-moving agents correspond to ~50% of the data.
    TrajectoryType.TYPE_STRAIGHT: 5.0,
    # Straight-right agents correspond less than ~10% of the data.
    TrajectoryType.TYPE_STRAIGHT_RIGHT: 9.0,
    # Straight-left agents correspond to ~10% of the data.
    TrajectoryType.TYPE_STRAIGHT_LEFT: 9.0,
    # Right-turn agents correspond to less than 20% of the data.
    TrajectoryType.TYPE_RIGHT_TURN: 8.0,
    # Left-turn agents correspond to less than 20% of the data.
    TrajectoryType.TYPE_LEFT_TURN: 8.0,
    # Right-U-turn agents correspond to less than 10% of the data.
    TrajectoryType.TYPE_RIGHT_U_TURN: 9.0,
    # Left-U-turn agents correspond to less than 10% of the data.
    TrajectoryType.TYPE_LEFT_U_TURN: 9.0,
}
