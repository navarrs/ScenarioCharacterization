import logging
import os
import pickle  # nosec B403
from enum import Enum

import colorlog
import numpy as np
from omegaconf import DictConfig, OmegaConf
from characterization.utils.schemas.types import AgentType

EPS = 1e-6
SUPPORTED_SCENARIO_TYPES = ["gt", "ho"]


class InteractionStatus(Enum):
    UNKNOWN = -1
    COMPUTED_OK = 0
    PARTIAL_INVALID_HEADING = 1
    MASK_NOT_VALID = 2
    DISTANCE_TOO_FAR = 3
    STATIONARY = 4


def compute_dists_to_conflict_points(conflict_points: np.ndarray, trajectories: np.ndarray) -> np.ndarray:
    """Computes distances from agent trajectories to conflict points.

    Args:
        conflict_points (np.ndarray): Array of conflict points (shape: [num_conflict_points, 3]).
        trajectories (np.ndarray): Array of agent trajectories (shape: [num_agents, num_time_steps, 3]).

    Returns:
        np.ndarray: Distances from each agent at each timestep to each conflict point
            (shape: [num_agents, num_time_steps, num_conflict_points]).
    """
    diff = conflict_points[None, None, :] - trajectories[:, :, None, :]
    return np.linalg.norm(diff, axis=-1)  # shape (num_agents, num_time_steps, num_conflict_points)


class InteractionAgent:
    """Class representing an agent for interaction feature computation."""

    def __init__(self):
        """Initializes an InteractionAgent and resets all attributes."""
        self.reset()

    @property
    def position(self) -> np.ndarray | None:
        """np.ndarray: The positions of the agent over time (shape: [T, 2])."""
        return self._position

    @position.setter
    def position(self, value: np.ndarray | None) -> None:
        """Sets the positions of the agent.

        Args:
            value (np.ndarray): The positions of the agent over time (shape: [T, 2]).
        """
        if value is not None:
            self._position = np.asarray(value, dtype=np.float32)
        else:
            self._position = None

    @property
    def speed(self) -> np.ndarray | None:
        """np.ndarray: The velocities of the agent over time (shape: [T,])."""
        return self._speed

    @speed.setter
    def speed(self, value: np.ndarray | None) -> None:
        """Sets the velocities of the agent.

        Args:
            value (np.ndarray): The velocities of the agent over time (shape: [T,]).
        """
        if value is not None:
            self._speed = np.asarray(value, dtype=np.float32)
        else:
            self._speed = None

    @property
    def heading(self) -> np.ndarray | None:
        """np.ndarray: The headings of the agent over time (shape: [T,])."""
        return self._heading

    @heading.setter
    def heading(self, value: np.ndarray | None) -> None:
        """Sets the headings of the agent.

        Args:
            value (np.ndarray): The headings of the agent over time (shape: [T,]).
        """
        if value is not None:
            self._heading = np.asarray(value, dtype=np.float32)
        else:
            self._heading = None

    @property
    def length(self) -> np.ndarray | None:
        """np.ndarray or None: The lengths of the agent over time (shape: [T,])."""
        return self._length

    @length.setter
    def length(self, value: np.ndarray | None) -> None:
        """Sets the lengths of the agent.

        Args:
            value (np.ndarray): The lengths of the agent over time (shape: [T,]).
        """
        if value is not None:
            self._length = np.asarray(value, dtype=np.float32)
        else:
            self._length = None

    @property
    def width(self) -> np.ndarray | None:
        """np.ndarray or None: The widths of the agent over time (shape: [T,])."""
        return self._width

    @width.setter
    def width(self, value: np.ndarray | None) -> None:
        """Sets the widths of the agent.

        Args:
            value (np.ndarray): The widths of the agent over time (shape: [T,]).
        """
        if value is not None:
            self._height = np.asarray(value, dtype=np.float32)
        else:
            self._width = None

    @property
    def height(self) -> np.ndarray | None:
        """np.ndarray or None: The heights of the agent over time (shape: [T,])."""
        return self._height

    @height.setter
    def height(self, value: np.ndarray | None) -> None:
        """Sets the heights of the agent.

        Args:
            value (np.ndarray): The heights of the agent over time (shape: [T,]).
        """
        if value is not None:
            self._height = np.asarray(value, dtype=np.float32)
        else:
            self._height = None

    @property
    def agent_type(self) -> AgentType | None:
        """str: The type of the agent."""
        return self._agent_type

    @agent_type.setter
    def agent_type(self, value: AgentType | None) -> None:
        """Sets the type of the agent.

        Args:
            value (str): The type of the agent.
        """
        if value is not None:
            self._agent_type = value
        else:
            self._agent_type = None

    @property
    def is_stationary(self) -> bool | None:
        """Bool or None: Whether the agent is stationary (True/False), or None if unknown."""
        if self._speed is None:
            self._is_stationary = None
        else:
            self._is_stationary = self.speed.mean() < self._stationary_speed
        return self._is_stationary

    @property
    def stationary_speed(self) -> float:
        """float: The speed threshold below which the agent is considered stationary."""
        return self._stationary_speed

    @stationary_speed.setter
    def stationary_speed(self, value: float | None) -> None:
        """Sets the stationary speed threshold.

        Args:
            value (float): The speed threshold below which the agent is considered stationary.
        """
        if value is not None:
            self._stationary_speed = float(value)
        else:
            self._stationary_speed = 0.1

    @property
    def in_conflict_point(self) -> bool:
        """bool: Whether the agent is in a conflict point."""
        if self._dists_to_conflict is None:
            self._in_conflict_point = False
        else:
            self._in_conflict_point = np.any(
                self._dists_to_conflict <= self._agent_to_conflict_point_max_distance,
            ).__bool__()
        return self._in_conflict_point

    @property
    def agent_to_conflict_point_max_distance(self) -> float | None:
        """float: The maximum distance to a conflict point."""
        return self._agent_to_conflict_point_max_distance

    @agent_to_conflict_point_max_distance.setter
    def agent_to_conflict_point_max_distance(self, value: float | None) -> None:
        """Sets the maximum distance to a conflict point.

        Args:
            value (float): The maximum distance to a conflict point.
        """
        if value is not None:
            self._agent_to_conflict_point_max_distance = float(value)
        else:
            self._agent_to_conflict_point_max_distance = 0.5  # Default value

    @property
    def dists_to_conflict(self) -> np.ndarray | None:
        """np.ndarray: The distances to conflict points (shape: [T,])."""
        return self._dists_to_conflict

    @dists_to_conflict.setter
    def dists_to_conflict(self, value: np.ndarray | None) -> None:
        """Sets the distances to conflict points.

        Args:
            value (np.ndarray | None): The distances to conflict points (shape: [T,]).
        """
        if value is not None:
            self._dists_to_conflict = np.asarray(value, dtype=np.float32)
        else:
            self._dists_to_conflict = None

    @property
    def lane(self) -> np.ndarray | None:
        """np.ndarray or None: The lane of the agent, if available."""
        return self._lane

    @lane.setter
    def lane(self, value: np.ndarray | None) -> None:
        """Sets the lane of the agent.

        Args:
            value (np.ndarray or None): The lane of the agent, if available.
        """
        if value is not None:
            self._lane = np.asarray(value, dtype=np.float32)
        else:
            self._lane = None

    def reset(self) -> None:
        """Resets all agent attributes to their default values."""
        self._position = None
        self._speed = None
        self._heading = None
        self._dists_to_conflict = None
        self._stationary_speed = 0.1  # Default stationary speed threshold
        self._agent_to_conflict_point_max_distance = 0.5  # Default max distance to conflict point
        self._lane = None
        self._length = None
        self._width = None
        self._height = None
        self._agent_type = None


def make_output_paths(cfg: DictConfig) -> None:
    """Creates output directories as specified in the configuration.

    Args:
        cfg (DictConfig): Configuration dictionary containing output paths.

    Returns:
        None
    """
    os.makedirs(cfg.paths.cache_path, exist_ok=True)

    for path in cfg.paths.output_paths.values():
        os.makedirs(path, exist_ok=True)


def get_logger(name: str = __name__) -> logging.Logger:
    """Creates a logger with colorized output for better readability.

    Args:
        name (str, optional): Name of the logger. Defaults to the module's name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)s]%(reset)s %(name)s (%(filename)s:%(lineno)d): %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        ),
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def from_pickle(data_file: str) -> dict:  # pyright: ignore[reportMissingTypeArgument]
    """Loads data from a pickle file.

    Args:
        data_file (str): The path to the pickle file.

    Returns:
        dict: The loaded data.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} does not exist.")

    with open(data_file, "rb") as f:
        data = pickle.load(f)  # nosec B301

    return data


def to_pickle(output_path: str, input_data: dict, tag: str) -> None:  # pyright: ignore[reportMissingTypeArgument]
    """Saves data to a pickle file, merging with existing data if present.

    Args:
        output_path (str): Directory where the pickle file will be saved.
        input_data (dict): The data to save.
        tag (str): The tag to use for the output file name.

    Returns:
        None
    """
    data = {}
    data_file = os.path.join(output_path, f"{tag}.pkl")
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            data = pickle.load(f)  # nosec B301

    scenario_id_data = data.get("scenario_id", None)
    if scenario_id_data is not None and scenario_id_data != input_data["scenario_id"]:
        raise AttributeError("Mismatched scenario IDs when merging pickle data.")

    # NOTE: with current ScenarioScores and ScenarioFeatures implementation, computing interaction and individual
    # features will cause overrides. Need to address this better in the future.
    for key, value in input_data.items():
        if value is None:
            continue
        # if key in data and data[key] is not None:
        #     continue
        data[key] = value

    with open(data_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def print_config(cfg: DictConfig, theme: str = "monokai") -> None:
    """Prints the configuration in a readable format.

    Args:
        cfg (DictConfig): Configuration dictionary to print.

    Returns:
        None
    """
    from rich.console import Console
    from rich.syntax import Syntax

    yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
    console = Console()
    syntax = Syntax(yaml_str, "yaml", theme=theme, word_wrap=True)
    console.print(syntax)
