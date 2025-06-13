import logging
import os
import pickle  # nosec B403
from enum import Enum

import colorlog
import numpy as np
from omegaconf import DictConfig

EPS = 1e-6
SUPPORTED_SCENARIO_TYPES = ["gt"]


class InteractionStatus(Enum):
    UNKNOWN = -1
    COMPUTED_OK = 0
    MASK_NOT_VALID = 1
    AGENT_DISTANCE_TOO_FAR = 2
    AGENTS_STATIONARY = 3


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


def get_logger(name=__name__):
    """Creates a logger with colorized output for better readability.

    Args:
        name (str, optional): Name of the logger. Defaults to the module's name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)s]%(reset)s %(name)s " "(%(filename)s:%(lineno)d): %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def from_pickle(data_file: str) -> dict:
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


def to_pickle(output_path: str, input_data: dict, tag: str) -> None:
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

    # NOTE: with current ScenarioScores and ScenarioFeatures implementation, computing interaction and individual
    # features will cause overrides. Need to address this better in the future.
    for key, value in input_data.items():
        if key in data and data[key] is not None:
            continue
        data[key] = value

    with open(data_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
