import numpy as np
import logging
import os

import colorlog
from omegaconf import DictConfig

EPS = 1e-6
SUPPORTED_SCENARIO_TYPES = ["gt"]


def compute_dists_to_conflict_points(conflict_points: np.ndarray, trajectories: np.ndarray) -> np.ndarray:
    """Compute Distances to conflict points for all trajectories.
    Args:
        conflict_points (np.ndarray): The conflict points in shape (num_conflict_points, 3).
        trajectories (np.ndarray): The trajectories of the agents in shape (num_agents, num_time_steps, 3).
    Returns:
    """

    diff = conflict_points[None, None, :] - trajectories[:, :, None, :]
    return np.linalg.norm(diff, axis=-1)  # shape (num_agents, num_time_steps, num_conflict_points)


def make_output_paths(cfg: DictConfig) -> None:
    """
    Create output paths based on the configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        EasyDict: Dictionary containing output paths.
    """
    os.makedirs(cfg.paths.cache_path, exist_ok=True)

    for path in cfg.paths.output_paths.values():
        os.makedirs(path, exist_ok=True)


def get_logger(name=__name__):
    """
    Create a logger with colorized output for better readability.
    Args:
        name (str): Name of the logger. Defaults to the module's name.
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
