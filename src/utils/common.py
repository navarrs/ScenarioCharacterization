import colorlog
import logging
import os

from omegaconf import DictConfig

EPS = 1e-6
SUPPORTED_SCENARIO_TYPES = ["gt"]

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
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s[%(levelname)s]%(reset)s %(name)s '
            '(%(filename)s:%(lineno)d): %(message)s',
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
