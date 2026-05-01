import logging

import colorlog
from pytorch_lightning.utilities import rank_zero_only


def get_pylogger(name: str = __name__, *, use_rank_zero_only: bool = True) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logger = logging.getLogger(name)

    if use_rank_zero_only:
        # this ensures all logging levels get marked with the rank zero decorator
        # otherwise logs would get multiplied for each GPU process in multi-GPU setup
        logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
        for level in logging_levels:
            setattr(logger, level, rank_zero_only(getattr(logger, level)))
    else:
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
