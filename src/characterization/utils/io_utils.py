import logging
import os
import pickle  # nosec B403

import colorlog
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.syntax import Syntax


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
        error_message = f"Data file {data_file} does not exist."
        raise FileNotFoundError(error_message)

    with open(data_file, "rb") as f:
        return pickle.load(f)  # nosec B301


def to_pickle(output_path: str, input_data: dict, tag: str, *, overwrite: bool = True) -> None:  # pyright: ignore[reportMissingTypeArgument]
    """Saves data to a pickle file, merging with existing data if present.

    Args:
        output_path (str): Directory where the pickle file will be saved.
        input_data (dict): The data to save.
        tag (str): The tag to use for the output file name.
        overwrite (bool, optional): Whether to overwrite existing data. Defaults to True.

    Returns:
        None
    """
    data = {}
    data_file = os.path.join(output_path, f"{tag}.pkl")
    if overwrite:
        with open(data_file, "wb") as f:
            pickle.dump(input_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            data = pickle.load(f)  # nosec B301

    scenario_id_data = data.get("scenario_id", None)
    if scenario_id_data is not None and scenario_id_data != input_data["scenario_id"]:
        error_message = "Mismatched scenario IDs when merging pickle data."
        raise AttributeError(error_message)

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
        theme (str, optional): Theme for syntax highlighting. Defaults to "monokai".

    Returns:
        None
    """
    yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
    console = Console()
    syntax = Syntax(yaml_str, "yaml", theme=theme, word_wrap=True)
    console.print(syntax)
