from collections.abc import Callable, Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from warnings import filterwarnings

import hydra
import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from rich.prompt import Prompt

from characterization.utils.logging_utils import get_pylogger

_LOGGER = get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:  # pyright: ignore[reportMissingTypeArgument]
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
           @utils.task_wrapper
           def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
               ...

               return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            _LOGGER.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise AssertionError(ex)  # noqa: B904

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            _LOGGER.info("Output dir: %s", cfg.paths.output_dir)

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb  # noqa: PLC0415

                if wandb.run:
                    _LOGGER.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiate loggers from config.

    Args:
        logger_cfg: Configuration parameters for the loggers to instantiate.

    Returns:
        List of instantiated loggers.
    """
    logger: list[Logger] = []

    if not logger_cfg:
        _LOGGER.warning("No logger configs found! Skipping...")
        return logger

    for lg_conf in logger_cfg.values():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            _LOGGER.info("Instantiating logger <%s>", lg_conf._target_)
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiate callbacks from config.

    Args:
        callbacks_cfg: Configuration parameters for the callbacks to instantiate.

    Returns:
        List of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        _LOGGER.warning("No callback configs found! Skipping..")
        return callbacks

    for cb_conf in callbacks_cfg.values():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            _LOGGER.info("Instantiating callback <%s>", cb_conf._target_)
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "scenario",
        "model",
        "dataset",
        "trainer",
        "extras",
    ),
    *,
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print content of DictConfig using Rich library and its tree structure.

    Args:
        cfg: Configuration composed by Hydra.
        print_order: Determines the order in which config components are printed.
        resolve: Whether to resolve reference fields of DictConfig.
        save_to_file: Whether to export config to the hydra output folder.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        (
            queue.append(field)
            if field in cfg
            else _LOGGER.warning("Field '<%s>' not found in config. Skipping it in config printing...", field)
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with Path(cfg.paths.log_path, "config_tree.log").open("w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def _enforce_tags(cfg: DictConfig, *, save_to_file: bool = False) -> None:
    """Prompt user to input tags from command line if no tags are provided in config.

    Args:
        cfg: Configuration composed by Hydra.
        save_to_file: Whether to export config to the hydra output folder.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            error_message = "Specify tags before launching a multirun!"
            raise ValueError(error_message)

        _LOGGER.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        _LOGGER.info("Tags: %s", cfg.tags)

    if save_to_file:
        with Path(cfg.paths.log_path, "config_tree.log").open("w") as file:
            rich.print(cfg.tags, file=file)


def apply_extras(cfg: DictConfig) -> None:
    """Apply optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing

    Args:
        cfg: Configuration composed by Hydra.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        _LOGGER.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        _LOGGER.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        _LOGGER.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        _enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        _LOGGER.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)
