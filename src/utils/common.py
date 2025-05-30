import os

from omegaconf import DictConfig

SUPPORTED_SCENARIO_TYPES = ["gt"]


def _make_output_paths(cfg: DictConfig) -> None:
    """
    Create output paths based on the configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        EasyDict: Dictionary containing output paths.
    """
    # TODO: is there an automated and clearn way to do this?
    os.makedirs(cfg.paths.output_paths.feature_cache_path, exist_ok=True)
    os.makedirs(cfg.paths.output_paths.scores_cache_path, exist_ok=True)
    os.makedirs(cfg.paths.output_paths.stats_cache_path, exist_ok=True)
    os.makedirs(cfg.paths.output_paths.vis_cache_path, exist_ok=True)
