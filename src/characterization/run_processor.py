"""Entrypoint for running the scenario characterization processor over a dataset.

Instantiates the dataset, characterizer (feature extractor, scorer, or prober), and processor via Hydra, then
dispatches to the processor's ``run()`` method. Configuration is loaded from ``config/run_processor.yaml`` by default.

Example usage::

    uv run python -m characterization.run_processor
    uv run python -m characterization.run_processor characterizer=individual_features scenario_type=gt
    uv run python -m characterization.run_processor num_scenarios=100 shard_index=0 num_shards=4
    uv run python -m characterization.run_processor characterizer=cvm_probe
    uv run python -m characterization.run_processor characterizer=cvm_probe viz=probe_scenario
"""

import hydra
from omegaconf import DictConfig

from characterization.datasets import BaseDataset
from characterization.features.base_feature import BaseFeature
from characterization.probing.base_prober import BaseProber
from characterization.processors.base_processor import BaseProcessor
from characterization.scorer.base_scorer import BaseScorer
from characterization.utils.io_utils import get_logger, make_output_paths, print_config

logger = get_logger(__name__)

type AnyProcessor = BaseProcessor[BaseFeature] | BaseProcessor[BaseScorer] | BaseProcessor[BaseProber]


@hydra.main(config_path="config", config_name="run_processor", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """Runs the scenario characterization processor with the provided configuration.

    Instantiates the dataset, characterizer, and processor using Hydra, then executes the processor's run method.
    Handles errors and logs progress throughout the process.

    Args:
        cfg (DictConfig): Configuration dictionary containing dataset, characterizer, and processor parameters.

    Raises:
        AssertionError: If an error occurs during processing.
    """
    make_output_paths(cfg.copy())
    print_config(cfg, theme="native")

    logger.info("Instatiating dataset: %s", cfg.dataset._target_)
    dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)

    logger.info("Instatiating characterizer: %s", cfg.characterizer._target_)
    characterizer: BaseFeature | BaseScorer | BaseProber = hydra.utils.instantiate(cfg.characterizer)

    logger.info("Instatiating processor: %s", cfg.processor._target_)
    processor: AnyProcessor = hydra.utils.instantiate(cfg.processor, dataset=dataset, characterizer=characterizer)

    try:
        logger.info("Running scenario processor...")
        processor.run()
    except AssertionError:
        logger.exception("Error Processing Data")
        raise
    logger.info("Processing completed successfully.")


if __name__ == "__main__":
    run()
