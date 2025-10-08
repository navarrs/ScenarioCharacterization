import hydra
from omegaconf import DictConfig

from characterization.features.base_feature import BaseFeature
from characterization.processors.base_processor import BaseProcessor
from characterization.scorer.base_scorer import BaseScorer
from characterization.utils.datasets import BaseDataset
from characterization.utils.io_utils import get_logger, make_output_paths, print_config

logger = get_logger(__name__)


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

    characterizer: BaseFeature | BaseScorer = hydra.utils.instantiate(cfg.characterizer)

    logger.info("Instatiating processor: %s", cfg.processor._target_)
    processor: BaseProcessor = hydra.utils.instantiate(cfg.processor, dataset=dataset, characterizer=characterizer)

    try:
        logger.info("Running scenario processor...")
        processor.run()
    except AssertionError:
        logger.exception("Error Processing Data")
        raise
    logger.info("Processing completed successfully.")


if __name__ == "__main__":
    """Entry point for running the scenario characterization processor."""
    # The run function is decorated with @hydra.main, which allows it to be executed
    run()  # pyright: ignore[reportCallIssue]
