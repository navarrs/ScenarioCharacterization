import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset

from characterization.features.base_feature import BaseFeature
from characterization.processors.base_processor import BaseProcessor
from characterization.scorer.base_scorer import BaseScorer
from characterization.utils.common import get_logger, make_output_paths

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

    logger.info("Instatiating dataset: %s", cfg.dataset._target_)
    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)

    logger.info("Instatiating characterizer: %s", cfg.characterizer._target_)
    characterizer: BaseFeature | BaseScorer = hydra.utils.instantiate(cfg.characterizer)

    logger.info("Instatiating processor: %s", cfg.processor._target_)
    processor: BaseProcessor = hydra.utils.instantiate(cfg.processor, dataset=dataset, characterizer=characterizer)

    try:
        logger.info("Generating scenario features...")
        processor.run()
    except AssertionError as e:
        logger.error(f"Error Processing Data: %{e}\n")
        raise e
    logger.info("Processing completed successfully.")


if __name__ == "__main__":
    """Entry point for running the scenario characterization processor."""
    run()
