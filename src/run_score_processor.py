import hydra
from omegaconf import DictConfig

from utils.common import _make_output_paths
from utils.logger import get_logger

logger = get_logger(__name__)

@hydra.main(config_path="config", config_name="run_score_processor", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """
    Run the processor with the given configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.
    """
    # TODO: remove?
    _make_output_paths(cfg.copy())

    logger.info("Instatiating dataset: %s", cfg.dataset._target_)
    dataset = hydra.utils.instantiate(cfg.dataset)
    try:
        logger.info("Loading Scenario Data...")
        dataset.load_data()
    except Exception as e:
        logger.error("Error Loading Scenario Data: %s", e)
        return

    logger.info("Instatiating scorer: %s", cfg.scorer._target_)
    scorer = hydra.utils.instantiate(cfg.scorer)

    logger.info("Instatiating processor: %s", cfg.processor._target_)
    processor = hydra.utils.instantiate(cfg.processor, dataset=dataset, scorer=scorer)
    try:
        logger.info("Generating scenario scores...")
        processor.run()
    except Exception as e:
        logger.error("Error Processing Data: %s", e)
        return
    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    run()
