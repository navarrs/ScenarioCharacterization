import hydra

from omegaconf import DictConfig
from torch.utils.data import Dataset

from processors.scores_processor import ScoresProcessor
from scorer.base_scorer import BaseScorer
from utils.common import make_output_paths, get_logger

logger = get_logger(__name__)

@hydra.main(config_path="config", config_name="run_score_processor", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """
    Run the processor with the given configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.
    """
    # TODO: is this automatically handled?
    make_output_paths(cfg.copy())

    logger.info("Instatiating dataset: %s", cfg.dataset._target_)
    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)

    logger.info("Instatiating scorer: %s", cfg.scorer._target_)
    scorer: BaseScorer = hydra.utils.instantiate(cfg.scorer)

    logger.info("Instatiating processor: %s", cfg.processor._target_)
    processor: ScoresProcessor = hydra.utils.instantiate(cfg.processor, dataset=dataset, scorer=scorer)
    try:
        logger.info("Generating scenario scores...")
        processor.run()
    except Exception as e:
        logger.error("Error Processing Data: %s", e)
        return
    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    run()
