import hydra

from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.features.base_feature import BaseFeature
from src.scorer.base_scorer import BaseScorer
from src.processors.base_processor import BaseProcessor
from src.utils.common import make_output_paths, get_logger

logger = get_logger(__name__)

@hydra.main(config_path="config", config_name="run_processor", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """
    Run the processor with the given configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.
    """
    make_output_paths(cfg.copy())
    
    logger.info("Instatiating dataset: %s", cfg.dataset._target_)
    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)

    logger.info("Instatiating characterizer: %s", cfg.characterizer._target_)
    characterizer: BaseFeature | BaseScorer = hydra.utils.instantiate(cfg.characterizer)

    logger.info("Instatiating processor: %s", cfg.processor._target_)
    processor: BaseProcessor = hydra.utils.instantiate(
        cfg.processor, 
        dataset=dataset, 
        characterizer=characterizer
    )
    
    try:
        logger.info("Generating scenario features...")
        processor.run()
    except AssertionError as e:
        import traceback
        logger.error(f"Error Processing Data: %{e}\n{traceback.print_exc}")
        raise e
    
    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    run()