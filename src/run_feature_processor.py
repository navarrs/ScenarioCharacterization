import hydra

from omegaconf import DictConfig
from torch.utils.data import Dataset

from features.base_feature import BaseFeature
from processors.feature_processor import FeatureProcessor
from utils.common import make_output_paths, get_logger

logger = get_logger(__name__)

@hydra.main(config_path="config", config_name="run_feature_processor", version_base="1.3")
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

    logger.info("Instatiating feature: %s", cfg.feature._target_)
    feature: BaseFeature = hydra.utils.instantiate(cfg.feature)

    logger.info("Instatiating processor: %s", cfg.processor._target_)
    processor: FeatureProcessor = hydra.utils.instantiate(
        cfg.processor, 
        dataset=dataset, 
        feature=feature
    )
    try:
        logger.info("Generating scenario features...")
        processor.run()
    except Exception as e:
        logger.error("Error Processing Data: %s", e)
        return
    logger.info("Processing completed successfully.")

if __name__ == "__main__":
    run()