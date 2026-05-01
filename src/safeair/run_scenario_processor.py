"""Scenario processor runner for SafeAir.

Example usage:
    Run with default configs:
        uv run -m safeair.run_scenario_processor

    Overwrite, if already processed:
        uv run -m safeair.run_scenario_processor overwrite=True
"""

import copy
from pathlib import Path

import hydra
import pyrootutils
from omegaconf import DictConfig

from safeair import utils

_LOGGER = utils.get_pylogger(__name__)
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(config_path="configs", config_name="processor", version_base=None)
def run(config: DictConfig) -> None:
    """Runs the SafeAir scenario processor.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    _LOGGER.info("Running SafeAir Scenario Processor")

    # Load airport
    processor_config = config.processor
    supported_airports = utils.get_available_airports_from_assets_path(Path(processor_config.assets_path))
    _LOGGER.info("Found %s airports: %s", len(supported_airports), supported_airports)

    # Sanity checks
    airports = utils.parse_airport(processor_config.airport, supported_airports)
    if len(airports) == 0:
        _LOGGER.error("No airports found, check paths!")
        return
    _LOGGER.info("Processing %s airports: %s", len(airports), airports)

    # Run processor for each airport
    for airport in airports:
        _LOGGER.info("Processing airport: %s", airport)
        airport_config = copy.deepcopy(processor_config)  # Modify config for compatibility with each airport.
        airport_config.airport = airport
        utils.print_config_tree(airport_config, print_order=["config", "processor"], resolve=True, save_to_file=False)

        try:
            # Intantiate processor
            airport_processor = hydra.utils.get_class(airport_config._target_)
            airport_processor(config=airport_config).process_data()
        except Exception:
            _LOGGER.exception("Error processing %s", airport)


if __name__ == "__main__":
    run()
