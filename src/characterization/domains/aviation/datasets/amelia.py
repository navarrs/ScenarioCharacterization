from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from torch.utils.data import Dataset

from characterization.domains.aviation.schemas.scenario import MapData, Scenario
from characterization.domains.aviation.utils.file_io_utils import load_scenario
from characterization.utils.logging_utils import get_pylogger

logger = get_pylogger(__name__)


class AmeliaDataset(Dataset):  # pyright: ignore[reportMissingTypeArgument]
    """PyTorch Dataset for Amelia aviation scenarios.

    Wraps the existing load_scenario utility to provide a standard Dataset interface
    compatible with BaseProcessor and its DataLoader.

    Each scenario is stored as a ``.pkl`` file under ``scenarios_dir``, organised in
    per-airport subdirectories (``<scenarios_dir>/<airport_id>/<scenario>.pkl``).

    Args:
        config: Hydra config with the following keys:
            - scenarios_dir: root directory containing per-airport scenario folders.
            - maps_dir: directory containing per-airport map pickles (optional).
            - num_scenarios: cap on the total number of scenarios to load (optional).
    """

    def __init__(self, config: DictConfig) -> None:
        """Discover all scenario files under ``config.scenarios_dir`` and apply an optional count cap."""
        super().__init__()
        self.scenarios_dir = Path(config.scenarios_dir)
        self.maps_dir = Path(config.maps_dir) if config.get("maps_dir") else None
        self.num_scenarios: int | None = config.get("num_scenarios", None)

        self._scenario_files: list[Path] = []
        self._map_cache: dict[str, MapData | None] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Discover all scenario files and apply optional count cap."""
        files = sorted(self.scenarios_dir.rglob("*.pkl"))
        if self.num_scenarios is not None:
            files = files[: self.num_scenarios]
        self._scenario_files = files
        logger.info("Loaded %d scenario files from %s", len(self._scenario_files), self.scenarios_dir)

    @property
    def name(self) -> str:
        """Return a human-readable label including the scenarios directory."""
        return f"AmeliaDataset (loaded from: {self.scenarios_dir})"

    def __len__(self) -> int:
        """Return the number of scenario files in this dataset."""
        return len(self._scenario_files)

    def __getitem__(self, index: int) -> Scenario:
        """Load and return the scenario at the given index."""
        pkl_path = self._scenario_files[index]
        # Convention: airport ID is the immediate parent directory name.
        airport_id = pkl_path.parent.name
        scenario = load_scenario(pkl_path, self.maps_dir, airport_id, self._map_cache)
        if scenario is None:
            error_message = f"Could not load scenario from {pkl_path}"
            raise ValueError(error_message)
        return scenario

    def collate_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        """Collate a list of Scenario objects into the format expected by processors."""
        return {"scenario": batch}
