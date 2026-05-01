import os
import pickle  # nosec B403
import random
from pathlib import Path
from typing import Any
from warnings import warn

from characterization.domains.aviation.utils.scenario_characterization_utils import load_map
from characterization.utils.constants import LARGE_FLOAT
from characterization.utils.logging_utils import get_pylogger
from safeair.schemas import model_outputs as output_data
from safeair.schemas.scenario import MapData, Scenario

_LOGGER = get_pylogger(__name__)


def from_pickle(data_file: str) -> dict[str, Any] | None:
    """Load data from a pickle file.

    Args:
        data_file: Path to the pickle file.

    Returns:
        The loaded data.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not os.path.exists(data_file):
        warning_message = f"Data file {data_file} does not exist."
        warn(warning_message, UserWarning, stacklevel=2)
        return None

    with open(data_file, "rb") as f:
        return pickle.load(f)  # nosec B301


def to_pickle(
    output_path: str, input_data: dict[str, Any], tag: str, *, overwrite: bool = False, update: bool = False
) -> None:
    """Save data to a pickle file, merging with existing data if present.

    Args:
        output_path: Directory where the pickle file will be saved.
        input_data: Data to save.
        tag: Tag to use for the output file name.
        overwrite: Whether to overwrite existing data.
        update: Whether to update existing data.
    """
    data = {}
    data_file = os.path.join(output_path, f"{tag}.pkl")
    if overwrite:
        with open(data_file, "wb") as f:
            pickle.dump(input_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    # If not overwriting and file exists, load existing data to merge with new data
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            data = pickle.load(f)  # nosec B301

    scenario_id_data = data.get("scenario_id", None)
    if scenario_id_data is not None and scenario_id_data != input_data["scenario_id"]:
        error_message = "Mismatched scenario IDs when merging pickle data."
        raise AttributeError(error_message)

    # Iterate over input values and merge into existing data
    for key, value in input_data.items():
        if value is None:
            continue

        # Only add/modify data if the key does not exist or if update is True
        if key not in data or update:
            if isinstance(value, dict) and key in data:
                # Merge dictionaries
                data[key].update(value)
            else:
                data[key] = value

    with open(data_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_scenario(
    pkl_path: Path,
    maps_dir: Path | None,
    airport_id: str,
    map_cache: dict[str, MapData | None],
) -> Scenario | None:
    """Load a Scenario from *pkl_path* and optionally attach the airport map.

    The map for each airport is loaded at most once and cached in *map_cache*.

    Args:
        pkl_path: Path to the scenario ``.pkl`` file.
        maps_dir: Directory containing per-airport map pickle files, or ``None`` to skip map attachment.
        airport_id: Airport identifier used to look up the map in ``{maps_dir}/{airport_id}/{airport_id}.pkl``.
        map_cache: Mutable dict used to avoid re-loading the same map multiple times.

    Returns:
        The loaded Scenario, or ``None`` if loading fails.
    """
    try:
        scenario = Scenario.from_pickle(pkl_path)
    except Exception:
        _LOGGER.exception("Failed to load scenario from %s", pkl_path)
        return None

    if maps_dir is not None and scenario.static_map_data is None:
        if airport_id not in map_cache:
            map_cache[airport_id] = load_map(maps_dir, airport_id)
        if map_cache[airport_id] is not None:
            scenario.static_map_data = map_cache[airport_id]

    return scenario


def save_model_outputs(model_outputs: output_data.ModelOutput, cache_path: Path) -> None:
    """Save model outputs to a file.

    Args:
        model_outputs: Model outputs to be saved.
        cache_path: Path to the file where the outputs will be saved.
    """
    resplit_model_outputs = _resplit_batch(model_outputs)
    for scenario_id, scenario_outputs in resplit_model_outputs.items():
        scenario_cache_path = Path(cache_path, f"{scenario_id}.pkl")
        with scenario_cache_path.open("wb") as f:
            pickle.dump(scenario_outputs, f)


def save_cache(cache_infos: Any, filepath: Path) -> None:  # noqa: ANN401
    """Save cache information to a file.

    Args:
        cache_infos: Information to be cached.
        filepath: Path to the file where the cache will be saved.
    """
    with filepath.open("wb") as f:
        pickle.dump(cache_infos, f)


def _resplit_batch(batch: output_data.ModelOutput) -> dict[str, output_data.ModelOutput]:
    """Resplit a batch of model outputs into individual scenario outputs.

    Args:
        batch: A batch of model outputs containing multiple scenarios.

    Returns:
        Dictionary mapping scenario IDs to their corresponding ModelOutput data.
    """
    batch_resplit = {}

    # Unkpack model output
    batch_scenario_embedding = batch.scenario_embedding
    batch_trajectory_output = batch.trajectory_prediction_output
    batch_tokenization_output = batch.tokenization_output
    batch_agent_tokenization_output = batch.agent_tokenization_output
    batch_csi_output = batch.critical_scenario_identification_output

    batch_history_gt = batch.history_ground_truth.value
    batch_future_gt = batch.future_ground_truth.value
    batch_ego_gt = batch.ego_ground_truth.value
    batch_dataset_name = batch.dataset_name
    batch_agent_ids = batch.agent_ids.value

    for n, scenario_id in enumerate(batch.scenario_id):
        # Scenario Embedding
        scenario_embedding = None
        if batch_scenario_embedding is not None:
            scenario_enc = batch_scenario_embedding.scenario_enc
            scenario_embedding = output_data.ScenarioEmbedding(
                scenario_enc=None if scenario_enc is None else scenario_enc.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                scenario_dec=batch_scenario_embedding.scenario_dec.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            )

        trajectory_prediction_output = None
        if batch_trajectory_output is not None:
            trajectory_prediction_output = output_data.TrajectoryPredictionOutput(
                decoded_trajectories=batch_trajectory_output.decoded_trajectories.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                mode_probabilities=batch_trajectory_output.mode_probabilities.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                mode_logits=batch_trajectory_output.mode_logits.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            )

        tokenization_output = None
        if batch_tokenization_output is not None:
            probs = batch_tokenization_output.token_probabilities
            rec_emb = batch_tokenization_output.reconstructed_embedding
            quant_emb = batch_tokenization_output.quantized_embedding
            tokenization_output = output_data.TokenizationOutput(
                token_probabilities=None if probs is None else probs.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                token_indices=batch_tokenization_output.token_indices.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                input_embedding=batch_tokenization_output.input_embedding.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                reconstructed_embedding=None if rec_emb is None else rec_emb.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                quantized_embedding=None if quant_emb is None else quant_emb.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            )

        agent_tokenization_output = None
        if batch_agent_tokenization_output is not None:
            probs = batch_agent_tokenization_output.token_probabilities
            rec_emb = batch_agent_tokenization_output.reconstructed_embedding
            quant_emb = batch_agent_tokenization_output.quantized_embedding
            agent_tokenization_output = output_data.TokenizationOutput(
                token_probabilities=None if probs is None else probs.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                token_indices=batch_agent_tokenization_output.token_indices.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                input_embedding=batch_agent_tokenization_output.input_embedding.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                reconstructed_embedding=None if rec_emb is None else rec_emb.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                quantized_embedding=None if quant_emb is None else quant_emb.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            )

        csi_output = None
        if batch_csi_output is not None:
            csi_output = output_data.CriticalScenarioIdentificationOutput(
                critical_agents_gt_mask=batch_csi_output.critical_agents_gt_mask.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                critical_agents_pred_probabilities=batch_csi_output.critical_agents_pred_probabilities.value[n]
                .detach()
                .cpu(),  # pyright: ignore[reportArgumentType]
                los_timestep_index=batch_csi_output.los_timestep_index.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
                los_timestep_probabilities=batch_csi_output.los_timestep_probabilities.value[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            )

        # Output
        batch_resplit[scenario_id] = output_data.ModelOutput(
            scenario_embedding=scenario_embedding,
            trajectory_prediction_output=trajectory_prediction_output,
            tokenization_output=tokenization_output,
            agent_tokenization_output=agent_tokenization_output,
            critical_scenario_identification_output=csi_output,
            history_ground_truth=batch_history_gt[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            future_ground_truth=batch_future_gt[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            ego_ground_truth=batch_ego_gt[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
            dataset_name=[batch_dataset_name[n]],
            scenario_id=[scenario_id],
            agent_ids=batch_agent_ids[n].detach().cpu(),  # pyright: ignore[reportArgumentType]
        )

    return batch_resplit


def load_batches(
    base_data_path: Path, num_batches: int | None, num_scenarios: int | None, seed: int
) -> dict[str, output_data.ModelOutput]:
    """Load scenario batches from a directory, with options to limit the number of batches and scenarios.

    Args:
        base_data_path: Directory path where the batch files are located. Pass the split-specific subfolder
            (e.g. ``batch_cache_path / "test"``) to load only that split's outputs.
        num_batches: Maximum number of batch files to load. If None, all batch files are loaded.
        num_scenarios: Maximum number of scenarios to select from the loaded batches. If None, all are selected.
        seed: Random seed for reproducibility when selecting scenarios.

    Returns:
        Dictionary mapping scenario IDs to their corresponding ModelOutput data.
    """
    _LOGGER.info("Loading scenario batches from %s", base_data_path)
    num_batches = int(LARGE_FLOAT) if num_batches is None else min(num_batches, int(LARGE_FLOAT))
    random.seed(seed)

    batches = {}
    for n, batch_file in enumerate(base_data_path.glob("*.pkl")):
        if n >= num_batches:
            break
        with batch_file.open("rb") as f:
            batch: output_data.ModelOutput = pickle.load(f)

        batch_resplit = _resplit_batch(batch)
        batches.update(batch_resplit)

    if not batches:
        error_message = f"No batches found in {base_data_path}"
        raise ValueError(error_message)

    # Select scenarios
    scenario_ids = batches.keys()
    total_scenarios = len(scenario_ids)

    num_scenarios = max(1, total_scenarios) if num_scenarios is None else max(1, min(num_scenarios, total_scenarios))

    _LOGGER.info("Selecting %d / %d scenarios", num_scenarios, total_scenarios)
    selected_scenarios = random.sample(list(batches.keys()), num_scenarios)
    return {scenario: batches[scenario] for scenario in selected_scenarios}
