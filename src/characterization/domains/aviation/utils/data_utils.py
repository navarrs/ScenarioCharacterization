import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from airportsdata import load as tz_load
from numpy.typing import NDArray

from characterization.domains.aviation.scenario_types import INVALID_STATE_VALUE
from characterization.domains.aviation.schemas.airport_metadata import TimeMetadata
from characterization.utils.logging_utils import get_pylogger

log = get_pylogger(__name__)


def minmax_scaler(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize an input array using Min-Max normalization.

    Args:
        x: Input array.

    Returns:
        Normalized array.
    """
    value_range = np.max(x) - np.min(x)
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


def parse_airport(airports: list[str], supported_airports: list[str]) -> list[str]:
    """Parse airport input from config.

    Args:
        airports: List of airport ICAO codes or "all".
        supported_airports: List of supported airport ICAO codes.

    Returns:
        List of airport ICAO codes to process.
    """
    assert set(airports).issubset(supported_airports), f"[ERROR] Airport {airports} not supported!"
    return airports


def impute_state_data(
    sequence: NDArray[np.float32],
    frames: NDArray[np.float32],
    imputed_state_flag: int = INVALID_STATE_VALUE,
    interpolation_method: Literal["linear", "time", "index", "values", "nearest", "zero", "slinear"] = "linear",
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Impute missing data through a specified interpolation.

    Args:
        sequence: Trajectory sequence to impute.
        frames: Frame indices corresponding to the trajectory sequence.
        imputed_state_flag: Flag value to indicate imputed states.
        interpolation_method: Interpolation method to use. Options: "linear", "time", "index", "values",
            "nearest", "zero", "slinear".

    Returns:
        Trajectory sequence after imputation.
        Frame indices after imputation.
    """
    # Create a list from starting frame to ending frame in agent sequence
    start_frame, end_frame = frames[0], frames[-1]
    expected_frames = set(range(start_frame, end_frame + 1))

    # Create a list of the actual frames in the agent sequence. There may be missing data from which need to be impute
    frames_in_sequence = set(frames.tolist())

    # Compute the difference between the expected frames and the frames in the sequence, i.e., the missing data points.
    missing_frames = sorted(expected_frames - frames_in_sequence)

    # Insert nan rows where the missing data is. Then, interpolate.
    if len(missing_frames) > 0:
        imputed_sequence_df = pd.DataFrame(sequence, index=frames)
        for missing_frame in missing_frames:
            idx = np.where(imputed_sequence_df.index > missing_frame)[0][0]
            df_pre = imputed_sequence_df[:idx].copy()
            df_post = imputed_sequence_df[idx:].copy()
            # NOTE: this assumes that the last column is reserved for the imputed state flag
            df_pre.loc[missing_frame] = [np.nan] * (sequence.shape[1] - 1) + [imputed_state_flag]
            imputed_sequence_df = pd.concat([df_pre, df_post]).astype(float)

        # Now interpolate the missing data
        imputed_sequence_df = imputed_sequence_df.interpolate(method=interpolation_method)
        return imputed_sequence_df.to_numpy(), imputed_sequence_df.index.to_numpy()

    return sequence, frames


def process_local_time(timestamp: int, airport_code: str) -> datetime:
    """Return the local time at the specified airport for a given UNIX timestamp.

    Args:
        timestamp: UNIX timestamp of the scene.
        airport_code: ICAO airport code.

    Returns:
        Local time at the specified airport.
    """
    try:
        airport_tz = tz_load("ICAO")[airport_code.upper()]["tz"]
    except KeyError:
        airport_tz = "UTC"

        warning_message = f"Airport code {airport_code} not found in timezone database. Defaulting to UTC."
        log.warning(warning_message)

    # Parse time at frame
    base_time = datetime.fromtimestamp(timestamp, tz=UTC)

    # Return local time
    return base_time.astimezone(ZoneInfo(airport_tz))


def process_timestamp(timestamp: int, frame_idx: int, airport_code: str, max_interval: int = 3600) -> TimeMetadata:
    """Calculate current time from a UNIX timestamp and frame index, and return local time metadata.

    Args:
        timestamp: UNIX timestamp of the scene.
        frame_idx: Frame index in seconds to add to the timestamp.
        airport_code: ICAO airport code.
        max_interval: Maximum allowed interval between timestamp and calculated time.

    Returns:
        Object containing timezone, unix_epoch, utc_iso, and local_iso time strings.
    """
    use_localtime = True
    local_time = None
    try:
        airport_tz = tz_load("ICAO")[airport_code.upper()]["tz"]
    except KeyError:
        airport_tz = "UTC"
        use_localtime = False

        warning_message = f"Airport code {airport_code} not found in timezone database. Defaulting to UTC."
        log.warning(warning_message)

    # Parse time at frame
    base_time = datetime.fromtimestamp(timestamp, tz=UTC)
    time_at_frame = base_time + timedelta(seconds=frame_idx)
    unix_epoch = int(time_at_frame.timestamp())

    # Sanity check
    if abs(unix_epoch - timestamp) > max_interval:
        time_diff = abs(unix_epoch - int(timestamp))
        error_message = (
            f"Time difference between scene timestamp and frame index ({time_diff} seconds)",
            f" exceeds maximum interval ({max_interval} seconds).",
        )
        log.error(error_message)
        raise ValueError(error_message)

    # If possible, convert to local time
    if use_localtime:
        local_time = time_at_frame.astimezone(ZoneInfo(airport_tz))

    # Build output dictionary
    return TimeMetadata(
        timezone=airport_tz,
        unix_epoch=unix_epoch,
        utc_iso=base_time.isoformat(),
        local_iso=None if local_time is None else local_time.isoformat(),
    )


def get_file_timestamp(filepath: Path) -> int:
    """Extract the timestamp from a file path.

    Assumes filepaths are named in the format: <airport_id>_<file_id>_<additional_info>_<timestamp>.

    Args:
        filepath: Path to the file.

    Returns:
        Extracted timestamp as an integer.
    """
    stem = filepath.stem
    file_id = stem.split("_")
    scene_ts = file_id[-1]
    return int(scene_ts)


def get_available_airports_from_assets_path(assets_path: Path) -> list[str]:
    """Get list of available airports from assets directory.

    Args:
        assets_path: Assets directory.

    Returns:
        List of available airport codes.
    """
    assert assets_path.exists(), f"[ERROR] Assets directory {assets_path} does not exist!"
    available_airports = [
        sub_dir.name for sub_dir in assets_path.glob("*") if sub_dir.is_dir() and "blacklist" not in sub_dir.name
    ]
    available_airports.sort()
    return available_airports


def load_airport_reference(assets_path: Path, airport: str) -> tuple[float, float]:
    """Load the reference lat/lon for an airport from its limits.json.

    Args:
        assets_path: Path to the assets directory containing airport subfolders.
        airport: ICAO airport code.

    Returns:
        Tuple of (ref_lat, ref_lon).

    Raises:
        FileNotFoundError: If limits.json does not exist for the airport.
        KeyError: If ref_lat or ref_lon keys are missing from the file.
    """
    limits_file = assets_path / airport / "limits.json"
    if not limits_file.exists():
        error_message = f"limits.json not found for airport '{airport}': {limits_file}"
        raise FileNotFoundError(error_message)

    with limits_file.open() as f:
        data: dict[str, Any] = json.load(f)

    return data["ref_lat"], data["ref_lon"]


def is_ddp() -> bool:
    """Check if running in Distributed Data Parallel (DDP) mode based on environment variable.

    Returns:
        True if DDP mode is enabled, False otherwise.
    """
    return "WORLD_SIZE" in os.environ
