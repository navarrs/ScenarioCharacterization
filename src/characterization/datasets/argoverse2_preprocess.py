"""Preprocessing script for the Argoverse 2 Motion Forecasting dataset.

Reads raw AV2 data using the av2 Python API, extracts scenario data, and writes pickle files and a metadata index to
disk. AV2 is already at 10 Hz; no interpolation is required.

Example usage::

    uv run python -m characterization.datasets.argoverse2_preprocess \
        <raw_data_path> <output_path> [--split train|val|test|sample]
"""

import argparse
import logging
import os
import pickle  # nosec B403
from pathlib import Path
from typing import Any

import numpy as np
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType, TrackCategory
from av2.map.lane_segment import LaneMarkType
from av2.map.map_api import ArgoverseStaticMap
from numpy.typing import NDArray
from rich.progress import track

from characterization.utils.geometric_utils import build_polyline
from characterization.utils.scenario_types import PolylineType

_LOGGER = logging.getLogger(__name__)

# Temporal structure
# AV2 Motion Forecasting: 11 seconds at 10 Hz = 110 timesteps.
# History: timesteps [0, 49]  (50 timesteps = 5 s observed)
# Future:  timesteps [50, 109] (60 timesteps = 6 s predicted)
NUM_TIMESTEPS: int = 110
SCENARIO_FREQ_HZ: float = 10.0
CURRENT_TIME_INDEX: int = 49  # last observed timestep

# Agent type mapping
AV2_TO_AGENT_TYPE: dict[ObjectType, str] = {
    ObjectType.VEHICLE: "TYPE_VEHICLE",
    ObjectType.BUS: "TYPE_VEHICLE",
    ObjectType.MOTORCYCLIST: "TYPE_VEHICLE",
    ObjectType.RIDERLESS_BICYCLE: "TYPE_CYCLIST",
    ObjectType.PEDESTRIAN: "TYPE_PEDESTRIAN",
    ObjectType.CYCLIST: "TYPE_CYCLIST",
    ObjectType.STATIC: "TYPE_OTHER",
    ObjectType.BACKGROUND: "TYPE_OTHER",
    ObjectType.CONSTRUCTION: "TYPE_OTHER",
    ObjectType.UNKNOWN: "TYPE_OTHER",
}

# Default bounding-box dimensions (length_m, width_m, height_m).
# AV2 motion forecasting does not include 3D bounding boxes (Wilson et al., NeurIPS 2023,
# https://arxiv.org/abs/2301.00493); values below are engineering estimates based on typical real-world object sizes. A
# structurally similar approach is used in HPTR (https://github.com/zhejz/HPTR/blob/main/src/pack_h5_av2.py) with
# slightly different numbers.
DEFAULT_DIMS: dict[ObjectType, tuple[float, float, float]] = {
    ObjectType.VEHICLE: (4.5, 2.0, 1.7),
    ObjectType.BUS: (12.0, 2.9, 3.5),
    ObjectType.MOTORCYCLIST: (2.3, 0.9, 1.3),
    ObjectType.RIDERLESS_BICYCLE: (1.8, 0.6, 1.2),
    ObjectType.PEDESTRIAN: (0.8, 0.8, 1.8),
    ObjectType.CYCLIST: (2.0, 0.8, 1.6),
    ObjectType.STATIC: (0.5, 0.5, 0.5),
    ObjectType.BACKGROUND: (0.5, 0.5, 0.5),
    ObjectType.CONSTRUCTION: (1.0, 1.0, 1.5),
    ObjectType.UNKNOWN: (0.5, 0.5, 0.5),
}

# LaneMarkType → polyline type ID; None = skip this boundary entirely.
# For compound markings (SOLID_DASH, DASH_SOLID), the most restrictive side (solid) is used
# since no exact PolylineType analog exists. DOUBLE_DASH_WHITE maps to BROKEN_SINGLE_WHITE
# because TYPE_BROKEN_DOUBLE_WHITE does not exist in the enum.
LANE_MARK_TO_POLYLINE_TYPE: dict[LaneMarkType, int | None] = {
    LaneMarkType.SOLID_WHITE: PolylineType.TYPE_SOLID_SINGLE_WHITE.value,
    LaneMarkType.SOLID_YELLOW: PolylineType.TYPE_SOLID_SINGLE_YELLOW.value,
    LaneMarkType.DOUBLE_SOLID_WHITE: PolylineType.TYPE_SOLID_DOUBLE_WHITE.value,
    LaneMarkType.DOUBLE_SOLID_YELLOW: PolylineType.TYPE_SOLID_DOUBLE_YELLOW.value,
    LaneMarkType.DASHED_WHITE: PolylineType.TYPE_BROKEN_SINGLE_WHITE.value,
    LaneMarkType.DASHED_YELLOW: PolylineType.TYPE_BROKEN_SINGLE_YELLOW.value,
    LaneMarkType.DOUBLE_DASH_WHITE: PolylineType.TYPE_BROKEN_SINGLE_WHITE.value,
    LaneMarkType.DOUBLE_DASH_YELLOW: PolylineType.TYPE_BROKEN_DOUBLE_YELLOW.value,
    LaneMarkType.SOLID_DASH_WHITE: PolylineType.TYPE_SOLID_SINGLE_WHITE.value,
    LaneMarkType.SOLID_DASH_YELLOW: PolylineType.TYPE_SOLID_SINGLE_YELLOW.value,
    LaneMarkType.DASH_SOLID_WHITE: PolylineType.TYPE_SOLID_SINGLE_WHITE.value,
    LaneMarkType.DASH_SOLID_YELLOW: PolylineType.TYPE_SOLID_SINGLE_YELLOW.value,
    LaneMarkType.NONE: None,
    LaneMarkType.UNKNOWN: None,
}


def build_agent_trajectories(
    scenario: ArgoverseScenario,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Converts AV2 Track objects into the canonical track_infos dict.

    The focal track is placed at index 0 with integer ID 0. All other tracks are assigned consecutive integer IDs
    starting at 1, in the order they appear in scenario.tracks.

    AV2 tracks are sparse: ObjectState only exists for timesteps where the agent was observed or for which a state is
    provided. Missing timesteps are left as zeros (valid flag = 0.0). valid=1.0 is set for any timestep with a state
    entry, regardless of state.observed, so that ground-truth scenarios have full coverage.

    Returns:
        track_infos: dict with "object_id" (list[int]), "object_type" (list[str]), and "trajs" (ndarray float32, shape
            (N, 110, 10)).
        track_id_to_int: mapping from AV2 string track_id to assigned integer ID.
    """
    focal_id = scenario.focal_track_id

    # Order tracks: focal first, then all others in original order.
    focal_track = next(t for t in scenario.tracks if t.track_id == focal_id)
    other_tracks = [t for t in scenario.tracks if t.track_id != focal_id]
    ordered_tracks = [focal_track, *other_tracks]

    track_id_to_int: dict[str, int] = {t.track_id: i for i, t in enumerate(ordered_tracks)}

    object_ids: list[int] = []
    object_types: list[str] = []
    trajs_list: list[NDArray[np.float32]] = []

    for agent_track in ordered_tracks:
        int_id = track_id_to_int[agent_track.track_id]
        agent_type = AV2_TO_AGENT_TYPE.get(agent_track.object_type, "TYPE_OTHER")
        length, width, height = DEFAULT_DIMS.get(agent_track.object_type, (0.5, 0.5, 0.5))

        traj = np.zeros((NUM_TIMESTEPS, 10), dtype=np.float32)
        for state in agent_track.object_states:
            t = state.timestep
            x, y = state.position
            vx, vy = state.velocity
            traj[t] = [x, y, 0.0, length, width, height, state.heading, vx, vy, 1.0]

        object_ids.append(int_id)
        object_types.append(agent_type)
        trajs_list.append(traj)

    trajs = np.stack(trajs_list, axis=0) if trajs_list else np.zeros((0, NUM_TIMESTEPS, 10), dtype=np.float32)
    track_infos: dict[str, Any] = {"object_id": object_ids, "object_type": object_types, "trajs": trajs}
    return track_infos, track_id_to_int


def build_tracks_to_predict(
    scenario: ArgoverseScenario,
    track_id_to_int: dict[str, int],
    object_types: list[str],
) -> dict[str, Any]:
    """Builds the tracks_to_predict dict from AV2 TrackCategory labels.

    FOCAL_TRACK and SCORED_TRACK get difficulty=1. TRACK_FRAGMENT gets difficulty=0. UNSCORED_TRACK is excluded from
    tracks_to_predict.

    Returns:
        dict with "track_index" (list[int]), "difficulty" (list[int]),
        "object_type" (list[str]).
    """
    track_indices: list[int] = []
    difficulties: list[int] = []
    pred_types: list[str] = []

    for agent_track in scenario.tracks:
        if agent_track.category == TrackCategory.UNSCORED_TRACK:
            continue
        idx = track_id_to_int[agent_track.track_id]
        difficulty = 0 if agent_track.category == TrackCategory.TRACK_FRAGMENT else 1
        track_indices.append(idx)
        difficulties.append(difficulty)
        pred_types.append(object_types[idx])

    return {"track_index": track_indices, "difficulty": difficulties, "object_type": pred_types}


def decode_map_features(static_map: ArgoverseStaticMap) -> dict[str, Any]:
    """Extracts map polylines from an ArgoverseStaticMap into the canonical dict format.

    Produces the same key structure as nuscenes_preprocess.decode_map_features(): "lane", "road_line", "road_edge",
    "crosswalk", "stop_sign", and "all_polylines".

    Lane centerlines are extracted per LaneSegment; left and right boundaries are added to "road_line" with a type based
    on the LaneMarkType. Adjacent lanes share boundaries — both sides are included without deduplication, matching the
    nuScenes and Waymo behaviour. Road edges and stop signs are not available in the AV2 map, they remain empty.

    Args:
        static_map: ArgoverseStaticMap loaded from the scenario's map directory.

    Returns:
        map_infos dict ready to be stored in the scenario pickle.
    """
    map_infos: dict[str, Any] = {"lane": [], "road_line": [], "road_edge": [], "crosswalk": [], "stop_sign": []}
    polylines_list: list[NDArray[np.float32]] = []
    point_cnt = 0
    feature_id = 0

    def _append(category: str, polyline: NDArray[np.float32], extra: dict[str, Any] | None = None) -> None:
        nonlocal point_cnt, feature_id
        entry: dict[str, Any] = {"id": feature_id, "polyline_index": (point_cnt, point_cnt + len(polyline))}
        if extra:
            entry.update(extra)
        map_infos[category].append(entry)
        polylines_list.append(polyline)
        point_cnt += len(polyline)
        feature_id += 1

    for ls in static_map.get_scenario_lane_segments():
        # Lane centerline
        centerline = static_map.get_lane_segment_centerline(ls.id).astype(np.float32)
        pl = build_polyline(centerline, PolylineType.TYPE_SURFACE_STREET.value)
        if pl is not None:
            _append("lane", pl, {"speed_limit_mph": float("nan")})

        # Left and right lane boundaries → road_line (if mark type is not NONE/UNKNOWN)
        for boundary, mark_type in (
            (ls.left_lane_boundary, ls.left_mark_type),
            (ls.right_lane_boundary, ls.right_mark_type),
        ):
            polyline_type = LANE_MARK_TO_POLYLINE_TYPE.get(mark_type)
            if polyline_type is None:
                continue
            pl = build_polyline(boundary.xyz.astype(np.float32), polyline_type)
            if pl is not None:
                _append("road_line", pl)

    for pc in static_map.get_scenario_ped_crossings():
        # Concatenate both crosswalk edges (second reversed) to form a closed boundary.
        combined = np.concatenate([pc.edge1.xyz, np.flip(pc.edge2.xyz, axis=0)], axis=0).astype(np.float32)
        pl = build_polyline(combined, PolylineType.TYPE_CROSSWALK.value)
        if pl is not None:
            _append("crosswalk", pl)

    map_infos["all_polylines"] = (
        np.concatenate(polylines_list, axis=0).astype(np.float32)
        if polylines_list
        else np.zeros((0, 7), dtype=np.float32)
    )
    return map_infos


def process_av2_scenario(scenario_dir: Path, output_path: str) -> dict[str, Any]:
    """Processes a single AV2 scenario directory and writes a pickle file.

    Args:
        scenario_dir: Path to one scenario directory containing the parquet and map JSON.
        output_path: Directory to write the output pickle file.

    Returns:
        Lightweight metadata dict (without track/map arrays) for the index,
        or an empty dict on failure.
    """
    scenario_id = scenario_dir.name
    parquet_files = list(scenario_dir.glob("scenario_*.parquet"))
    if not parquet_files:
        _LOGGER.warning("No parquet file found in %s, skipping.", scenario_dir)
        return {}

    parquet_path = parquet_files[0]
    map_dir = scenario_dir  # log_map_archive_*.json lives in the same directory

    try:
        scenario: ArgoverseScenario = scenario_serialization.load_argoverse_scenario_parquet(parquet_path)
        static_map = ArgoverseStaticMap.from_map_dir(log_map_dirpath=map_dir)
    except Exception:  # noqa: BLE001
        _LOGGER.warning("Failed to load scenario %s, skipping.", scenario_id, exc_info=True)
        return {}

    timestamps_seconds = np.arange(NUM_TIMESTEPS, dtype=np.float64) / SCENARIO_FREQ_HZ

    track_infos, track_id_to_int = build_agent_trajectories(scenario)
    tracks_to_predict = build_tracks_to_predict(scenario, track_id_to_int, track_infos["object_type"])
    objects_of_interest = [track_id_to_int[scenario.focal_track_id]]

    try:
        map_infos = decode_map_features(static_map)
    except Exception:  # noqa: BLE001
        _LOGGER.warning("Map extraction failed for %s, using empty map.", scenario_id, exc_info=True)
        map_infos = {
            "lane": [],
            "road_line": [],
            "road_edge": [],
            "crosswalk": [],
            "stop_sign": [],
            "all_polylines": np.zeros((0, 7), dtype=np.float32),
        }

    dynamic_map_infos: dict[str, list[Any]] = {"stop_point": [], "lane_id": [], "state": []}

    info: dict[str, Any] = {
        "scenario_id": scenario_id,
        "timestamps_seconds": timestamps_seconds,
        "current_time_index": CURRENT_TIME_INDEX,
        "sdc_track_index": 0,
        "objects_of_interest": objects_of_interest,
        "tracks_to_predict": tracks_to_predict,
        "track_infos": track_infos,
        "map_infos": map_infos,
        "dynamic_map_infos": dynamic_map_infos,
    }

    output_file = os.path.join(output_path, f"{scenario_id}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(info, f)  # nosec B301

    return {k: v for k, v in info.items() if k not in ("track_infos", "map_infos", "dynamic_map_infos")}


def create_infos_from_av2(raw_data_path: str, output_path: str, split: str = "train") -> None:
    """Creates processed scenario pickle files from raw AV2 Motion Forecasting data.

    Discovers all scenario directories under ``{raw_data_path}/{split}/``, processes each one, and writes:
        - ``{output_path}/scenarios/{scenario_id}.pkl`` — one pickle per scenario
        - ``{output_path}/processed_scenario_samples_infos.pkl`` — lightweight metadata index

    Args:
        raw_data_path: Root directory of the AV2 Motion Forecasting dataset (contains subdirs: train/, val/, test/).
        output_path: Directory to save processed files.
        split: Dataset split to process (``"train"``, ``"val"``, or ``"test"``).

    Raises:
        ValueError: If ``raw_data_path`` or the split directory does not exist.
    """
    if not os.path.exists(raw_data_path):
        msg = f"The raw data path {raw_data_path} does not exist."
        raise ValueError(msg)

    split_path = os.path.join(raw_data_path, split)
    if not os.path.exists(split_path):
        msg = f"Split directory {split_path} does not exist."
        raise ValueError(msg)

    scenario_dirs = sorted(Path(split_path).iterdir())
    scenario_dirs = [d for d in scenario_dirs if d.is_dir()]
    _LOGGER.info("Found %d scenario directories in %s/%s", len(scenario_dirs), raw_data_path, split)

    os.makedirs(output_path, exist_ok=True)
    scenario_output_path = os.path.join(output_path, "scenarios")
    os.makedirs(scenario_output_path, exist_ok=True)

    sample_infos: list[dict[str, Any]] = []
    for scenario_dir in track(scenario_dirs, description="Processing AV2 scenarios"):
        info = process_av2_scenario(scenario_dir, scenario_output_path)
        if info:
            sample_infos.append(info)

    sample_filename = os.path.join(output_path, "processed_scenario_samples_infos.pkl")
    with open(sample_filename, "wb") as f:
        pickle.dump(sample_infos, f)  # nosec B301

    _LOGGER.info("AV2 info file saved to %s (%d scenarios processed)", sample_filename, len(sample_infos))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Preprocess Argoverse 2 Motion Forecasting data.")
    parser.add_argument("raw_data_path", type=str, help="Root directory of the AV2 dataset.")
    parser.add_argument("output_path", type=str, help="Output directory for processed pickle files.")
    parser.add_argument("--split", type=str, default="sample", choices=["train", "val", "test", "sample"])
    args = parser.parse_args()
    create_infos_from_av2(raw_data_path=args.raw_data_path, output_path=args.output_path, split=args.split)
