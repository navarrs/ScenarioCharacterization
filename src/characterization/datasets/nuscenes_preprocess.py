"""Preprocessing script for the nuScenes dataset.

Reads raw nuScenes data using the nuScenes devkit, extracts scenario data, and writes pickle
files and a metadata index to disk. Trajectories are interpolated from native 2Hz keyframes
to 10Hz to match the Waymo convention.

Example usage::

    uv run python -m characterization.datasets.nuscenes_preprocess <raw_data_path> <output_path> [version]
"""

import json
import logging
import os
import pickle  # nosec B403
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from rich.progress import track

logger = logging.getLogger(__name__)

NUSCENES_CATEGORY_TO_AGENT_TYPE: dict[str, str] = {
    "vehicle.car": "TYPE_VEHICLE",
    "vehicle.truck": "TYPE_VEHICLE",
    "vehicle.bus.bendy": "TYPE_VEHICLE",
    "vehicle.bus.rigid": "TYPE_VEHICLE",
    "vehicle.trailer": "TYPE_VEHICLE",
    "vehicle.construction": "TYPE_VEHICLE",
    "vehicle.emergency.ambulance": "TYPE_VEHICLE",
    "vehicle.emergency.police": "TYPE_VEHICLE",
    "vehicle.bicycle": "TYPE_CYCLIST",
    "vehicle.motorcycle": "TYPE_CYCLIST",
}

MAP_LOCATIONS = [
    "boston-seaport",
    "singapore-onenorth",
    "singapore-hollandvillage",
    "singapore-queenstown",
]

# Polyline type IDs matching PolylineType enum in scenario_types.py
POLYLINE_TYPE_LANE = 2
POLYLINE_TYPE_ROAD_DIVIDER = 6
POLYLINE_TYPE_LANE_DIVIDER = 9
POLYLINE_TYPE_ROAD_EDGE = 15
POLYLINE_TYPE_CROSSWALK = 18
POLYLINE_TYPE_STOP_LINE = 17

# Fixed ego vehicle dimensions in metres (approximate values for a typical sedan)
EGO_LENGTH = 4.084
EGO_WIDTH = 1.730
EGO_HEIGHT = 1.562

# Scenario parameters: 6 seconds at 10Hz, with 2-second history
SCENARIO_DURATION_S = 6.0
SCENARIO_FREQ_HZ = 10.0
NUM_TIMESTEPS = 60
CURRENT_TIME_INDEX = 20

# Native nuScenes keyframe rate; 13 keyframes span 6 seconds at 2Hz
SOURCE_FREQ_HZ = 2.0
NUM_KEYFRAMES = int(SCENARIO_DURATION_S * SOURCE_FREQ_HZ) + 1

# Numerical tolerances and minimum sizes
_TIMESTAMP_EPS = 1e-6
_ZERO_EPS = 1e-9
_MIN_POINTS = 2
_MIN_SAMPLES = 2
_MIN_TIMESTEPS = 2
_MAP_MARGIN_M = 50.0
_LANE_RESOLUTION_M = 0.5


def quaternion_to_yaw(q: list[float]) -> float:
    """Extracts yaw angle in radians from quaternion [w, x, y, z]."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def get_agent_type(category_name: str) -> str:
    """Maps a nuScenes category name to an AgentType string."""
    if category_name in NUSCENES_CATEGORY_TO_AGENT_TYPE:
        return NUSCENES_CATEGORY_TO_AGENT_TYPE[category_name]
    if category_name.startswith("human.pedestrian"):
        return "TYPE_PEDESTRIAN"
    return "TYPE_OTHER"


def get_polyline_dir(polyline: NDArray[np.float32]) -> NDArray[np.float32]:
    """Computes unit direction vectors for each point of a polyline."""
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    return diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=_ZERO_EPS, a_max=1e9)


def get_sample_tokens(nusc: NuScenes, scene: dict[str, Any], max_tokens: int) -> list[str]:
    """Returns an ordered list of sample tokens for a scene up to max_tokens."""
    tokens: list[str] = []
    token: str = scene["first_sample_token"]
    while token and len(tokens) < max_tokens:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    return tokens


def interpolate_to_10hz(
    traj_2hz: NDArray[np.float32],
    ts_2hz: NDArray[np.float64],
    ts_10hz: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Linearly interpolates a 2Hz trajectory to 10Hz.

    Args:
        traj_2hz: (T_2hz, 10) array with columns [x, y, z, l, w, h, yaw, vx, vy, valid].
        ts_2hz: (T_2hz,) timestamps in seconds for each 2Hz keyframe.
        ts_10hz: (T_10hz,) timestamps in seconds for the 10Hz output grid.

    Returns:
        (T_10hz, 10) interpolated trajectory. Valid flag is 1.0 only between two consecutive
        valid keyframes; velocity is recomputed via central finite differences.
    """
    n_out = len(ts_10hz)
    n_in = len(ts_2hz)
    traj_10hz = np.zeros((n_out, 10), dtype=np.float32)
    valid_2hz = traj_2hz[:, 9].astype(bool)

    for out_idx in range(n_out):
        t = ts_10hz[out_idx]
        pos = int(np.searchsorted(ts_2hz, t + _TIMESTAMP_EPS, side="right")) - 1

        if pos < 0:
            continue
        if pos >= n_in - 1:
            if abs(t - ts_2hz[-1]) < _TIMESTAMP_EPS and valid_2hz[-1]:
                traj_10hz[out_idx, :7] = traj_2hz[-1, :7]
                traj_10hz[out_idx, 9] = 1.0
            continue

        if not (valid_2hz[pos] and valid_2hz[pos + 1]):
            continue

        dt_seg = float(ts_2hz[pos + 1] - ts_2hz[pos])
        alpha = (t - ts_2hz[pos]) / dt_seg if dt_seg > _ZERO_EPS else 0.0
        traj_10hz[out_idx, :7] = (1.0 - alpha) * traj_2hz[pos, :7] + alpha * traj_2hz[pos + 1, :7]
        traj_10hz[out_idx, 9] = 1.0

    # Recompute velocities via central finite differences on the 10Hz grid
    valid_10hz = traj_10hz[:, 9].astype(bool)
    for i in range(n_out):
        if not valid_10hz[i]:
            continue
        i_prev = i - 1 if i > 0 and valid_10hz[i - 1] else i
        i_next = i + 1 if i < n_out - 1 and valid_10hz[i + 1] else i
        if i_prev == i_next:
            continue
        dt_actual = float(ts_10hz[i_next] - ts_10hz[i_prev])
        traj_10hz[i, 7] = float((traj_10hz[i_next, 0] - traj_10hz[i_prev, 0]) / dt_actual)
        traj_10hz[i, 8] = float((traj_10hz[i_next, 1] - traj_10hz[i_prev, 1]) / dt_actual)

    return traj_10hz


def decode_ego_trajectory(
    nusc: NuScenes,
    sample_tokens: list[str],
    ts_2hz: NDArray[np.float64],
    ts_10hz: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Extracts the ego vehicle trajectory from ego_pose records and interpolates to 10Hz.

    Returns:
        (T_10hz, 10) ego trajectory with fixed vehicle dimensions and valid=1 at all keyframes.
    """
    n_in = len(sample_tokens)
    traj_2hz = np.zeros((n_in, 10), dtype=np.float32)

    for t, sample_token in enumerate(sample_tokens):
        sample = nusc.get("sample", sample_token)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        x, y, z = ep["translation"]
        yaw = quaternion_to_yaw(ep["rotation"])
        traj_2hz[t] = [x, y, z, EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT, yaw, 0.0, 0.0, 1.0]

    return interpolate_to_10hz(traj_2hz, ts_2hz, ts_10hz)


def decode_annotations_to_trajectories(
    nusc: NuScenes,
    sample_tokens: list[str],
    ts_2hz: NDArray[np.float64],
    ts_10hz: NDArray[np.float64],
) -> dict[str, Any]:
    """Builds per-agent trajectories from nuScenes sample annotations and interpolates to 10Hz.

    Returns:
        dict with keys "object_id" (list[int]), "object_type" (list[str]),
        "trajs" (ndarray of shape (N, T_10hz, 10)), and "instance_tokens" (list[str]).
    """
    n_in = len(sample_tokens)
    # instance_token -> {timestep_index: annotation_token}
    instance_ann_map: dict[str, dict[int, str]] = {}

    for t, sample_token in enumerate(sample_tokens):
        sample = nusc.get("sample", sample_token)
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            inst_token = ann["instance_token"]
            if inst_token not in instance_ann_map:
                instance_ann_map[inst_token] = {}
            instance_ann_map[inst_token][t] = ann_token

    object_ids: list[int] = []
    object_types: list[str] = []
    trajs_list: list[NDArray[np.float32]] = []
    instance_tokens: list[str] = []

    for inst_id, (inst_token, ann_map) in enumerate(instance_ann_map.items(), start=1):
        inst = nusc.get("instance", inst_token)
        category_name = nusc.get("category", inst["category_token"])["name"]
        agent_type = get_agent_type(category_name)

        traj_2hz = np.zeros((n_in, 10), dtype=np.float32)
        for t in range(n_in):
            if t not in ann_map:
                continue
            ann = nusc.get("sample_annotation", ann_map[t])
            x, y, z = ann["translation"]
            yaw = quaternion_to_yaw(ann["rotation"])
            # nuScenes size is [width, length, height]; trajectory format is [l, w, h]
            width, length, height = ann["size"]
            traj_2hz[t] = [x, y, z, length, width, height, yaw, 0.0, 0.0, 1.0]

        traj_10hz = interpolate_to_10hz(traj_2hz, ts_2hz, ts_10hz)
        trajs_list.append(traj_10hz)
        object_ids.append(inst_id)
        object_types.append(agent_type)
        instance_tokens.append(inst_token)

    n_out = len(ts_10hz)
    trajs = np.stack(trajs_list, axis=0) if trajs_list else np.zeros((0, n_out, 10), dtype=np.float32)
    return {"object_id": object_ids, "object_type": object_types, "trajs": trajs, "instance_tokens": instance_tokens}


def _build_lane_polyline(nusc_map: NuScenesMap, token: str) -> NDArray[np.float32] | None:
    """Discretizes a lane or lane_connector arcline into a (P, 7) polyline array."""
    try:
        lane = nusc_map.get_arcline_path(token)
        poses = arcline_path_utils.discretize_lane(lane, _LANE_RESOLUTION_M)
    except Exception:  # noqa: BLE001
        logger.debug("Skipping lane %s: discretize_lane failed", token)
        return None
    if len(poses) < _MIN_POINTS:
        return None
    points = np.array([[p[0], p[1], 0.0] for p in poses], dtype=np.float32)
    dirs = np.array([[np.cos(p[2]), np.sin(p[2]), 0.0] for p in poses], dtype=np.float32)
    type_col = np.full((len(points), 1), POLYLINE_TYPE_LANE, dtype=np.float32)
    return np.concatenate([points, dirs, type_col], axis=1)


def _build_line_polyline(
    nusc_map: NuScenesMap, layer_name: str, token: str, type_id: int
) -> NDArray[np.float32] | None:
    """Extracts a line layer record (road_divider or lane_divider) as a (P, 7) polyline array."""
    record = nusc_map.get(layer_name, token)
    line = nusc_map.get("line", record["line_token"])
    nodes = [nusc_map.get("node", n_token) for n_token in line["node_tokens"]]
    if len(nodes) < _MIN_POINTS:
        return None
    points = np.array([[n["x"], n["y"], 0.0] for n in nodes], dtype=np.float32)
    dirs = get_polyline_dir(points)
    type_col = np.full((len(points), 1), type_id, dtype=np.float32)
    return np.concatenate([points, dirs, type_col], axis=1)


def _build_polygon_polyline(
    nusc_map: NuScenesMap, layer_name: str, token: str, type_id: int
) -> NDArray[np.float32] | None:
    """Extracts a polygon layer record's exterior boundary as a (P, 7) polyline array."""
    record = nusc_map.get(layer_name, token)
    try:
        polygon = nusc_map.extract_polygon(record["polygon_token"])
    except Exception:  # noqa: BLE001
        logger.debug("Skipping %s %s: extract_polygon failed", layer_name, token)
        return None
    coords = list(polygon.exterior.coords)[:-1]
    if len(coords) < _MIN_POINTS:
        return None
    points = np.array([[c[0], c[1], 0.0] for c in coords], dtype=np.float32)
    dirs = get_polyline_dir(points)
    type_col = np.full((len(points), 1), type_id, dtype=np.float32)
    return np.concatenate([points, dirs, type_col], axis=1)


def decode_map_features(
    nusc_map: NuScenesMap,
    patch_box: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Extracts map features from a NuScenesMap within a given patch and converts to polyline format.

    Args:
        nusc_map: NuScenesMap for the relevant map location.
        patch_box: (x_center, y_center, height, width) in map coordinates.

    Returns:
        dict with "all_polylines" (P, 7) and per-category polyline index lists matching
        the Waymo pickle format ("lane", "road_line", "road_edge", "crosswalk", "stop_sign").
    """
    layer_names = [
        "lane",
        "lane_connector",
        "road_divider",
        "lane_divider",
        "road_segment",
        "ped_crossing",
        "stop_line",
    ]
    records = nusc_map.get_records_in_patch(patch_box, layer_names=layer_names)

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

    for layer_name in ["lane", "lane_connector"]:
        for token in records.get(layer_name, []):
            polyline = _build_lane_polyline(nusc_map, token)
            if polyline is not None:
                _append("lane", polyline, {"speed_limit_mph": 0.0})

    for layer_name, type_id in [
        ("road_divider", POLYLINE_TYPE_ROAD_DIVIDER),
        ("lane_divider", POLYLINE_TYPE_LANE_DIVIDER),
    ]:
        for token in records.get(layer_name, []):
            polyline = _build_line_polyline(nusc_map, layer_name, token, type_id)
            if polyline is not None:
                _append("road_line", polyline)

    for token in records.get("road_segment", []):
        polyline = _build_polygon_polyline(nusc_map, "road_segment", token, POLYLINE_TYPE_ROAD_EDGE)
        if polyline is not None:
            _append("road_edge", polyline)

    for token in records.get("ped_crossing", []):
        polyline = _build_polygon_polyline(nusc_map, "ped_crossing", token, POLYLINE_TYPE_CROSSWALK)
        if polyline is not None:
            _append("crosswalk", polyline)

    for token in records.get("stop_line", []):
        polyline = _build_polygon_polyline(nusc_map, "stop_line", token, POLYLINE_TYPE_STOP_LINE)
        if polyline is not None:
            # lane_ids is empty since nuScenes stop_line records don't reference lane IDs
            _append("stop_sign", polyline, {"lane_ids": []})

    map_infos["all_polylines"] = (
        np.concatenate(polylines_list, axis=0).astype(np.float32)
        if polylines_list
        else np.zeros((0, 7), dtype=np.float32)
    )
    return map_infos


def load_prediction_split(nusc: NuScenes, raw_data_path: str) -> dict[str, list[str]]:
    """Loads the nuScenes prediction challenge split file if available.

    Returns:
        dict mapping scene_token to list of instance_tokens to predict.
        Returns an empty dict if the prediction split file is not found.
    """
    pred_file = os.path.join(raw_data_path, "maps", "prediction", "nuscenes-prediction-challenge-trainval-v2.json")
    if not os.path.exists(pred_file):
        return {}

    with open(pred_file) as f:
        pred_data = json.load(f)

    scene_to_instances: dict[str, list[str]] = {}
    for item in pred_data:
        sample = nusc.get("sample", item["sample"])
        scene_token = sample["scene_token"]
        if scene_token not in scene_to_instances:
            scene_to_instances[scene_token] = []
        scene_to_instances[scene_token].append(item["instance"])

    return scene_to_instances


def process_nuscenes_scene(
    nusc: NuScenes,
    nusc_maps: dict[str, NuScenesMap],
    scene: dict[str, Any],
    output_path: str,
    prediction_agents_map: dict[str, list[str]],
) -> dict[str, Any]:
    """Processes a single nuScenes scene and saves a scenario pickle file.

    Args:
        nusc: NuScenes devkit instance.
        nusc_maps: Mapping from map location name to NuScenesMap instance.
        scene: nuScenes scene record.
        output_path: Directory to write the output pickle file.
        prediction_agents_map: Mapping from scene_token to list of instance_tokens to predict.

    Returns:
        Lightweight metadata dict (without track/map data) for the metadata index file,
        or an empty dict if the scene is too short to process.
    """
    scene_token: str = scene["token"]

    sample_tokens = get_sample_tokens(nusc, scene, max_tokens=NUM_KEYFRAMES)
    if len(sample_tokens) < _MIN_SAMPLES:
        return {}

    timestamps_us = np.array([nusc.get("sample", t)["timestamp"] for t in sample_tokens], dtype=np.float64)
    ts_2hz = (timestamps_us - timestamps_us[0]) / 1e6

    ts_10hz = np.round(np.arange(NUM_TIMESTEPS) / SCENARIO_FREQ_HZ, decimals=6)

    ego_traj = decode_ego_trajectory(nusc, sample_tokens, ts_2hz, ts_10hz)
    agent_track_infos = decode_annotations_to_trajectories(nusc, sample_tokens, ts_2hz, ts_10hz)

    # Prepend ego vehicle at index 0; NuScenesData.repack_agent_data overrides type to TYPE_EGO_AGENT
    all_trajs = np.concatenate([ego_traj[np.newaxis], agent_track_infos["trajs"]], axis=0).astype(np.float32)
    all_ids = [0, *agent_track_infos["object_id"]]
    all_types = ["TYPE_VEHICLE", *agent_track_infos["object_type"]]

    log = nusc.get("log", scene["log_token"])
    location: str = log["location"]
    nusc_map = nusc_maps[location]

    valid_ego = ego_traj[:, 9].astype(bool)
    ego_x = ego_traj[valid_ego, 0] if valid_ego.sum() > 0 else all_trajs[0, :, 0]
    ego_y = ego_traj[valid_ego, 1] if valid_ego.sum() > 0 else all_trajs[0, :, 1]

    x_center = float((ego_x.min() + ego_x.max()) / 2)
    y_center = float((ego_y.min() + ego_y.max()) / 2)
    patch_height = float(ego_y.max() - ego_y.min()) + 2 * _MAP_MARGIN_M
    patch_width = float(ego_x.max() - ego_x.min()) + 2 * _MAP_MARGIN_M
    patch_box = (x_center, y_center, patch_height, patch_width)

    map_infos = decode_map_features(nusc_map, patch_box)
    dynamic_map_infos: dict[str, list[Any]] = {"stop_point": [], "lane_id": [], "state": []}

    pred_instance_tokens = set(prediction_agents_map.get(scene_token, []))
    instance_tokens: list[str] = agent_track_infos["instance_tokens"]
    token_to_int_id = {tok: idx + 1 for idx, tok in enumerate(instance_tokens)}
    objects_of_interest = [token_to_int_id[tok] for tok in pred_instance_tokens if tok in token_to_int_id]

    num_non_ego = all_trajs.shape[0] - 1
    tracks_to_predict = {
        "track_index": list(range(1, all_trajs.shape[0])),
        "difficulty": [1.0] * num_non_ego,
        "object_type": all_types[1:],
    }

    info: dict[str, Any] = {
        "scenario_id": scene_token,
        "timestamps_seconds": ts_10hz.astype(np.float64),
        "current_time_index": min(CURRENT_TIME_INDEX, len(ts_10hz) - 1),
        "sdc_track_index": 0,
        "objects_of_interest": objects_of_interest,
        "tracks_to_predict": tracks_to_predict,
        "track_infos": {"object_id": all_ids, "object_type": all_types, "trajs": all_trajs},
        "map_infos": map_infos,
        "dynamic_map_infos": dynamic_map_infos,
    }

    output_file = os.path.join(output_path, f"{scene_token}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(info, f)  # nosec B301

    return {k: v for k, v in info.items() if k not in ("track_infos", "map_infos", "dynamic_map_infos")}


def get_infos_from_nuscenes(
    nusc: NuScenes,
    nusc_maps: dict[str, NuScenesMap],
    output_path: str,
    prediction_agents_map: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Processes all nuScenes scenes and collects lightweight metadata.

    Args:
        nusc: NuScenes devkit instance.
        nusc_maps: Mapping from map location name to NuScenesMap instance.
        output_path: Directory where per-scene pickle files are written.
        prediction_agents_map: Optional mapping from scene_token to prediction instance_tokens.

    Returns:
        List of lightweight metadata dicts (one per successfully processed scene).
    """
    os.makedirs(output_path, exist_ok=True)
    pred_map = prediction_agents_map or {}

    return [
        info
        for scene in track(nusc.scene, description="Processing nuScenes scenes")
        if (info := process_nuscenes_scene(nusc, nusc_maps, scene, output_path, pred_map))
    ]


def create_infos_from_nuscenes(
    raw_data_path: str,
    output_path: str,
    version: str = "v1.0-trainval",
) -> None:
    """Creates processed scenario pickle files from raw nuScenes data.

    Args:
        raw_data_path: Path to the root directory of the nuScenes dataset.
        output_path: Directory to save processed scenario pickle files and the metadata index.
        version: nuScenes dataset version string (e.g. "v1.0-trainval", "v1.0-mini").

    Raises:
        ValueError: If raw_data_path does not exist.
    """
    if not os.path.exists(raw_data_path):
        msg = f"The raw data path {raw_data_path} does not exist."
        raise ValueError(msg)

    nusc = NuScenes(version=version, dataroot=raw_data_path, verbose=True)
    nusc_maps = {loc: NuScenesMap(dataroot=raw_data_path, map_name=loc) for loc in MAP_LOCATIONS}

    prediction_agents_map = load_prediction_split(nusc, raw_data_path)

    os.makedirs(output_path, exist_ok=True)
    scenario_path = os.path.join(output_path, "scenarios")

    sample_infos = get_infos_from_nuscenes(nusc, nusc_maps, scenario_path, prediction_agents_map)

    sample_filename = os.path.join(output_path, "processed_scenario_samples_infos.pkl")
    with open(sample_filename, "wb") as f:
        pickle.dump(sample_infos, f)  # nosec B301

    logger.info("nuScenes info file saved to %s", sample_filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < _MIN_SAMPLES + 1:  # need at least raw_data_path and output_path
        logger.error(
            "Usage: python -m characterization.datasets.nuscenes_preprocess <raw_data_path> <output_path> [version]"
        )
        sys.exit(1)
    create_infos_from_nuscenes(
        raw_data_path=sys.argv[1],
        output_path=sys.argv[2],
        version=sys.argv[3] if len(sys.argv) > _MIN_SAMPLES + 1 else "v1.0-trainval",
    )
