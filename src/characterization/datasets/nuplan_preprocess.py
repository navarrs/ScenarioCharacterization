"""Preprocessing script for the nuPlan dataset.

Reads raw nuPlan data using the nuplan-devkit, extracts fixed-length scenario windows, and writes
pickle files and a metadata index to disk in the same Waymo format used by the other datasets.
Native 20Hz tracks are subsampled to 10Hz to match the Waymo/nuScenes convention.

Example usage::

    uv run python -m characterization.datasets.nuplan_preprocess \
        --data-root <db_logs_dir> --map-root <maps_dir> --output-path <out> --limit 5000
"""

import argparse
import logging
import os
import pickle  # nosec B403
from typing import Any

import numpy as np
from numpy.typing import NDArray
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from rich.progress import track

logger = logging.getLogger(__name__)

NUPLAN_TO_AGENT_TYPE: dict[TrackedObjectType, str] = {
    TrackedObjectType.VEHICLE: "TYPE_VEHICLE",
    TrackedObjectType.PEDESTRIAN: "TYPE_PEDESTRIAN",
    TrackedObjectType.BICYCLE: "TYPE_CYCLIST",
}

# Scenario parameters: 6 seconds at 10Hz, with 2-second history (matches the nuScenes convention).
SCENARIO_FREQ_HZ = 10.0
NUM_TIMESTEPS = 60
CURRENT_TIME_INDEX = 20
_TARGET_DT_S = 1.0 / SCENARIO_FREQ_HZ

# Empty map placeholder; replaced with real map features in a later step.
_EMPTY_DYNAMIC_MAP: dict[str, list[Any]] = {"stop_point": [], "lane_id": [], "state": []}


def get_agent_type(object_type: TrackedObjectType) -> str:
    """Maps a nuPlan TrackedObjectType to an AgentType string."""
    return NUPLAN_TO_AGENT_TYPE.get(object_type, "TYPE_OTHER")


def source_stride(scenario: AbstractScenario) -> int:
    """Returns the iteration stride that resamples the scenario's native rate to 10Hz."""
    db_dt = scenario.database_interval
    if db_dt <= 0:
        return 1
    return max(1, round(_TARGET_DT_S / db_dt))


def recompute_velocities(traj: NDArray[np.float32]) -> None:
    """Recomputes vx, vy in-place via central finite differences on the 10Hz grid.

    Operates only on valid timesteps (column 9). Positions are columns 0 (x) and 1 (y); velocities
    are written to columns 7 (vx) and 8 (vy). Using finite differences avoids nuPlan's mixed
    body-frame (ego) vs global-frame (agents) velocity conventions.
    """
    n = traj.shape[0]
    valid = traj[:, 9].astype(bool)
    for i in range(n):
        if not valid[i]:
            continue
        i_prev = i - 1 if i > 0 and valid[i - 1] else i
        i_next = i + 1 if i < n - 1 and valid[i + 1] else i
        if i_prev == i_next:
            continue
        dt = _TARGET_DT_S * (i_next - i_prev)
        traj[i, 7] = float((traj[i_next, 0] - traj[i_prev, 0]) / dt)
        traj[i, 8] = float((traj[i_next, 1] - traj[i_prev, 1]) / dt)


def decode_ego_trajectory(scenario: AbstractScenario, source_indices: list[int]) -> NDArray[np.float32]:
    """Extracts the ego trajectory at the sampled iterations as a (NUM_TIMESTEPS, 10) array."""
    vp = get_pacifica_parameters()
    traj = np.zeros((NUM_TIMESTEPS, 10), dtype=np.float32)
    for out_idx, i in enumerate(source_indices):
        center = scenario.get_ego_state_at_iteration(i).center
        traj[out_idx] = [center.x, center.y, 0.0, vp.length, vp.width, vp.height, center.heading, 0.0, 0.0, 1.0]
    recompute_velocities(traj)
    return traj


def decode_tracked_objects_to_trajectories(scenario: AbstractScenario, source_indices: list[int]) -> dict[str, Any]:
    """Builds per-agent trajectories from nuPlan tracked objects, subsampled to 10Hz.

    Returns:
        dict with keys "object_id" (list[int]), "object_type" (list[str]),
        "trajs" (ndarray of shape (N, NUM_TIMESTEPS, 10)), and "track_tokens" (list[str]).
    """
    # track_token -> {out_idx: box}, and track_token -> object type
    boxes: dict[str, dict[int, Any]] = {}
    types: dict[str, TrackedObjectType] = {}
    for out_idx, i in enumerate(source_indices):
        for obj in scenario.get_tracked_objects_at_iteration(i).tracked_objects:
            boxes.setdefault(obj.track_token, {})[out_idx] = obj.box
            types[obj.track_token] = obj.tracked_object_type

    object_ids: list[int] = []
    object_types: list[str] = []
    trajs_list: list[NDArray[np.float32]] = []
    track_tokens: list[str] = []

    for obj_id, (token, per_step) in enumerate(boxes.items(), start=1):
        traj = np.zeros((NUM_TIMESTEPS, 10), dtype=np.float32)
        for out_idx, box in per_step.items():
            traj[out_idx] = [
                box.center.x,
                box.center.y,
                0.0,
                box.length,
                box.width,
                box.height,
                box.center.heading,
                0.0,
                0.0,
                1.0,
            ]
        recompute_velocities(traj)
        trajs_list.append(traj)
        object_ids.append(obj_id)
        object_types.append(get_agent_type(types[token]))
        track_tokens.append(token)

    trajs = np.stack(trajs_list, axis=0) if trajs_list else np.zeros((0, NUM_TIMESTEPS, 10), dtype=np.float32)
    return {"object_id": object_ids, "object_type": object_types, "trajs": trajs, "track_tokens": track_tokens}


def process_nuplan_scenario(scenario: AbstractScenario, output_path: str) -> dict[str, Any]:
    """Processes a single nuPlan scenario and saves a scenario pickle file.

    Returns:
        Lightweight metadata dict (without track/map data) for the metadata index file,
        or an empty dict if the scenario is too short to yield NUM_TIMESTEPS samples.
    """
    stride = source_stride(scenario)
    num_source_steps = NUM_TIMESTEPS * stride
    if scenario.get_number_of_iterations() < num_source_steps:
        return {}

    source_indices = list(range(0, num_source_steps, stride))

    ego_traj = decode_ego_trajectory(scenario, source_indices)
    agent_track_infos = decode_tracked_objects_to_trajectories(scenario, source_indices)

    # Prepend ego vehicle at index 0; NuScenesData.repack_agent_data overrides type to TYPE_EGO_AGENT.
    all_trajs = np.concatenate([ego_traj[np.newaxis], agent_track_infos["trajs"]], axis=0).astype(np.float32)
    all_ids = [0, *agent_track_infos["object_id"]]
    all_types = ["TYPE_VEHICLE", *agent_track_infos["object_type"]]

    timestamps = np.round(np.arange(NUM_TIMESTEPS) / SCENARIO_FREQ_HZ, decimals=6).astype(np.float64)

    num_non_ego = all_trajs.shape[0] - 1
    tracks_to_predict = {
        "track_index": list(range(1, all_trajs.shape[0])),
        "difficulty": [1.0] * num_non_ego,
        "object_type": all_types[1:],
    }

    info: dict[str, Any] = {
        "scenario_id": scenario.token,
        "timestamps_seconds": timestamps,
        "current_time_index": min(CURRENT_TIME_INDEX, NUM_TIMESTEPS - 1),
        "sdc_track_index": 0,
        "objects_of_interest": [],
        "tracks_to_predict": tracks_to_predict,
        "track_infos": {"object_id": all_ids, "object_type": all_types, "trajs": all_trajs},
        "map_infos": None,  # populated by decode_map_features in a later step
        "dynamic_map_infos": _EMPTY_DYNAMIC_MAP,
    }

    output_file = os.path.join(output_path, f"{scenario.token}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(info, f)  # nosec B301

    return {k: v for k, v in info.items() if k not in ("track_infos", "map_infos", "dynamic_map_infos")}


def build_scenarios(
    data_root: str,
    map_root: str,
    output_path: str,
    limit: int,
    map_version: str,
    sensor_root: str | None,
    db_files: str | None,
) -> list[dict[str, Any]]:
    """Selects up to `limit` nuPlan scenarios and writes one pickle per scenario."""
    os.makedirs(output_path, exist_ok=True)
    builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=map_root,
        sensor_root=sensor_root,
        db_files=db_files,
        map_version=map_version,
        scenario_mapping=ScenarioMapping(scenario_map={}, subsample_ratio_override=None),
        vehicle_parameters=get_pacifica_parameters(),
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=limit,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=True,
    )
    scenarios = builder.get_scenarios(scenario_filter, Sequential())

    return [
        info
        for scenario in track(scenarios, description="Processing nuPlan scenarios")
        if (info := process_nuplan_scenario(scenario, output_path))
    ]


def create_infos_from_nuplan(
    data_root: str,
    map_root: str,
    output_path: str,
    limit: int,
    map_version: str = "nuplan-maps-v1.0",
    sensor_root: str | None = None,
    db_files: str | None = None,
) -> None:
    """Creates processed scenario pickle files and a metadata index from raw nuPlan data."""
    scenario_path = os.path.join(output_path, "scenarios")
    sample_infos = build_scenarios(data_root, map_root, scenario_path, limit, map_version, sensor_root, db_files)

    sample_filename = os.path.join(output_path, "processed_scenario_samples_infos.pkl")
    with open(sample_filename, "wb") as f:
        pickle.dump(sample_infos, f)  # nosec B301
    logger.info("nuPlan info file saved to %s (%d scenarios)", sample_filename, len(sample_infos))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess nuPlan scenarios into Waymo-format pickles.")
    parser.add_argument("--data-root", required=True, help="Directory containing nuPlan .db log files.")
    parser.add_argument("--map-root", required=True, help="Directory containing the nuPlan maps.")
    parser.add_argument("--output-path", required=True, help="Output directory for scenario pickles.")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum number of scenarios to extract.")
    parser.add_argument("--map-version", default="nuplan-maps-v1.0", help="nuPlan map version string.")
    parser.add_argument("--sensor-root", default=None, help="Optional sensor blob root (not required).")
    parser.add_argument("--db-files", default=None, help="Optional specific .db file(s); defaults to all in data-root.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    create_infos_from_nuplan(
        data_root=args.data_root,
        map_root=args.map_root,
        output_path=args.output_path,
        limit=args.limit,
        map_version=args.map_version,
        sensor_root=args.sensor_root,
        db_files=args.db_files,
    )
