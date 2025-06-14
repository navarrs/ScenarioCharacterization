# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved

import glob
import multiprocessing
import os
import pickle  # nosec B403
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

object_type = {0: "TYPE_UNSET", 1: "TYPE_VEHICLE", 2: "TYPE_PEDESTRIAN", 3: "TYPE_CYCLIST", 4: "TYPE_OTHER"}

lane_type = {0: "TYPE_UNDEFINED", 1: "TYPE_FREEWAY", 2: "TYPE_SURFACE_STREET", 3: "TYPE_BIKE_LANE"}

road_line_type = {
    0: "TYPE_UNKNOWN",
    1: "TYPE_BROKEN_SINGLE_WHITE",
    2: "TYPE_SOLID_SINGLE_WHITE",
    3: "TYPE_SOLID_DOUBLE_WHITE",
    4: "TYPE_BROKEN_SINGLE_YELLOW",
    5: "TYPE_BROKEN_DOUBLE_YELLOW",
    6: "TYPE_SOLID_SINGLE_YELLOW",
    7: "TYPE_SOLID_DOUBLE_YELLOW",
    8: "TYPE_PASSING_DOUBLE_YELLOW",
}

road_edge_type = {
    0: "TYPE_UNKNOWN",
    # // Physical road boundary that doesn't have traffic on the other side (e.g.,
    # // a curb or the k-rail on the right side of a freeway).
    1: "TYPE_ROAD_EDGE_BOUNDARY",
    # // Physical road boundary that separates the car from other traffic
    # // (e.g. a k-rail or an island).
    2: "TYPE_ROAD_EDGE_MEDIAN",
}

polyline_type = {
    # for lane
    "TYPE_UNDEFINED": -1,
    "TYPE_FREEWAY": 1,
    "TYPE_SURFACE_STREET": 2,
    "TYPE_BIKE_LANE": 3,
    # for roadline
    "TYPE_UNKNOWN": -1,
    "TYPE_BROKEN_SINGLE_WHITE": 6,
    "TYPE_SOLID_SINGLE_WHITE": 7,
    "TYPE_SOLID_DOUBLE_WHITE": 8,
    "TYPE_BROKEN_SINGLE_YELLOW": 9,
    "TYPE_BROKEN_DOUBLE_YELLOW": 10,
    "TYPE_SOLID_SINGLE_YELLOW": 11,
    "TYPE_SOLID_DOUBLE_YELLOW": 12,
    "TYPE_PASSING_DOUBLE_YELLOW": 13,
    # for roadedge
    "TYPE_ROAD_EDGE_BOUNDARY": 15,
    "TYPE_ROAD_EDGE_MEDIAN": 16,
    # for stopsign
    "TYPE_STOP_SIGN": 17,
    # for crosswalk
    "TYPE_CROSSWALK": 18,
    # for speed bump
    "TYPE_SPEED_BUMP": 19,
}


signal_state = {
    0: "LANE_STATE_UNKNOWN",
    # // States for traffic signals with arrows.
    1: "LANE_STATE_ARROW_STOP",
    2: "LANE_STATE_ARROW_CAUTION",
    3: "LANE_STATE_ARROW_GO",
    # // Standard round traffic signals.
    4: "LANE_STATE_STOP",
    5: "LANE_STATE_CAUTION",
    6: "LANE_STATE_GO",
    # // Flashing light signals.
    7: "LANE_STATE_FLASHING_STOP",
    8: "LANE_STATE_FLASHING_CAUTION",
}

signal_state_to_id = {}
for key, val in signal_state.items():
    signal_state_to_id[val] = key


def decode_tracks_from_proto(tracks):
    """Decodes agent tracks from Waymo scenario proto.

    Args:
        tracks: List of scenario_pb2.Track objects.

    Returns:
        dict: Dictionary with keys 'object_id', 'object_type', and 'trajs' containing
            agent IDs, types, and trajectories as numpy arrays.
    """
    track_infos = {
        "object_id": [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
        "object_type": [],
        "trajs": [],
    }
    for cur_data in tracks:  # number of objects
        cur_traj = [
            np.array(
                [
                    x.center_x,
                    x.center_y,
                    x.center_z,
                    x.length,
                    x.width,
                    x.height,
                    x.heading,
                    x.velocity_x,
                    x.velocity_y,
                    x.valid,
                ],
                dtype=np.float32,
            )
            for x in cur_data.states
        ]
        cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp, 10)

        track_infos["object_id"].append(cur_data.id)
        track_infos["object_type"].append(object_type[cur_data.object_type])
        track_infos["trajs"].append(cur_traj)

    track_infos["trajs"] = np.stack(track_infos["trajs"], axis=0)  # (num_objects, num_timestamp, 9)
    return track_infos


def get_polyline_dir(polyline):
    """Computes direction vectors for each segment of a polyline.

    Args:
        polyline (np.ndarray): Array of polyline points (shape: [N, 3]).

    Returns:
        np.ndarray: Array of direction vectors (shape: [N, 3]).
    """
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def decode_map_features_from_proto(map_features):
    """Decodes map features from Waymo scenario proto.

    Args:
        map_features: List of scenario_pb2.MapFeature objects.

    Returns:
        dict: Dictionary containing map features (lanes, road lines, road edges, stop signs,
            crosswalks, speed bumps) and all polylines as numpy arrays.
    """
    map_infos = {"lane": [], "road_line": [], "road_edge": [], "stop_sign": [], "crosswalk": [], "speed_bump": []}
    polylines_list = []

    point_cnt = 0
    for cur_data in map_features:
        cur_info = {"id": cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info["speed_limit_mph"] = cur_data.lane.speed_limit_mph
            cur_info["type"] = lane_type[
                cur_data.lane.type
            ]  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane

            cur_info["interpolating"] = cur_data.lane.interpolating
            cur_info["entry_lanes"] = list(cur_data.lane.entry_lanes)
            cur_info["exit_lanes"] = list(cur_data.lane.exit_lanes)

            cur_info["left_boundary"] = [
                {
                    "start_index": x.lane_start_index,
                    "end_index": x.lane_end_index,
                    "feature_id": x.boundary_feature_id,
                    "boundary_type": x.boundary_type,  # roadline type
                }
                for x in cur_data.lane.left_boundaries
            ]
            cur_info["right_boundary"] = [
                {
                    "start_index": x.lane_start_index,
                    "end_index": x.lane_end_index,
                    "feature_id": x.boundary_feature_id,
                    "boundary_type": road_line_type[x.boundary_type],  # roadline type
                }
                for x in cur_data.lane.right_boundaries
            ]

            global_type = polyline_type[cur_info["type"]]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["lane"].append(cur_info)

        elif cur_data.road_line.ByteSize() > 0:
            cur_info["type"] = road_line_type[cur_data.road_line.type]

            global_type = polyline_type[cur_info["type"]]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["road_line"].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info["type"] = road_edge_type[cur_data.road_edge.type]

            global_type = polyline_type[cur_info["type"]]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["road_edge"].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info["lane_ids"] = list(cur_data.stop_sign.lane)
            point = cur_data.stop_sign.position
            cur_info["position"] = np.array([point.x, point.y, point.z])

            global_type = polyline_type["TYPE_STOP_SIGN"]
            cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)

            map_infos["stop_sign"].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type["TYPE_CROSSWALK"]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["crosswalk"].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type["TYPE_SPEED_BUMP"]
            cur_polyline = np.stack(
                [np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0
            )
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos["speed_bump"].append(cur_info)

        else:
            continue
            # print(cur_data)
            # raise ValueError

        polylines_list.append(cur_polyline)
        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    polylines = np.zeros((0, 7), dtype=np.float32)
    if len(polylines_list) == 0:
        print("No polylines found in the map features.")
        return map_infos

    polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    map_infos["all_polylines"] = polylines
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):
    """Decodes dynamic map states (e.g., traffic signals) from Waymo scenario proto.

    Args:
        dynamic_map_states: List of scenario_pb2.DynamicMapState objects.

    Returns:
        dict: Dictionary with lane IDs, signal states, and stop points for each timestep.
    """
    dynamic_map_infos = {"lane_id": [], "state": [], "stop_point": []}
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state, stop_point = [], [], []
        # Skip over empty ones
        if not len(cur_data.lane_states):
            continue
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])
            stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])

        dynamic_map_infos["lane_id"].append(np.array([lane_id]))
        dynamic_map_infos["state"].append(np.array([state]))
        dynamic_map_infos["stop_point"].append(np.array([stop_point]))

    return dynamic_map_infos


def process_waymo_data_with_scenario_proto(data_file, output_path=None):
    """Processes a single Waymo scenario proto file and saves parsed data.

    Args:
        data_file (str): Path to the .tfrecord scenario file.
        output_path (str, optional): Directory to save parsed scenario .pkl files.

    Returns:
        list: List of dictionaries with scenario metadata for each scenario in the file.
    """
    dataset = tf.data.TFRecordDataset(data_file, compression_type="")
    ret_infos = []
    for cnt, data in enumerate(dataset):
        info = {}
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data.numpy()))

        info["scenario_id"] = scenario.scenario_id
        info["timestamps_seconds"] = list(scenario.timestamps_seconds)  # list of int of shape (91)
        info["current_time_index"] = scenario.current_time_index  # int, 10
        info["sdc_track_index"] = scenario.sdc_track_index  # int
        info["objects_of_interest"] = list(scenario.objects_of_interest)  # list, could be empty list

        info["tracks_to_predict"] = {
            "track_index": [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
            "difficulty": [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict],
        }  # for training: suggestion of objects to train on, for val/test: need to be predicted

        track_infos = decode_tracks_from_proto(scenario.tracks)
        info["tracks_to_predict"]["object_type"] = [
            track_infos["object_type"][cur_idx] for cur_idx in info["tracks_to_predict"]["track_index"]
        ]

        # decode map related data
        map_infos = decode_map_features_from_proto(scenario.map_features)
        dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)

        save_infos = {"track_infos": track_infos, "dynamic_map_infos": dynamic_map_infos, "map_infos": map_infos}
        save_infos.update(info)

        output_file = os.path.join(output_path, f"sample_{scenario.scenario_id}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(save_infos, f)

        ret_infos.append(info)
    return ret_infos


def get_infos_from_protos(data_path, output_path=None, num_workers=8):
    """Processes all Waymo scenario proto files in a directory in parallel.

    Args:
        data_path (str): Directory containing .tfrecord scenario files.
        output_path (str, optional): Directory to save parsed scenario .pkl files.
        num_workers (int, optional): Number of parallel workers. Defaults to 8.

    Returns:
        list: List of dictionaries with scenario metadata for all scenarios.
    """
    from functools import partial

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    tf.config.set_visible_devices([], "GPU")
    func = partial(process_waymo_data_with_scenario_proto, output_path=output_path)

    src_files = glob.glob(os.path.join(data_path, "*.tfrecord*"))
    src_files.sort()

    # func(src_files[0])
    with multiprocessing.Pool(num_workers) as p:
        data_infos = list(tqdm(p.imap(func, src_files), total=len(src_files)))

    all_infos = [item for infos in data_infos for item in infos]
    return all_infos


def create_infos_from_protos(raw_data_path, output_path, num_workers=8):
    """Creates processed scenario info files from raw Waymo scenario protos.

    Args:
        raw_data_path (str): Path to directory with raw .tfrecord scenario files.
        output_path (str): Directory to save processed scenario info files.
        num_workers (int, optional): Number of parallel workers. Defaults to 8.

    Raises:
        ValueError: If the raw data path does not exist.
    """
    if not os.path.exists(raw_data_path):
        raise ValueError("The raw data path %s does not exist." % raw_data_path)
    os.makedirs(output_path, exist_ok=True)

    scenario_path = os.path.join(output_path, "scenarios")
    sample_infos = get_infos_from_protos(data_path=raw_data_path, output_path=scenario_path, num_workers=num_workers)
    sample_filename = os.path.join(output_path, "processed_scenario_samples_infos.pkl")
    with open(sample_filename, "wb") as f:
        pickle.dump(sample_infos, f)
    print("----------------Waymo info train file is saved to %s----------------" % sample_filename)


if __name__ == "__main__":
    """Entry point for preprocessing Waymo scenario protos.

    Usage:
        python waymo_preprocess.py <raw_data_path> <output_path>
    """
    create_infos_from_protos(raw_data_path=sys.argv[1], output_path=sys.argv[2], num_workers=8)
