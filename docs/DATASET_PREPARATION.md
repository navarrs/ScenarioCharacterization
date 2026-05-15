# Dataset Preparation

This page explains how to obtain and preprocess each supported dataset for use with the pipeline. For the full step-by-step pipeline walkthrough (feature computation, scoring, visualization) see the individual dataset guides below.

- [Waymo Open Motion Dataset](WAYMO_EXAMPLE.md)
- [nuScenes](NUSCENES_EXAMPLE.md)
- [Argoverse 2 Motion Forecasting](ARGOVERSE2_EXAMPLE.md)

---

## Dataset Comparison

| Property | Waymo | nuScenes | Argoverse 2 |
|---|---|---|---|
| Native frequency | 10 Hz | 2 Hz (interpolated to 10 Hz) | 10 Hz |
| Total timesteps | 91 (9.1 s) | 60 (6.0 s) | 110 (11.0 s) |
| History timesteps | 11 (1.1 s) | 21 (2.0 s) | 50 (5.0 s) |
| `current_time_index` | 10 | 20 | 49 |
| Dynamic map (traffic signals) | Yes | No (always empty) | No (always empty) |
| Speed limits in map | Yes (mph) | No (set to 0) | No (set to 0) |
| Bounding box dimensions | Per agent (from proto) | Per agent (from annotations) | Per type (fixed defaults) |
| z coordinate | Per agent | Per agent | Always 0.0 |
| Road edges in map | Yes | Yes | No |
| Difficulty ratings | 0/1/2 (easy/medium/hard) | Uniform (all 1.0) | Uniform (all 1.0) |
| Agent relevance | Difficulty-weighted | Uniform (all 1.0) | 1.0 for FOCAL/SCORED, 0.0 otherwise |
| Required Python version | 3.10 | 3.12 | 3.12 |

---

## Dataset-Specific Notes

### Waymo

- Requires Python 3.10 (TensorFlow dependency).
- Preprocesses `.tfrecord` files using `waymo-open-dataset`.
- Provides per-timestep traffic signal states in `DynamicMapData`.
- Agent difficulty ratings (0/1/2) are used to weight `agent_relevance`.

### nuScenes

- Requires Python 3.12+ (`nuscenes-devkit` pins `numpy<2.0`).
- Native 2 Hz keyframes are interpolated to 10 Hz during preprocessing.
- No traffic signals; `DynamicMapData` fields are always `None`.
- All agents receive uniform `agent_relevance=1.0`.

### Argoverse 2

- Requires Python 3.12+.
- Already at 10 Hz; no interpolation required.
- `ObjectState.position` is 2D — all agent z values are set to 0.0.
- Per-agent bounding box sizes are not provided; type-based defaults are used (e.g., 4.5 × 2.0 × 1.7 m for vehicles).
- No dedicated road-edge layer; `road_edge_ids` / `road_edge_polyline_idxs` are `None`. Lane boundaries with solid markings are stored as `road_line` entries.
- Track categories (`FOCAL_TRACK`, `SCORED_TRACK`, `UNSCORED_TRACK`, `TRACK_FRAGMENT`) determine `agent_relevance`: FOCAL and SCORED tracks receive 1.0, others 0.0.
