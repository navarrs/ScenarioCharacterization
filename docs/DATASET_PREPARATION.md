# Dataset Preparation

This page explains how to obtain and preprocess each supported dataset for use with the pipeline. For the full step-by-step pipeline walkthrough (feature computation, scoring, visualization) see the individual dataset guides below.

- [Waymo Open Motion Dataset](WAYMO_EXAMPLE.md)
- [nuScenes](NUSCENES_EXAMPLE.md)
- [Argoverse 2 Motion Forecasting](ARGOVERSE2_EXAMPLE.md)
- [nuPlan](NUPLAN_EXAMPLE.md)

All preprocess scripts accept `--overwrite`, which removes the existing `scenarios/` directory and `processed_scenario_samples_infos.pkl` index under the output path before writing, clearing stale scenario pickles from a previous run. Without it, each pickle is overwritten in place and any stale files are left untouched.

## Scenario Selection

Preprocessing determines the *pool* of scenarios on disk; `num_scenarios` then draws a seeded random subset from that pool at load time, identically for every dataset (`sample_scenarios` in `utils/common.py`). The draw depends on both the seed and the pool contents, so re-preprocessing a dataset changes which subset comes out.

For a comparable ~5000-scenario set per dataset, each pool is obtained from a held-out split without downloading the full dataset:

| Dataset | Split | How the pool is obtained | Approx. pool size |
|---|---|---|---|
| Waymo | training | ~11 of 1000 tfrecord shards ([WAYMO_EXAMPLE.md](WAYMO_EXAMPLE.md)) | ~5,200 |
| Argoverse 2 | val | Random subset of the 24,988 extracted scenario directories ([ARGOVERSE2_EXAMPLE.md](ARGOVERSE2_EXAMPLE.md)) | 5,000 |
| nuPlan | val | Seeded, non-overlapping `--limit` draw over the `.db` logs ([NUPLAN_EXAMPLE.md](NUPLAN_EXAMPLE.md)) | mini caps at 3,473; 5,000 needs ~140 logs |
| nuScenes | trainval | All 1,000 scenes ([NUSCENES_EXAMPLE.md](NUSCENES_EXAMPLE.md)) | 850 |

The Argoverse 2 and nuPlan pool draws are not currently seeded, so they are random but not reproducible across preprocessing runs.

### Scenario overlap

The datasets differ in whether their pre-cut scenarios may overlap in time, which affects how many *independent* scenarios a pool really contains:

- **Argoverse 2** extracts non-overlapping 11-second windows by construction.
- **Waymo** cuts each 20-second segment into overlapping 9.1-second windows (validation/test at start offsets `{0, 5, 10}` s, training at `{0, 2, 4, 5, 6, 8, 10}` s). Sibling windows are spread across shards, so realised overlap scales with the fraction of shards downloaded — see the note in [WAYMO_EXAMPLE.md](WAYMO_EXAMPLE.md).
- **nuPlan** enumerates one candidate scenario per lidar frame (20 Hz), so consecutive candidates are near-duplicates and a uniform draw over them produces heavily overlapping scenarios.

Measured on the current ~5000-scenario pools (scenarios sharing at least 5 identical ego positions with another): Argoverse 2 2.4%, Waymo 4.0%, nuPlan 52.7%.

---

## Dataset Comparison

| Property | Waymo | nuScenes | Argoverse 2 | nuPlan |
|---|---|---|---|---|
| Native frequency | 10 Hz | 2 Hz (interpolated to 10 Hz) | 10 Hz | 20 Hz (subsampled to 10 Hz) |
| Total timesteps | 91 (9.1 s) | 60 (6.0 s) | 110 (11.0 s) | 60 (6.0 s) |
| History timesteps | 11 (1.1 s) | 21 (2.0 s) | 50 (5.0 s) | 21 (2.0 s) |
| `current_time_index` | 10 | 20 | 49 | 20 |
| Dynamic map (traffic signals) | Yes | No (always empty) | No (always empty) | No (always empty) |
| Speed limits in map | Yes (mph) | No (set to 0) | No (set to 0) | Yes (mph) |
| Bounding box dimensions | Per agent (from proto) | Per agent (from annotations) | Per type (fixed defaults) | Per agent (from tracks) |
| z coordinate | Per agent | Per agent | Always 0.0 | Always 0.0 |
| Road edges in map | Yes | Yes | No | Yes (roadblock polygons) |
| Difficulty ratings | 0/1/2 (easy/medium/hard) | Uniform (all 1.0) | Uniform (all 1.0) | Uniform (all 1.0) |
| Agent relevance | Difficulty-weighted | Uniform (all 1.0) | 1.0 for FOCAL/SCORED, 0.0 otherwise | Uniform (all 1.0) |
| Required Python version | 3.10 | 3.12 | 3.12 | 3.12 |

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

### nuPlan

- Runs on Python 3.12 (same environment as the rest of the pipeline). The PyPI `nuplan-devkit` wheel declares no dependencies, so the `[nuplan]` extra lists them explicitly.
- Native 20 Hz tracks are subsampled to 10 Hz (stride derived from `database_interval`); no interpolation required.
- The number processed is controlled by `--limit` (use `5000` to match Waymo/Argoverse 2), drawn with `--seed` from candidates spaced at least `--min_gap_seconds` apart so selected scenarios do not overlap. The mini split yields only 3,473 non-overlapping scenarios, so 5,000 requires a larger split such as `val`.
- Agent velocities are recomputed via finite differences to avoid mixing ego body-frame and agent global-frame conventions.
- No traffic signals; `DynamicMapData` fields are always `None`. All agents receive uniform `agent_relevance=1.0`.
- No dedicated road-line layer (`road_line` is empty); roadblock polygons stand in for `road_edge`.
