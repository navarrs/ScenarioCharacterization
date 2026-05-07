# nuScenes Dataset: Example Pipeline Usage

## Overview

This guide demonstrates how to process and analyze scenarios from the [nuScenes dataset](https://www.nuscenes.org/) using the provided pipeline. nuScenes trajectories are interpolated from the native 2 Hz keyframe rate to 10 Hz during preprocessing, producing 6-second scenarios (60 timesteps) with a 2-second observation window (21 timesteps).

---

## Batch Processing: Multiple Scenarios (Hydra-based)

> **Note:** Hydra is required for this workflow.

### Prerequisite: Install nuScenes Dependencies

The `[nuscenes]` extra requires Python 3.12 (`nuscenes-devkit` pins `numpy<2.0`, which conflicts
with the Waymo extra's `numpy==1.21.5` on Python 3.10). Make sure Python 3.12 is active:

```bash
uv python pin 3.12
uv sync
uv pip install -e ".[nuscenes]"
```

---

### 1. Obtain Sample Data

1. **Register and accept nuScenes' terms of use** at [nuscenes.org](https://www.nuscenes.org/nuscenes#download).

2. **Download the `v1.0-mini` split** (the smallest available split, ~4 GB):
   ```bash
   mkdir -p samples/nuscenes/raw
   ```
   Download `v1.0-mini.tgz` from the nuScenes website and extract it:
   ```bash
   tar -xzf v1.0-mini.tgz -C samples/nuscenes/raw/
   ```
   The extracted directory should contain `v1.0-mini/` with `maps/`, `samples/`, and `sweeps/` subdirectories.

3. **Download the Map Expansion pack** from the nuScenes website (listed as "Map expansion pack (v1.3)" on the download page) and extract it into the same directory:
   ```bash
   unzip nuScenes-map-expansion-v1.3.zip -d samples/nuscenes/raw/
   ```
   This adds `maps/expansion/*.json` files containing the vector lane and road geometry required for polyline extraction.

4. **Pre-process the data:**
   ```bash
   uv run python -m characterization.datasets.nuscenes_preprocess \
       ./samples/nuscenes/raw ./samples/nuscenes/ v1.0-mini
   ```
   This reads all 10 scenes from the mini split, interpolates trajectories to 10 Hz, extracts map polylines, and writes one `.pkl` file per scene to `./samples/nuscenes/scenarios/`.

   A sample config file (`nuscenes_sample.yaml`) is provided under `config/paths` with local paths to the sample data.

   The setup uses ground truth data (`scenario_type: gt`) and computes critical features (`return_criterion: critical`).

---

### 2. Compute Features

```bash
uv run python -m characterization.run_processor \
    characterizer=individual_features paths=nuscenes_sample dataset=nuscenes scenario_type=gt
uv run python -m characterization.run_processor \
    characterizer=interaction_features paths=nuscenes_sample dataset=nuscenes scenario_type=gt
```

This step creates a `./cache` directory with temporary feature data:
- `./cache/conflict_points`: Conflict region info per scenario.
- `./cache/features/gt_critical`: Per-agent individual features per scenario.

---

### 3. Compute Scores

```bash
uv run python -m characterization.run_processor \
    characterizer=individual_scores paths=nuscenes_sample dataset=nuscenes scenario_type=gt
uv run python -m characterization.run_processor \
    characterizer=interaction_scores paths=nuscenes_sample dataset=nuscenes scenario_type=gt
uv run python -m characterization.run_processor \
    characterizer=safeshift_scores paths=nuscenes_sample dataset=nuscenes scenario_type=gt
```

---

### 4. Analyze and Visualize Scores

To visualize the scenarios the viz dependencies are required. Install them with:

```bash
uv pip install -e ".[viz]"
```

**Score analysis** — generates score density plots, a `scene_to_scores_mapping.csv`, and OOD split files:

```bash
uv run python -m characterization.run_score_analysis \
    paths=nuscenes_sample dataset=nuscenes scenario_type=gt
```

Outputs are written to a timestamped folder under `./cache/analysis/`.

**Scenario visualization** (optional) — renders per-scenario visual outputs:

```bash
uv run python -m characterization.run_scenario_viz \
    paths=nuscenes_sample dataset=nuscenes scenario_type=gt
```

Outputs are written to `./cache/analysis/scenario_viz/`.

---

## Notes on nuScenes vs Waymo

| Property | Waymo | nuScenes |
|---|---|---|
| Native frequency | 10 Hz | 2 Hz (interpolated to 10 Hz) |
| Ground truth timesteps | 91 (9.1 s) | 60 (6.0 s) |
| History timesteps | 11 (1.1 s) | 21 (2.0 s) |
| Dynamic map (traffic signals) | Yes | No (always empty) |
| Speed limits in map | Yes (mph) | No (set to 0) |
| Difficulty ratings | 0/1/2 (easy/medium/hard) | Uniform (all 1.0) |
| Required Python version | 3.10 | 3.12 |
