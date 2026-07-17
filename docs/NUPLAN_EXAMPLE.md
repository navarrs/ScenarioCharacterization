# nuPlan Dataset: Example Pipeline Usage

## Overview

This guide demonstrates how to process and analyze scenarios from the [nuPlan dataset](https://www.nuplan.org/) using the provided pipeline. nuPlan tracks are subsampled from the native 20 Hz rate to 10 Hz during preprocessing, producing 6-second scenarios (60 timesteps) with a 2-second observation window (21 timesteps). Because nuPlan provides tens of thousands of scenarios, this dataset is the practical choice for a large-scale comparison (e.g. ~5,000 scenarios) against Waymo and Argoverse 2; nuScenes, by contrast, has only 850 annotated scenes.

---

## Batch Processing: Multiple Scenarios (Hydra-based)

> **Note:** Hydra is required for this workflow.

### Prerequisite: Install nuPlan Dependencies

The `[nuplan]` extra requires Python 3.10 (`nuplan-devkit` pins `numpy<2.0` and targets Python 3.9/3.10). Because its dependency pins conflict with the other extras, install nuPlan in a dedicated environment:

```bash
uv python pin 3.10
uv sync
uv pip install -e ".[nuplan]"
```

---

### 1. Obtain Sample Data

1. **Register and accept nuPlan's terms of use** at [nuplan.org](https://www.nuplan.org/nuplan#download).

2. **Download the nuPlan Mini split** (SQLite `.db` log files) and the **nuPlan Maps** pack from the Download section. The mini split is the recommended starting point and contains tens of thousands of scenarios. Extract them, for example:
   ```bash
   mkdir -p samples/nuplan/raw
   # extract the mini .db logs under samples/nuplan/raw/splits/mini
   # extract the maps under samples/nuplan/raw/maps
   ```
   The preprocessor reads the auto-labeled object tracks and vector map geometry; raw sensor blobs are **not required**.

3. **Pre-process the data:**
   ```bash
   uv run python -m characterization.datasets.nuplan_preprocess \
       --data-root ./samples/nuplan/raw/splits/mini \
       --map-root ./samples/nuplan/raw/maps \
       --output-path ./samples/nuplan/ \
       --limit 5000
   ```
   This selects up to `--limit` scenarios, subsamples trajectories to 10 Hz, extracts map polylines, and writes one `.pkl` file per scenario to `./samples/nuplan/scenarios/`. Use `--limit 5000` to match the Waymo and Argoverse 2 examples.

   A sample config file (`nuplan_sample.yaml`) is provided under `config/paths` with local paths to the sample data.

   The setup uses ground truth data (`scenario_type: gt`).

---

### 2. Compute Features

```bash
uv run python -m characterization.run_processor \
    characterizer=individual_features paths=nuplan_sample dataset=nuplan scenario_type=gt
uv run python -m characterization.run_processor \
    characterizer=interaction_features paths=nuplan_sample dataset=nuplan scenario_type=gt
```

This step creates a `./cache` directory with temporary feature data:
- `./cache/conflict_points`: Conflict region info per scenario.
- `./cache/features/gt_critical_continuous`: Per-agent individual features per scenario.

---

### 3. Compute Scores

```bash
uv run python -m characterization.run_processor \
    characterizer=individual_scores paths=nuplan_sample dataset=nuplan scenario_type=gt
uv run python -m characterization.run_processor \
    characterizer=interaction_scores paths=nuplan_sample dataset=nuplan scenario_type=gt
uv run python -m characterization.run_processor \
    characterizer=safeshift_scores paths=nuplan_sample dataset=nuplan scenario_type=gt
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
    paths=nuplan_sample dataset=nuplan scenario_type=gt
```

Outputs are written to a timestamped folder under `./cache/analysis/`.

**Scenario visualization** (optional) — renders per-scenario visual outputs:

```bash
uv run python -m characterization.run_scenario_viz \
    paths=nuplan_sample dataset=nuplan scenario_type=gt
```

Outputs are written to `./cache/analysis/scenario_viz/`.

---

## Notes on nuPlan vs Waymo

| Property | Waymo | nuPlan |
|---|---|---|
| Native frequency | 10 Hz | 20 Hz (subsampled to 10 Hz) |
| Ground truth timesteps | 91 (9.1 s) | 60 (6.0 s) |
| History timesteps | 11 (1.1 s) | 21 (2.0 s) |
| Dynamic map (traffic signals) | Yes | No (always empty) |
| Speed limits in map | Yes (mph) | Yes (mph, from lane speed limits) |
| Difficulty ratings | 0/1/2 (easy/medium/hard) | Uniform (all 1.0) |
| Required Python version | 3.10 | 3.10 |
