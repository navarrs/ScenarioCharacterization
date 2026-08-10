# nuPlan Dataset: Example Pipeline Usage

## Overview

This guide demonstrates how to process and analyze scenarios from the [nuPlan dataset](https://www.nuplan.org/) using the provided pipeline. nuPlan tracks are subsampled from the native 20 Hz rate to 10 Hz during preprocessing, producing 6-second scenarios (60 timesteps) with a 2-second observation window (21 timesteps). nuPlan is the practical choice for a large-scale comparison (e.g. ~5,000 scenarios) against Waymo and Argoverse 2: the mini split alone yields on the order of ~15k scenarios (the exact count depends on the scenario filter), and the trainval split far more, whereas nuScenes has only 850 annotated scenes.

---

## Batch Processing: Multiple Scenarios (Hydra-based)

> **Note:** Hydra is required for this workflow.

### Prerequisite: Install nuPlan Dependencies

The `[nuplan]` extra runs on the same Python 3.12 as the rest of the pipeline. The PyPI `nuplan-devkit` wheel ships broken metadata that declares no dependencies, so the extra lists the devkit's runtime requirements explicitly — `uv pip install -e ".[nuplan]"` pulls everything needed:

```bash
uv python pin 3.12
uv sync
uv pip install -e ".[nuplan]"
```

---

### 1. Obtain Sample Data

1. **Register and accept nuPlan's terms of use** at [nuplan.org](https://www.nuplan.org/nuplan#download).

2. **Download the nuPlan Mini split** (SQLite `.db` log files) and the **nuPlan Maps** pack from the Download section. The mini split (64 logs, ~7 hours, ~15k scenarios) is the recommended starting point; use the trainval split if you need more. Unzip both archives under a single data root (`./samples/nuplan` below) — keep the folder names the archives create, no renaming needed. The result matches the standard devkit layout:
   ```text
   samples/nuplan/
   ├── maps/
   │   ├── nuplan-maps-v1.0.json
   │   └── <us-ma-boston | us-nv-las-vegas-strip | sg-one-north | us-pa-pittsburgh-hazelwood>/...
   └── nuplan-v1.1/
       └── splits/
           └── mini/
               ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
               └── ...
   ```
   The preprocessor reads the auto-labeled object tracks and vector map geometry; the `sensor_blobs` are **not required**.

3. **Pre-process the data:**
   ```bash
   uv run python -m characterization.datasets.nuplan_preprocess \
       --input_path ./samples/nuplan \
       --db_files ./samples/nuplan/nuplan-v1.1/splits/mini \
       --map_root ./samples/nuplan/maps \
       --map_version nuplan-maps-v1.0 \
       --output_path ./samples/nuplan/ \
       --limit 5000
   ```
   This selects up to `--limit` scenarios, subsamples trajectories to 10 Hz, extracts map polylines, and writes one `.pkl` file per scenario to `./samples/nuplan/scenarios/`. Use `--limit 5000` to match the Waymo and Argoverse 2 examples. To use the larger split instead, point `--db_files` at `.../nuplan-v1.1/splits/trainval`.

   > **First run can be slow, and that is expected.** Scenario selection scans every `.db` log before `--limit` is applied, and the first map load per city builds a spatial index (minutes per city; Las Vegas is the slowest). The progress bar may sit at `0` for several minutes on the first scenario — this is normal, not a hang. To sanity-check the pipeline quickly, point `--db_files` at a single small-city `.db` (e.g. Boston or Singapore, avoiding Las Vegas) with `--limit 5`. If it truly never advances, verify `--map_root` is the directory that directly contains `nuplan-maps-v1.0.json` and the city folders, and delete a stale `<map_root>/.maplocks` directory left by an interrupted run.

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
| Required Python version | 3.10 | 3.12 |
