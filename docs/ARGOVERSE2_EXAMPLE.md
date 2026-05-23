# Argoverse 2 Motion Forecasting Dataset: Example Pipeline Usage

## Overview

This guide demonstrates how to process and analyze scenarios from the [Argoverse 2 Motion Forecasting dataset](https://www.argoverse.org/av2.html#forecasting-link) using the provided pipeline. AV2 scenarios are already recorded at 10 Hz; no interpolation is required. Each scenario spans 11 seconds (110 timesteps): 50 timesteps of observed history (5 seconds) followed by 60 timesteps of future trajectory (6 seconds).

---

## Batch Processing: Multiple Scenarios (Hydra-based)

> **Note:** Hydra is required for this workflow.

### Prerequisite: Install Argoverse 2 Dependencies

The `[argoverse2]` extra requires Python 3.12+:

```bash
uv python pin 3.12
uv sync
uv pip install -e ".[argoverse2]"
```

---

### 1. Obtain Sample Data

1. **Download the motion forecasting dataset** from the [Argoverse 2 download page](https://www.argoverse.org/av2.html#download-link). For example, download the `val.tar` file and extract it:
   ```bash
   mkdir -p samples/argoverse2/raw
   tar -xf val.tar -C samples/argoverse2/raw/
   ```
   The directory should have the layout:
   ```
   samples/argoverse2/raw/
     val/
       {scenario_id}/
         scenario_{scenario_id}.parquet
         log_map_archive_{scenario_id}.json
   ```

2. **Create a sample subset** — randomly select 5000 scenarios from the extracted val set:
   ```bash
   mkdir -p samples/argoverse2/raw/sample
   ls samples/argoverse2/raw/val | shuf -n 5000 | xargs -I{} cp -r samples/argoverse2/raw/val/{} samples/argoverse2/raw/sample/
   ```
   This creates `samples/argoverse2/raw/sample/` containing 5000 randomly chosen scenario directories, which is what the pre-processor expects when `--split sample` is passed.

3. **Pre-process the data:**
   ```bash
   uv run python -m characterization.datasets.argoverse2_preprocess \
       ./samples/argoverse2/raw ./samples/argoverse2/ --split sample
   ```
   This reads all scenario directories from the `val` split, extracts agent trajectories and map polylines, and writes one `.pkl` file per scenario to `./samples/argoverse2/scenarios/`.

   A sample config file (`argoverse2_sample.yaml`) is provided under `config/paths` with local paths to the sample data.

   The setup uses ground truth data (`scenario_type: gt`) and computes critical features (`return_criterion: critical`).

---

### 2. Compute Features

```bash
uv run python -m characterization.run_processor \
    characterizer=individual_features paths=argoverse2_sample dataset=argoverse2 scenario_type=gt
uv run python -m characterization.run_processor \
    characterizer=interaction_features paths=argoverse2_sample dataset=argoverse2 scenario_type=gt
```

This step creates a `./cache` directory with temporary feature data:
- `./cache/conflict_points`: Conflict region info per scenario.
- `./cache/features/gt_critical_continuous`: Per-agent individual features per scenario.

---

### 3. Compute Scores

```bash
uv run python -m characterization.run_processor \
    characterizer=individual_scores paths=argoverse2_sample dataset=argoverse2 scenario_type=gt
uv run python -m characterization.run_processor \
    characterizer=interaction_scores paths=argoverse2_sample dataset=argoverse2 scenario_type=gt
uv run python -m characterization.run_processor \
    characterizer=safeshift_scores paths=argoverse2_sample dataset=argoverse2 scenario_type=gt
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
    paths=argoverse2_sample dataset=argoverse2 scenario_type=gt
```

Outputs are written to a timestamped folder under `./cache/analysis/`.

**Scenario visualization** (optional) — renders per-scenario visual outputs:

```bash
uv run python -m characterization.run_scenario_viz \
    paths=argoverse2_sample dataset=argoverse2 scenario_type=gt
```

Outputs are written to `./cache/analysis/scenario_viz/`.

---

For a comparison of AV2 against Waymo and nuScenes, and dataset-specific notes, see [DATASET_PREPARATION.md](DATASET_PREPARATION.md).
