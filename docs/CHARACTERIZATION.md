# Scenario Characterization

The processor classes are designed to take a set of input scenarios and produce the specified characterization outputs.

---

## Feature Processor

The feature processor uses a feature class specified in the `characterizer` configuration to compute specialized features for input scenarios defined in the `paths` configuration.

**Example usage:**
```bash
uv run python -m characterization.run_processor characterizer=[feature_type]
```

Available feature groups (see `config/characterizer`):
- **`individual_features`**: Computes descriptors for individual agents.
- **`interaction_features`**: Computes descriptors for agent interactions.
- **`safeshift_features`**: Combines individual and interaction features.

### Individual Features

To run the individual features characterizer:
```bash
uv run python -m characterization.run_processor characterizer=individual_features
```

Currently supported features:
- Derived from [SafeShift](https://github.com/cmubig/SafeShift/tree/master):
  - Agent speed
  - Agent speed limit difference (difference between agent speed and speed limit)
  - Agent acceleration
  - Agent deceleration
  - Agent jerk
  - Agent waiting period (interval an agent waits near a conflict point)
- Derived from [UniTraj](https://github.com/vita-epfl/UniTraj/tree/main):
  - Agent trajectory type (stationary, straight, straight-right, straight-left, right turn, right u-turn, left turn, left u-turn)
  - Agent kalman difficulty (how difficult to predict is an agent's trajectory based on an estimated trajectory using Kalman filters)

### Interaction Features

To run the interaction features characterizer:
```bash
uv run python -m characterization.run_processor characterizer=interaction_features
```

Currently supported features:
- Derived from [SafeShift](https://github.com/cmubig/SafeShift/tree/master):
  - Agent-pair separation distance
  - Agent-pair intersection area
  - Collisions
  - Minimum Time to Conflict Point (mTTCP)
  - Time headway
  - Time to collision
  - Deceleration Rate to Avoid a Crash (DRAC)

---

## Score Processor

The score processor uses a list of features specified in the `characterizer` configuration to compute specialized scores for input scenarios.

**Example usage:**
```bash
uv run python -m characterization.run_processor characterizer=[score_type]
```

Available score groups (see `config/characterizer`):
- **`individual_scores`**: Computes agent and scenario scores from individual agent descriptors.
- **`interaction_scores`**: Computes agent and scenario scores from interaction descriptors.
- **`safeshift_scores`**: Combines individual and interaction scores.
- **`individual_scores_categorical`**, **`interaction_scores_categorical`**, **`safeshift_scores_categorical`**: Categorical variants that discretize scores into buckets. Use these with `feature_type=categorical` or when running the full categorical profiling pipeline.


## Categorical Profiling

Use the categorical profiling runner script to execute the full profiling pipeline (features, scores, and distribution analyses):

```bash
bash src/scripts/run_categorical_profiler.sh [options]
```

### Options

- `-D <dataset>`: Dataset name (default: `waymo`). Sets the default paths config and meta directory. Use `waymo` or `nuscenes`.
- `-p <paths_config>`: Paths configuration to use (overrides the `-D` default)
- `-d <meta_dir>`: Meta directory where analysis JSON files are copied (overrides the `-D` default)
- `-u <output_dir>`: Output directory for categorical profiling analyses (default: `outputs/categorical_profiler`)
- `-m <mode>`: Run mode, either `resume` (default) or `scratch`
- `-s <step>`: Repeat a specific step by number (see `-l` for the step list); ignores the progress file
- `-l`: List all steps with their numbers and exit
- `-c`: Create metadata for feature computation
- `-o`: Overwrite existing outputs
- `-n`: Dry run (print commands without executing)

### Common examples

Resume from the last completed step (default behavior):

```bash
bash src/scripts/run_categorical_profiler.sh -m resume
```

Run from scratch (clears progress and starts from step 1):

```bash
bash src/scripts/run_categorical_profiler.sh -m scratch
```

Run with metadata creation and overwrite enabled:

```bash
bash src/scripts/run_categorical_profiler.sh -c -o
```

Run with custom meta and output directories:

```bash
bash src/scripts/run_categorical_profiler.sh -d ./meta_custom -u outputs/categorical_profiler_custom
```

Preview commands without running:

```bash
bash src/scripts/run_categorical_profiler.sh -n
```

## Switching Datasets and Characterizers

All `run_processor` commands accept Hydra overrides for `dataset=`, `characterizer=`, and `paths=`. For example, to run individual features on nuScenes:

```bash
uv run python -m characterization.run_processor \
    dataset=nuscenes characterizer=individual_features paths=nuscenes_sample
```

See [ORGANIZATION.md](ORGANIZATION.md) for the full list of available values for each config group.

---

## ![TO-DO](https://img.shields.io/badge/status-TODO-red) Scenario Probing
