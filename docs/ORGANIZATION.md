# Repository Organization

## Configuration Files (Using Hydra)

The main configuration files are:

1. **`run_processor.yaml`**
   Used for computing scenario features and scores.

2. **`run_analysis.yaml`**
   Used for analyzing features and scores distributions, and for scenario visualization.

Both configuration files are built hierarchically from the following components:

- **`characterizer`**: Specifies the type of characterization to run (e.g., features, scores).
- **`dataset`**: Defines which dataset adapter to use.
- **`paths`**: Sets the input and output data paths.
- **`processor`**: Determines the type of processor to run. Currently, `feature` and `score` processors are supported.
- **`viz`**: Configures scenario visualization settings.

---

## Switching Datasets, Feature Extractors, and Scorers

All scripts accept Hydra overrides on the command line. The key groups and their available options are:

### Dataset (`dataset=`)

| Value | Description |
|---|---|
| `waymo` | Waymo Open Motion Dataset (default) |
| `nuscenes` | nuScenes dataset |

### Characterizer (`characterizer=`)

| Value | Description |
|---|---|
| `individual_features` | Per-agent kinematic and behavioral features |
| `interaction_features` | Agent-pair interaction features |
| `safeshift_features` | Combined individual + interaction features |
| `individual_scores` | Scores derived from individual agent features |
| `interaction_scores` | Scores derived from interaction features |
| `safeshift_scores` | Combined individual + interaction scores |
| `individual_scores_categorical` | Categorical variant of individual scores |
| `interaction_scores_categorical` | Categorical variant of interaction scores |
| `safeshift_scores_categorical` | Categorical variant of safeshift scores |

### Paths (`paths=`)

| Value | Description |
|---|---|
| `default` | Default local paths |
| `waymo_sample` | Paths for the Waymo sample data under `samples/` |
| `nuscenes_sample` | Paths for the nuScenes sample data under `samples/nuscenes/` |

### Example

To run interaction feature extraction on nuScenes sample data:

```bash
uv run python -m characterization.run_processor \
    dataset=nuscenes characterizer=interaction_features paths=nuscenes_sample
```

To run safeshift scoring on Waymo with categorical output:

```bash
uv run python -m characterization.run_processor \
    dataset=waymo characterizer=safeshift_scores_categorical paths=waymo_sample
```
