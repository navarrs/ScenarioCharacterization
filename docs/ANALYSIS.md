# Scenario Characterization Analysis

This assumes you have already generated feature and score artifacts with the processor pipeline.
If not, run those first using [CHARACTERIZATION](CHARACTERIZATION.md).

## Feature Analysis

The feature analysis utility loads cached feature artifacts, regroups features by agent or agent-pair type,
and generates feature distribution plots with optional KDE and percentile markers.

### What this produces

Each run writes a timestamped folder under `output_dir` (default: `${paths.cache_path}/analysis`) with files such as:
- `individual_<feature>_<agent_type>_distributions.png`
- `interaction_<feature>_<agent_pair_type>_distributions.png`
- `<agent_type>_feature_percentiles.json`
- `<agent_pair_type>_feature_percentiles.json`

### Example usage

Run with default config values from `src/characterization/config/run_analysis.yaml`:
```bash
uv run -m characterization.run_feature_analysis
```

Run GT categorical feature analysis only:
```bash
uv run -m characterization.run_feature_analysis scenario_types="['gt']" criteria="['critical_categorical']"
```

Run a smaller subset of scenarios for quick iteration:
```bash
uv run -m characterization.run_feature_analysis total_scenarios=200 exp_tag=quick_check
```

Run without KDE and percentile overlays:
```bash
uv run -m characterization.run_feature_analysis show_kde=false show_percentiles=false exp_tag=minimal_plots
```

Change output location and DPI:
```bash
uv run -m characterization.run_feature_analysis output_dir=./outputs/feature_analysis_test dpi=200 exp_tag=feature_debug
```

### Useful config overrides

Commonly overridden keys:
- `features_path` (default: `${paths.cache_path}/features`)
- `scenario_types` (e.g., `[gt]`, `[ho]`)
- `criteria` (e.g., `[critical_categorical]`, `[critical_continuous]`, `[average]`)
- `total_scenarios` (limit scenario count for faster runs)
- `show_kde`, `show_percentiles`
- `output_dir`, `exp_tag`, `dpi`

### Notes

- Scenario IDs are intersected across all selected `scenario_types x criteria`; missing artifacts in any selected
	branch can reduce the final set.

### Example outputs

#### Feature Distributions by Agent Type

| | | |
|---|---|---|
| **Vehicle** | **Cyclist** | **Pedestrian** |
| <img width="300" height="180" alt="individual_speed_limit_diff_type_vehicle_distributions" src="https://github.com/user-attachments/assets/be2f6bbf-3ae3-400d-840d-35ac43758605" /> <!-- pragma: allowlist secret -->  | <img width="300" height="180" alt="individual_speed_limit_diff_type_cyclist_distributions" src="https://github.com/user-attachments/assets/189d1c0c-7627-4d94-8a51-7f975264a3e1" /> <!-- pragma: allowlist secret -->  | <img width="300" height="180" alt="individual_speed_limit_diff_type_pedestrian_distributions" src="https://github.com/user-attachments/assets/5e7c58bc-5636-4dbc-a7df-d6d6e360d2e8" /> <!-- pragma: allowlist secret -->  |
| | | |

## Score Analysis

The score analysis utility loads cached score/feature artifacts, computes scenario-level summaries,
and generates score distribution plots and OOD split files.

### What this produces

Each run writes a timestamped folder under `output_dir` (default: `${paths.cache_path}/analysis`) with files such as:
- `scene_to_scores_mapping.csv`
- `<tag>_score_density_plot.png`
- `scenario_splits.json`
- `agent_score_distribution_<scenario_type>_<criterion>_<score>.png`
- `<scenario_type>_<criterion>_<score>.json` (agent score percentiles)
- For categorical criteria:
	- `agent_score_heatmap_<criterion>.png`
	- `agent_score_voxel_<criterion>.png`
	- `agent_score_voxel_<criterion>_<AgentType>.png`

### Example usage

Run with default config values from `src/characterization/config/run_analysis.yaml`:
```bash
uv run -m characterization.run_score_analysis
```

Run GT categorical analysis only:
```bash
uv run -m characterization.run_score_analysis scenario_types="['gt']" criteria="['critical_categorical']"
```

Run a smaller subset of scenarios for quick iteration, and add an experiment tag to further identify the output folder:
```bash
uv run -m characterization.run_score_analysis total_scenarios=200 exp_tag=quick_check
```

Run only selected score heads:
```bash
uv run -m characterization.run_score_analysis scores=[individual,interaction] exp_tag=ind_inter
```

Change output location, split percentile, and plot DPI:
```bash
uv run -m characterization.run_score_analysis output_dir=./outputs/analysis_test test_percentile=90 dpi=200 tag=gt_cat
```

Change default paths fields:
```bash
uv run -m characterization.run_score_analysis paths=test exp_tag=score_cat criteria="['critical_categorical']" paths.base_path=/data/driving/scenario_characterization
```

### Useful config overrides

Commonly overridden keys:
- `scores_path` (default: `${paths.cache_path}/scores`)
- `features_path` (default: `${paths.cache_path}/features`)
- `scenario_types` (e.g., `[gt]`, `[ho]`)
- `criteria` (e.g., `[critical_categorical]`, `[critical_continuous]`, `[average]`)
- `scores` (any subset of `[individual, interaction, safeshift]`)
- `total_scenarios` (limit scenario count for faster runs)
- `test_percentile` (OOD split threshold)
- `output_dir`, `tag`, `exp_tag`, `dpi`

### Notes

- Scenario IDs are intersected across all selected `scenario_types x criteria`; missing artifacts in any selected
	branch can reduce the final set.
- Heatmap/voxel outputs are generated only for criteria containing `categorical`.

### Example Outputs

#### Score Density Plot

Shows the score density over a set of scenarios across our three scoring axes (individual, interaction, safeshift).

<img width="600" height="360" alt="Image" src="https://github.com/user-attachments/assets/85d0b900-4d64-46b6-9de1-a3e4d64cafe1" />


#### Categorical Density Plots

Shows the score density over a set of scenarios across either in 2D (individual, interaction) or 3D (individual, interaction, safeshift).

| | |
|---|---|
| **2D Categorical Heatmap** | **3D Categorical Voxel Grid** |
| <img width="450" height="360" alt="Image" src="https://github.com/user-attachments/assets/91032ac3-de90-40ad-b08b-06beb437c767" /> <!-- pragma: allowlist secret -->  | <img width="450" height="360" alt="Image" src="https://github.com/user-attachments/assets/81a0848c-7716-410d-97f8-a99d18a29500" /> <!-- pragma: allowlist secret -->  |
| | |

## Scenario Visualizer

The scenario visualizer renders per-scenario visual outputs and can optionally bucket scenarios by score percentile.

### Prerequisites

- Scenario `.pkl` files available under `paths.scenario_base_path`.
- Score artifacts available under `${scores_path}/${scores_tag}` (when `viz_scored_scenarios=true`).
- A score-mapping CSV at `scenario_to_score_mapping_filepath` (required when `organize_by_percentile=true`).

The easiest way to generate compatible score artifacts and mapping CSV is to run score analysis first.

### What this produces

Each run writes to a timestamped folder under `scenario_viz_dir` (default: `${output_dir}/scenario_viz`), for example:
- `<timestamp>_<scores_tag>_<score_to_visualize>/`
- If `organize_by_percentile=true`, additional subfolders such as:
	- `percentile_0-10/`
	- `percentile_10-50/`
	- `percentile_50-80/`
	- `percentile_80-100/`
	- `unknown/` (scenario IDs missing from the mapping CSV)

### Example usage

Run scenario visualization with default config values from `src/characterization/config/run_analysis.yaml`:
```bash
uv run -m characterization.run_scenario_viz
```

Visualize only scenarios with score files, limited to 200 scenarios:
```bash
uv run -m characterization.run_scenario_viz viz_scored_scenarios=true total_scenarios=200
```

Group outputs by custom percentile bins:
```bash
uv run -m characterization.run_scenario_viz organize_by_percentile=true percentiles=[10,50,80]
```

Visualize a different score head using a specific score tag:
```bash
uv run -m characterization.run_scenario_viz scores_tag=gt_critical_categorical score_to_visualize=interaction
```

Write visualizations to a custom location:
```bash
uv run -m characterization.run_scenario_viz scenario_viz_dir=./outputs/scenario_viz_test
```

### Useful config overrides

Commonly overridden keys:
- `scenario_viz_dir` (default: `${output_dir}/scenario_viz`)
- `scores_path`, `scores_tag`
- `score_to_visualize` (one of `individual`, `interaction`, `safeshift`)
- `viz_scored_scenarios`
- `organize_by_percentile`, `percentiles`
- `scenario_to_score_mapping_filepath`
- `total_scenarios`

### Notes

- Percentile grouping uses the score column `<scores_tag>_<score_to_visualize>` in `scenario_to_score_mapping_filepath`.
- If `viz_scored_scenarios=false`, all scenarios under `paths.scenario_base_path` are eligible for visualization.

### Example Outputs

#### Scenarios Organized by Score Percentile

| | | | |
|---|---|---|---|
| **[0, 10)** | **[10, 50)** | **[50, 80)** | **[80, 100]** |
| <img width="160" height="160" alt="aa3c7fe966200717" src="https://github.com/user-attachments/assets/3a7b583d-1161-4c64-a074-b48d56cb8c91" /> <!-- pragma: allowlist secret -->  | <img width="160" height="160" alt="3ebeef67db72c170" src="https://github.com/user-attachments/assets/8807a99f-5148-4691-8f9e-c1e96c706d18" /> <!-- pragma: allowlist secret -->  | <img width="160" height="160" alt="6e593bf6b9dbbf73" src="https://github.com/user-attachments/assets/06c0598f-3145-4b75-b2aa-a66cccde0638" /> <!-- pragma: allowlist secret -->  | <img width="160" height="160" alt="937761acb6800cab" src="https://github.com/user-attachments/assets/36e29c42-484b-41f2-bd2a-2004ee6a02ec" /> <!-- pragma: allowlist secret -->  |
| | | | |

#### Categorical Scenarios

| | | |
|---|---|---|
| <img width="270" height="270" alt="4dc0cacf62cfdb09_2 7" src="https://github.com/user-attachments/assets/ed9cacaa-df14-43d2-ba79-a70015d512d5" /> <!-- pragma: allowlist secret -->  | <img width="270" height="270" alt="5c1f8d26c481e36d_2 43" src="https://github.com/user-attachments/assets/2e078a15-34e3-40d8-b854-776c3cdbce3c" /> <!-- pragma: allowlist secret -->  | <img width="270" height="270" alt="1068c27cceb21de5_3 5" src="https://github.com/user-attachments/assets/60c0079f-d5ed-423c-bc1d-8cd6b55ac76d" /> <!-- pragma: allowlist secret -->  |
| | | |
