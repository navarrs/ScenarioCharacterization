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
uv run python -m characterization.run_feature_analysis
```

Run GT categorical feature analysis only:
```bash
uv run python -m characterization.run_feature_analysis scenario_types="['gt']" criteria="['critical_categorical']"
```

Run a smaller subset of scenarios for quick iteration:
```bash
uv run python -m characterization.run_feature_analysis total_scenarios=200 exp_tag=quick_check
```

Run without KDE and percentile overlays:
```bash
uv run python -m characterization.run_feature_analysis show_kde=false show_percentiles=false exp_tag=minimal_plots
```

Change output location and DPI:
```bash
uv run python -m characterization.run_feature_analysis output_dir=./outputs/feature_analysis_test dpi=200 exp_tag=feature_debug
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
| <img width="300" height="200" alt="individual_speed_limit_diff_type_vehicle_distributions" src="https://github.com/user-attachments/assets/be2f6bbf-3ae3-400d-840d-35ac43758605" /> <!-- pragma: allowlist secret -->  | <img width="300" height="200" alt="individual_speed_limit_diff_type_cyclist_distributions" src="https://github.com/user-attachments/assets/189d1c0c-7627-4d94-8a51-7f975264a3e1" /> <!-- pragma: allowlist secret -->  | <img width="300" height="200" alt="individual_speed_limit_diff_type_pedestrian_distributions" src="https://github.com/user-attachments/assets/5e7c58bc-5636-4dbc-a7df-d6d6e360d2e8" /> <!-- pragma: allowlist secret -->  |
| | | |

## Dataset Analysis

The dataset analysis utility counts how many individual trajectories and interaction pairs each dataset
contributes, broken down by agent type and agent-pair type. Both the population the pipeline enumerated
and the subset it successfully characterized come from a single streaming pass over the cached feature
artifacts.

### What this produces

Each run writes a timestamped folder under `output_dir` (default: `${paths.cache_path}/analysis`) with:
- `dataset_counts.csv` — one row per dataset, every agent type and pair type, `total` and `valid` columns
- `dataset_counts.json` — the same counts, nested by dataset label
- `dataset_counts_per_scenario.csv` — one row per scenario, so the choice of summary statistic can be
  revisited without re-reading every feature pickle
- `dataset_counts.tex` — booktabs table of the valid counts, ready to `\input` (needs `booktabs` and `graphicx`)
- `trajectory_counts.png`, `interaction_counts.png` — grouped bars, one hue per dataset, log y-axis
- `trajectory_composition.png`, `interaction_composition.png` — 100%-stacked bars of the class mix
- `trajectory_distribution.png`, `interaction_distribution.png` — per-scenario count distributions as
  boxes, symlog y-axis. These counts are right-skewed and skewed to different degrees across datasets,
  so the table reports an interquartile mean rather than a mean, and these plots show the spread behind it

### Example usage

Count a single dataset:
```bash
uv run python -m characterization.run_dataset_analysis paths=waymo_sample dataset=waymo
```

Count several datasets into one table, with LaTeX row labels:
```bash
uv run python -m characterization.run_dataset_analysis \
    "datasets=[{label: Waymo, dataset_name: waymo, latex_label: '\womd~\cite{ettinger2021large}'}, \
               {label: Argoverse2, dataset_name: argoverse2, latex_label: '\argoverse~\cite{wilson2021argoverse}'}, \
               {label: nuPlan, dataset_name: nuplan, latex_label: '\nuplan~\cite{karnchanachari2024nuplan}'}]"
```

`latex_label` is passed through verbatim, so the citation keys must match the manuscript's `ref.bib`.

Include the VRU-only pair types in the table and plots:
```bash
uv run python -m characterization.run_dataset_analysis \
    latex_pair_types="[TYPE_VEHICLE_VEHICLE,TYPE_VEHICLE_PEDESTRIAN,TYPE_VEHICLE_CYCLIST,TYPE_PEDESTRIAN_PEDESTRIAN,TYPE_PEDESTRIAN_CYCLIST,TYPE_CYCLIST_CYCLIST]"
```

Emit exact numbers instead of `300k` / `3.6M`:
```bash
uv run python -m characterization.run_dataset_analysis latex_compact_numbers=false
```

Report dataset totals alone, instead of `IQM (total)`:
```bash
uv run python -m characterization.run_dataset_analysis latex_per_scenario=false
```

### Useful config overrides

Commonly overridden keys:
- `features_path` (default: `${paths.cache_path}/features`)
- `scenario_types`, `criteria` (exactly one of each — see Notes)
- `total_scenarios` (limit scenario count for faster runs)
- `latex_agent_types`, `latex_pair_types` (which types the table and plots render)
- `latex_compact_numbers`, `latex_per_scenario`, `count_scenarios_on_disk`
- `output_dir`, `exp_tag`, `dpi`

### Notes

- **Exactly one `scenario_types` x `criteria` branch per run.** Feature artifacts are stored per branch,
  so counting several would count every scenario once per branch. The script raises rather than
  silently multiplying the numbers; re-run once per branch instead.
- **`total` means enumerated by the pipeline, not present in the raw dataset.** For trajectories it is
  the full `agent_types` list; for interactions it is every pair in `interaction_status`, which is
  `N-1` under `pair_scope=ego` and `C(N,2)` under `pair_scope=all`. `SafeShiftFeatures` derives the
  scope from `score_weighting_method`, whose shipped default `distance_to_ego_agent` gives
  `pair_scope=ego`. Compare `num_scenarios_counted` against `num_scenarios_on_disk` to see whether
  feature computation covered the whole dataset.
- **A pair is valid when its `interaction_status` is `COMPUTED_OK` or `PARTIAL_INVALID_HEADING`** — the
  same filter the interaction feature distributions use, so the counts match those plots exactly.
- **The ego agent is counted in its own `TYPE_EGO_AGENT` trajectory bucket**, but `AgentPairType` has no
  ego member and `get_agent_pair_type` folds ego into vehicle, so ego pairs land in V-V / V-P / V-C. The
  CSV and JSON carry `pairs_with_ego_total` / `pairs_with_ego_valid` so this stays visible. When
  `pairs_with_ego_valid` equals the sum over pair types, every pair contains the ego and the table
  relabels those columns `E-V` / `E-P` / `E-C`. A mismatch means the cache holds all-pairs artifacts.
- Composition plots take shares over the **selected** types only; excluding a type also removes it from
  the denominator.

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
uv run python -m characterization.run_score_analysis
```

Run GT categorical analysis only:
```bash
uv run python -m characterization.run_score_analysis scenario_types="['gt']" criteria="['critical_categorical']"
```

Run a smaller subset of scenarios for quick iteration, and add an experiment tag to further identify the output folder:
```bash
uv run python -m characterization.run_score_analysis total_scenarios=200 exp_tag=quick_check
```

Run only selected score heads:
```bash
uv run python -m characterization.run_score_analysis scores=[individual,interaction] exp_tag=ind_inter
```

Change output location, split percentile, and plot DPI:
```bash
uv run python -m characterization.run_score_analysis output_dir=./outputs/analysis_test test_percentile=90 dpi=200 tag=gt_cat
```

Change default paths fields:
```bash
uv run python -m characterization.run_score_analysis paths=test exp_tag=score_cat criteria="['critical_categorical']" paths.base_path=/data/driving/scenario_characterization
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

## Probe Analysis

The probe analysis utility ingests the `probe_summary.csv` produced by `run_processor` with `characterizer=cvm_probe` and
generates a suite of statistical plots characterising probe outcomes, score shifts, and affected agent distributions.

### What this produces

Each run writes to `output_dir` (default: `${paths.cache_path}/analysis`) with files such as:
- `probe_outcome_pie.png` — pie chart: *No probe* / *Ego probe* / *Non-ego probe*
- `score_distributions.png` — overlapping KDE density curves for `score_before` and `score_after`
- `score_delta_density.png` — KDE/histogram of `score_delta` with mean and median markers
- `score_delta_by_agent_type.png` — violin plot of `score_delta` split by ego vs non-ego
- `score_scatter.png` — scatter of `score_before` vs `score_after` with a y=x reference line
- `affected_agents_histogram.png` — stacked histogram of affected-agent count per probe
- `probe_analysis_summary.json` — aggregate stats (probe rate, ego/non-ego counts, score_delta stats)

### Prerequisites

- A `probe_summary.csv` produced by `run_processor` with `characterizer=cvm_probe` (default location: `${paths.cache_path}/probes/constant_velocity/probe_summary.csv`).

### Example usage

Run with the default `probe_csv` path:
```bash
uv run python -m characterization.run_probe_analysis
```

Point to a specific CSV and output directory:
```bash
uv run python -m characterization.run_probe_analysis \
    probe_csv=data/probed/constant_velocity/probe_summary.csv \
    output_dir=data/probe_analysis
```

Disable the timestamp subfolder and set a custom experiment tag:
```bash
uv run python -m characterization.run_probe_analysis \
    probe_csv=data/probed/constant_velocity/probe_summary.csv \
    add_timestamp=false exp_tag=run1
```

### Useful config overrides

Commonly overridden keys (from `config/run_analysis.yaml`):
- `probe_csv` (default: `${paths.cache_path}/probes/constant_velocity/probe_summary.csv`)
- `output_dir` (default: `${paths.cache_path}/analysis`)
- `dpi` (default: `300`)
- `add_timestamp`, `exp_tag`

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
uv run python -m characterization.run_scenario_viz
```

Visualize only scenarios with score files, limited to 200 scenarios:
```bash
uv run python -m characterization.run_scenario_viz viz_scored_scenarios=true total_scenarios=200
```

Group outputs by custom percentile bins:
```bash
uv run python -m characterization.run_scenario_viz organize_by_percentile=true percentiles=[10,50,80]
```

Visualize a different score head using a specific score tag:
```bash
uv run python -m characterization.run_scenario_viz scores_tag=gt_critical_categorical score_to_visualize=interaction
```

Visualize specific scenarios only (`scenario_id` takes a single ID or a list). These two are the scenarios used in the
manuscript figure `scenario_viz.png`:
```bash
uv run python -m characterization.run_scenario_viz paths=waymo viz=all_panes_scenario \
	'scenario_id=[1ad304afd32ea9a6,63dc1d4d391ccd24]'
```

Write visualizations to a custom location:
```bash
uv run python -m characterization.run_scenario_viz scenario_viz_dir=./outputs/scenario_viz_test
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
| <img width="200" height="200" alt="aa3c7fe966200717" src="https://github.com/user-attachments/assets/3a7b583d-1161-4c64-a074-b48d56cb8c91" /> <!-- pragma: allowlist secret -->  | <img width="200" height="200" alt="3ebeef67db72c170" src="https://github.com/user-attachments/assets/8807a99f-5148-4691-8f9e-c1e96c706d18" /> <!-- pragma: allowlist secret -->  | <img width="200" height="200" alt="6e593bf6b9dbbf73" src="https://github.com/user-attachments/assets/06c0598f-3145-4b75-b2aa-a66cccde0638" /> <!-- pragma: allowlist secret -->  | <img width="200" height="200" alt="937761acb6800cab" src="https://github.com/user-attachments/assets/36e29c42-484b-41f2-bd2a-2004ee6a02ec" /> <!-- pragma: allowlist secret -->  |
| | | | |

#### Categorical Scenarios

| | | |
|---|---|---|
| <img width="270" height="270" alt="4dc0cacf62cfdb09_2 7" src="https://github.com/user-attachments/assets/ed9cacaa-df14-43d2-ba79-a70015d512d5" /> <!-- pragma: allowlist secret -->  | <img width="270" height="270" alt="5c1f8d26c481e36d_2 43" src="https://github.com/user-attachments/assets/2e078a15-34e3-40d8-b854-776c3cdbce3c" /> <!-- pragma: allowlist secret -->  | <img width="270" height="270" alt="1068c27cceb21de5_3 5" src="https://github.com/user-attachments/assets/60c0079f-d5ed-423c-bc1d-8cd6b55ac76d" /> <!-- pragma: allowlist secret -->  |
| | | |
