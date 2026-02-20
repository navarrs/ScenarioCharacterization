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
uv run -m characterization.run_feature_analysis scenario_types=[gt] criteria=[critical_categorical]
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

**TBD**

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

**TBD**
