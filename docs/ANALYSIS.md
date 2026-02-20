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

#### Score Density Plot

Shows the score density over a set of scenarios across our three scoring axes (individual, interaction, safeshift).

![Alt text](https://private-user-images.githubusercontent.com/24197463/552908643-c8857deb-9a9d-4884-9099-1e7c6bc8c0a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE2MjIwMjEsIm5iZiI6MTc3MTYyMTcyMSwicGF0aCI6Ii8yNDE5NzQ2My81NTI5MDg2NDMtYzg4NTdkZWItOWE5ZC00ODg0LTkwOTktMWU3YzZiYzhjMGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIwVDIxMDg0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUyZmRiZmI2OGEwOGFhOTgzM2E2OWQ0ZDE5Y2FmOTU5MjIxNDhkN2MzNTRlZGNiMjM4MmM0YmI5MWVhNjk1NjMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.dWE4ZEK1jRrvKGyQieYWgmha1S_3xLZSifUFwx5jZlI) <!-- pragma: allowlist secret -->


#### Categorical Density Plots

Shows the score density over a set of scenarios across either in 2D (individual, interaction) or 3D (individual, interaction, safeshift).

![Alt text](https://private-user-images.githubusercontent.com/24197463/552908858-a6298513-da5e-42d8-97b6-8db2cdac5741.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE2MjI0NDMsIm5iZiI6MTc3MTYyMjE0MywicGF0aCI6Ii8yNDE5NzQ2My81NTI5MDg4NTgtYTYyOTg1MTMtZGE1ZS00MmQ4LTk3YjYtOGRiMmNkYWM1NzQxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIwVDIxMTU0M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWVlMTEwZGMxYjFmM2ZhNGE3OTFjY2JlZTNlYTk3MDAzMDM5YTQ1NTI5Yjk4NDBhZjQxNzAxNTA5NjQzMmI5ZTEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.fbBetv6uJla2KLQxVpara-C67EN7HOeQJAuxcb-XdDY) <!-- pragma: allowlist secret -->

![Alt text](https://private-user-images.githubusercontent.com/24197463/552908837-6a5bf40e-d261-4312-97af-2b3bdff18751.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE2MjI0NDMsIm5iZiI6MTc3MTYyMjE0MywicGF0aCI6Ii8yNDE5NzQ2My81NTI5MDg4MzctNmE1YmY0MGUtZDI2MS00MzEyLTk3YWYtMmIzYmRmZjE4NzUxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIwVDIxMTU0M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPThkOTMwMmZjYzViODRmOTU3OThiOGM4YzhjZTc1OTk2ZGVmOWZkZDg1ZmMwOTU3YTQxOGIxOWY0YjdjOTRkZjImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.QD5FdoAD0b2AvaPJC9sWCnLMf_xFhX0DST9wpJy_UBM) <!-- pragma: allowlist secret -->
