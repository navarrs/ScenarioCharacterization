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
| ![Alt text](https://private-user-images.githubusercontent.com/24197463/552924833-be2f6bbf-3ae3-400d-840d-35ac43758605.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE2MzE2MjEsIm5iZiI6MTc3MTYzMTMyMSwicGF0aCI6Ii8yNDE5NzQ2My81NTI5MjQ4MzMtYmUyZjZiYmYtM2FlMy00MDBkLTg0MGQtMzVhYzQzNzU4NjA1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIwVDIzNDg0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTAxYmNiOGI2YzdhZmFmYWFiMTA2NGE0MTgwNmIyOWUyYzczMTY0Y2M3NTZjMWU4NzFmNDk0MWQ0MWI0ODRmMjYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.hY-KyJfkrPQRgmIIti45Og1DMEFqpUZ7-AClnbZ1a3U) <!-- pragma: allowlist secret --> | ![Alt text](https://private-user-images.githubusercontent.com/24197463/552924897-189d1c0c-7627-4d94-8a51-7f975264a3e1.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE2MzE2MjEsIm5iZiI6MTc3MTYzMTMyMSwicGF0aCI6Ii8yNDE5NzQ2My81NTI5MjQ4OTctMTg5ZDFjMGMtNzYyNy00ZDk0LThhNTEtN2Y5NzUyNjRhM2UxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIwVDIzNDg0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTMyODg4MTRjNTUzNmIxZThjMGU5YjVmMDkwYzE1NWJmMGI4ZDJkOGNkY2IxY2M4MWUwOWFhYzQyMzdiNWVjYzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.LRy1FJLSnHHhgxz0B-hsKZJmxwCJ0OJsPhg2u9pJm4o) <!-- pragma: allowlist secret --> | ![Alt text](https://private-user-images.githubusercontent.com/24197463/552924995-5e7c58bc-5636-4dbc-a7df-d6d6e360d2e8.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE2MzE2MjEsIm5iZiI6MTc3MTYzMTMyMSwicGF0aCI6Ii8yNDE5NzQ2My81NTI5MjQ5OTUtNWU3YzU4YmMtNTYzNi00ZGJjLWE3ZGYtZDZkNmUzNjBkMmU4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIwVDIzNDg0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTI4MzM5YWU1MWRiMGE1MWIzZjU1OWI4NmFlZWY0NjRlZDM5MmEyN2UwZmY4MDUwYjk3MWFiOTE2MDNkY2I3NGUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.lEpg7MUJgo8WkSpTpkbvbYgIqPcmrDUFkRX_aiLL6f8) <!-- pragma: allowlist secret --> |
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

![Alt text](https://private-user-images.githubusercontent.com/24197463/552908643-c8857deb-9a9d-4884-9099-1e7c6bc8c0a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE2MjIwMjEsIm5iZiI6MTc3MTYyMTcyMSwicGF0aCI6Ii8yNDE5NzQ2My81NTI5MDg2NDMtYzg4NTdkZWItOWE5ZC00ODg0LTkwOTktMWU3YzZiYzhjMGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIwVDIxMDg0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUyZmRiZmI2OGEwOGFhOTgzM2E2OWQ0ZDE5Y2FmOTU5MjIxNDhkN2MzNTRlZGNiMjM4MmM0YmI5MWVhNjk1NjMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.dWE4ZEK1jRrvKGyQieYWgmha1S_3xLZSifUFwx5jZlI) <!-- pragma: allowlist secret -->


#### Categorical Density Plots

Shows the score density over a set of scenarios across either in 2D (individual, interaction) or 3D (individual, interaction, safeshift).

| | |
|---|---|
| **2D Categorical Heatmap** | **3D Categorical Voxel Grid** |
| ![Alt text](https://private-user-images.githubusercontent.com/24197463/552908858-a6298513-da5e-42d8-97b6-8db2cdac5741.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE2MjI0NDMsIm5iZiI6MTc3MTYyMjE0MywicGF0aCI6Ii8yNDE5NzQ2My81NTI5MDg4NTgtYTYyOTg1MTMtZGE1ZS00MmQ4LTk3YjYtOGRiMmNkYWM1NzQxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIwVDIxMTU0M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWVlMTEwZGMxYjFmM2ZhNGE3OTFjY2JlZTNlYTk3MDAzMDM5YTQ1NTI5Yjk4NDBhZjQxNzAxNTA5NjQzMmI5ZTEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.fbBetv6uJla2KLQxVpara-C67EN7HOeQJAuxcb-XdDY) <!-- pragma: allowlist secret --> | ![Alt text](https://private-user-images.githubusercontent.com/24197463/552908837-6a5bf40e-d261-4312-97af-2b3bdff18751.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE2MjI0NDMsIm5iZiI6MTc3MTYyMjE0MywicGF0aCI6Ii8yNDE5NzQ2My81NTI5MDg4MzctNmE1YmY0MGUtZDI2MS00MzEyLTk3YWYtMmIzYmRmZjE4NzUxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIwVDIxMTU0M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPThkOTMwMmZjYzViODRmOTU3OThiOGM4YzhjZTc1OTk2ZGVmOWZkZDg1ZmMwOTU3YTQxOGIxOWY0YjdjOTRkZjImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.QD5FdoAD0b2AvaPJC9sWCnLMf_xFhX0DST9wpJy_UBM) <!-- pragma: allowlist secret --> |
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
| ![Alt text](https://private-user-images.githubusercontent.com/24197463/553595374-3a7b583d-1161-4c64-a074-b48d56cb8c91.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE4NjU1NjIsIm5iZiI6MTc3MTg2NTI2MiwicGF0aCI6Ii8yNDE5NzQ2My81NTM1OTUzNzQtM2E3YjU4M2QtMTE2MS00YzY0LWEwNzQtYjQ4ZDU2Y2I4YzkxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIzVDE2NDc0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNkZjRmOGI4YTJhNjE3YTM0N2E0NWEyMzhjODQyOWVhMWQxM2FmMjgzMjhkNDRhMjFlOGI3YmE2OTRmODE1YTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.7GJEhkwbpIItE3HXO8pzWpS-vUbBgQkQNIlt34euA6c) <!-- pragma: allowlist secret --> | ![Alt text](https://private-user-images.githubusercontent.com/24197463/553595887-8807a99f-5148-4691-8f9e-c1e96c706d18.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE4NjU1NjIsIm5iZiI6MTc3MTg2NTI2MiwicGF0aCI6Ii8yNDE5NzQ2My81NTM1OTU4ODctODgwN2E5OWYtNTE0OC00NjkxLThmOWUtYzFlOTZjNzA2ZDE4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIzVDE2NDc0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBjMWRiOTkwNDRkOGU3N2ZmZDM1NWJmMGU3ODEyMTZiMTc4ZWQ3NmYzMGRlNDVkMWJhZjVkYTQ3MmNjNTM4NTkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.9twScS36TIj2impovBl_Tr_F32I_WRuT3KPaatxdqeY) <!-- pragma: allowlist secret --> | ![Alt text](https://private-user-images.githubusercontent.com/24197463/553596187-06c0598f-3145-4b75-b2aa-a66cccde0638.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE4NjU1NjIsIm5iZiI6MTc3MTg2NTI2MiwicGF0aCI6Ii8yNDE5NzQ2My81NTM1OTYxODctMDZjMDU5OGYtMzE0NS00Yjc1LWIyYWEtYTY2Y2NjZGUwNjM4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIzVDE2NDc0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTg1NWFkOTlmMjUwYjg5N2Y3Y2Q0OTU5ZjhmMzI1MjIwZTZiN2NjNThiMWZmODkyMDA3NmYyZjBiNmUwZjY2NTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.ENVgW3JgG8l5hzWDodOsQJySWQ7s2NBcNY7z61zEik0) <!-- pragma: allowlist secret --> | ![Alt text](https://private-user-images.githubusercontent.com/24197463/553597345-36e29c42-484b-41f2-bd2a-2004ee6a02ec.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE4NjU1NjIsIm5iZiI6MTc3MTg2NTI2MiwicGF0aCI6Ii8yNDE5NzQ2My81NTM1OTczNDUtMzZlMjljNDItNDg0Yi00MWYyLWJkMmEtMjAwNGVlNmEwMmVjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIzVDE2NDc0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM2YWNhOGQ0MzI4ODZjYTNjZjliYmY5ODYwMGRiMTdkNmM5ZDFhODZkNWVmYWQxMWFjYTMxNGFkMzc1NzA1MjcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.wNMHylIJRwnlTwxMBvpiW9OH8kkaqFGKyNSC3CVGJS0) <!-- pragma: allowlist secret --> |
| | | | |

#### Categorical Scenarios

| | | |
|---|---|---|
| ![Alt text](https://private-user-images.githubusercontent.com/24197463/553599381-ed9cacaa-df14-43d2-ba79-a70015d512d5.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE4NjU1NjIsIm5iZiI6MTc3MTg2NTI2MiwicGF0aCI6Ii8yNDE5NzQ2My81NTM1OTkzODEtZWQ5Y2FjYWEtZGYxNC00M2QyLWJhNzktYTcwMDE1ZDUxMmQ1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIzVDE2NDc0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWYyNTZlMGMwMzBlNDQ4NjgzNTU4ZWI0NWIzN2I5MzNjZjFmNzc0NjczMjUxM2VhZmMxZTEzYmM3Yzc4MDRmZjMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.a-VHJQsVq-phjUk-aCqZ7PRel-9OtxWlL_iV1Wy2GY4) <!-- pragma: allowlist secret --> | ![Alt text](https://private-user-images.githubusercontent.com/24197463/553599551-2e078a15-34e3-40d8-b854-776c3cdbce3c.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE4NjU1NjIsIm5iZiI6MTc3MTg2NTI2MiwicGF0aCI6Ii8yNDE5NzQ2My81NTM1OTk1NTEtMmUwNzhhMTUtMzRlMy00MGQ4LWI4NTQtNzc2YzNjZGJjZTNjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIzVDE2NDc0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWU5OTg0ZDdmNzA4M2YyYzg4NmJhYWU4YzM4ZjgyMmI0M2FmNWVmZWI4NTQyZWMxNmJjZGI2YTBlMTcxZjFkOTAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.gh719k2XUHSR1hXTCnysz0EmyOH2cg19jg0-zRvgwcs) <!-- pragma: allowlist secret --> | ![Alt text](https://private-user-images.githubusercontent.com/24197463/553599987-60c0079f-d5ed-423c-bc1d-8cd6b55ac76d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE4NjU1NjIsIm5iZiI6MTc3MTg2NTI2MiwicGF0aCI6Ii8yNDE5NzQ2My81NTM1OTk5ODctNjBjMDA3OWYtZDVlZC00MjNjLWJjMWQtOGNkNmI1NWFjNzZkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIzVDE2NDc0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTZhZDk3MjcyNzFiMjcxNzNhMTg2ZDNkZjQ0YjAyZTZmZTc2MTk5NzM2ZjM5ZTY5NTRlM2JmMzdkOTI1YjQyZjImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.JaP16Cbi9fTrcjNJfMm4z_aq7uUryKOtXPsbrp6JEuE) <!-- pragma: allowlist secret --> |
| | | |
