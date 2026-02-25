# ![WIP](https://img.shields.io/badge/status-WIP-orange) ScenarioCharacterization

>    **Note:** This project is a work in progress.

A generalizable, automated scenario characterization framework for trajectory datasets. This project is primarily a re-implementation of the scenario characterization approach introduced in [SafeShift](https://github.com/cmubig/SafeShift), as part of an internship project at **StackAV**.


| | | |
|---|---|---|
| **Categorical Scores** | **Animated Scenarios** | **Static Scenarios** |
| ![Alt text](https://private-user-images.githubusercontent.com/24197463/553599551-2e078a15-34e3-40d8-b854-776c3cdbce3c.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE4NjU1NjIsIm5iZiI6MTc3MTg2NTI2MiwicGF0aCI6Ii8yNDE5NzQ2My81NTM1OTk1NTEtMmUwNzhhMTUtMzRlMy00MGQ4LWI4NTQtNzc2YzNjZGJjZTNjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIzVDE2NDc0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWU5OTg0ZDdmNzA4M2YyYzg4NmJhYWU4YzM4ZjgyMmI0M2FmNWVmZWI4NTQyZWMxNmJjZGI2YTBlMTcxZjFkOTAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.gh719k2XUHSR1hXTCnysz0EmyOH2cg19jg0-zRvgwcs) <!-- pragma: allowlist secret --> | ![Alt text](https://private-user-images.githubusercontent.com/24197463/553598323-06e0ff2e-fdbf-4cff-a8b2-9eb235bcd634.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE4NjU5OTcsIm5iZiI6MTc3MTg2NTY5NywicGF0aCI6Ii8yNDE5NzQ2My81NTM1OTgzMjMtMDZlMGZmMmUtZmRiZi00Y2ZmLWE4YjItOWViMjM1YmNkNjM0LmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIzVDE2NTQ1N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTVlNzk4MzRlODQ2MGY3ZmY3OTdhMjU3MWEyOTBkMmQ5ZWRjMTk2Njg3NDA4ODYwNDRmNjc5MmFhMWM0MGQwYWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.zbN6DnZ3i1lIFcZsPfwjWdh0HAfgw0CWKAOP9T_YKRk) <!-- pragma: allowlist secret --> | ![Alt text](https://private-user-images.githubusercontent.com/24197463/553596187-06c0598f-3145-4b75-b2aa-a66cccde0638.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzE4NjU1NjIsIm5iZiI6MTc3MTg2NTI2MiwicGF0aCI6Ii8yNDE5NzQ2My81NTM1OTYxODctMDZjMDU5OGYtMzE0NS00Yjc1LWIyYWEtYTY2Y2NjZGUwNjM4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjIzVDE2NDc0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTg1NWFkOTlmMjUwYjg5N2Y3Y2Q0OTU5ZjhmMzI1MjIwZTZiN2NjNThiMWZmODkyMDA3NmYyZjBiNmUwZjY2NTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.ENVgW3JgG8l5hzWDodOsQJySWQ7s2NBcNY7z61zEik0) <!-- pragma: allowlist secret --> |
| Result from our categorical profiler. Agents are visualized from dark green (low crit.) to dark red (high crit.) based on their criticality w.r.t. the ego-agent (blue). | Result from our animated visualizer, Shows agents by color vehicle (gray), pedestrian (magenta), cyclist (green) and ego (blue), and the scenario's elapsed time throughout the episode. | Result from our static scenario visualizer. Episode's time is shown by increasing trajectory opacity over time.|

Repository: [github.com/navarrs/ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization)

This repository currently uses:
- [uv](https://docs.astral.sh/uv/) as the package manager.
- [Hydra](https://hydra.cc/docs/intro/) for hierarchical configuration management.
- [Pydantic](https://docs.pydantic.dev/latest/) for input/output data validation.

## Installation

### Install the package
```
uv pip install scenario-characterization
```

### Install the package in editable mode

Clone the repository and install the package in editable mode:
```bash
git clone git@github.com:navarrs/ScenarioCharacterization.git
cd ScenarioCharacterization
uv run pip install -e .
```

To install with Waymo dependencies (required for running the [example](#example)), use:
```bash
uv run pip install -e ".[waymo]"
```

If installing with dev, run
```bash
uv run pip install -e. ".[dev]"
uv run pre-commit install
```

## Documentation

- [Organization](./docs/ORGANIZATION.md): Overview of the Hydra configuration structure.
- [Schemas](./docs/SCHEMAS.md): Guidelines for creating dataset adapters and processors that comply with the required input/output schemas.
- [Characterization](./docs/CHARACTERIZATION.md): Details on supported scenario characterization and visualization tools, and how to use them.
- [Analysis](./docs/ANALYSIS.md): Shows how to run feature and score analyses.
- [Example](./docs/EXAMPLE.md): Step-by-step usage example using the [Waymo Open Motion Dataset](https://waymo.com/open).

## Citing

```
@INPROCEEDINGS{stoler2024safeshift,
  author={Stoler, Benjamin and Navarro, Ingrid and Jana, Meghdeep and Hwang, Soonmin and Francis, Jonathan and Oh, Jean},
  booktitle={2024 IEEE Intelligent Vehicles Symposium (IV)},
  title={SafeShift: Safety-Informed Distribution Shifts for Robust Trajectory Prediction in Autonomous Driving},
  year={2024},
  volume={},
  number={},
  pages={1179-1186},
  keywords={Meters;Collaboration;Predictive models;Robustness;Iron;Trajectory;Safety},
  doi={10.1109/IV55156.2024.10588828}}
```

## Development

Run `uv sync --frozen --all-groups` to set up environment.
Run `pre-commit run --all-files` to run all hooks on all files.
