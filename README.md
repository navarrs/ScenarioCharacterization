<p align="center">
  <a href="https://github.com/astral-sh/uv">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" /></a>
  <a href="https://github.com/astral-sh/ruff">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
  <a href="https://github.com/astral-sh/ty">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" /></a>
  <a href="https://docs.pydantic.dev">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json" /></a>
  <a href="https://hydra.cc">
  <img src="https://img.shields.io/badge/config-Hydra-E87615" /></a>
</p>


# ![WIP](https://img.shields.io/badge/status-WIP-orange) ScenarioCharacterization

An open-source framework for automated, dataset-agnostic profiling of driving scenarios in trajectory datasets. Built upon the scenario characterization approach introduced in [SafeShift](https://github.com/cmubig/SafeShift), this project extends it into a modular, configuration-driven pipeline with three layers:

1. **Dataset adapter** — ingests custom datasets and re-formats them into a common scenario representation, validated by Pydantic schemas.
2. **Characterizer** — performs feature extraction, behavior probing, and criticality scoring at the scenario and agent levels.
3. **Analysis** — supports scenario visualization, feature and score analyses, categorical profiling, and scenario mining.

New datasets plug in without rewriting the characterization and analysis stack. The framework is demonstrated on [Waymo Open Motion](https://waymo.com/open), [nuScenes](https://www.nuscenes.org/), and [Argoverse2](https://www.argoverse.org/av2.html). Developed as part of an internship project at **StackAV**.

<img width="100%" alt="Scenario Characterization workflow diagram" src="https://github.com/user-attachments/assets/2639ed69-6b7e-407a-a510-ff064e39453e" /> <!-- pragma: allowlist secret -->

## Visualization Examples

| | | |
|:---:|:---:|:---:|
| **Categorical Scores** | **Animated Scenarios** | **Static Scenarios** |
| <img width="200" alt="5c1f8d26c481e36d_2 43" src="https://github.com/user-attachments/assets/2e078a15-34e3-40d8-b854-776c3cdbce3c" /> <!-- pragma: allowlist secret --> | <img width="200" alt="5c1f8d26c481e36d_2 43" src="https://github.com/user-attachments/assets/07688b7b-5252-4db7-9960-524761878dee" /> <!-- pragma: allowlist secret -->  | <img width="200" alt="6e593bf6b9dbbf73" src="https://github.com/user-attachments/assets/06c0598f-3145-4b75-b2aa-a66cccde0638" /> <!-- pragma: allowlist secret --> |
| Results from our categorical profiler. Agents are visualized from dark green (low crit.) to dark red (high crit.) based on their criticality with respect to the ego agent (blue). | Result from our animated visualizer, showing agents by type: vehicle (gray), pedestrian (magenta), cyclist (green), and ego (blue), along with the scenario's elapsed time throughout the episode. | Result from our static scenario visualizer. The episode's time is shown by increasing trajectory opacity over time. |

Repository: [github.com/navarrs/ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization)

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

To install with dataset-specific dependencies, use the appropriate optional extra:

```bash
# Waymo Open Motion Dataset (requires Python 3.10)
uv run pip install -e ".[waymo]"

# nuScenes dataset (requires Python 3.12)
uv run pip install -e ".[nuscenes]"
```

If installing with development dependencies, run:
```bash
uv run pip install -e ".[dev]"
uv run pre-commit install
```

## Documentation

- [Organization](./docs/ORGANIZATION.md): Overview of the Hydra configuration structure.
- [Schemas](./docs/SCHEMAS.md): Guidelines for creating dataset adapters and processors that comply with the required input/output schemas.
- [Characterization](./docs/CHARACTERIZATION.md): Details on supported scenario characterization and visualization tools, and how to use them.
- [Analysis](./docs/ANALYSIS.md): Shows how to run feature and score analyses.
- [Waymo Example](./docs/WAYMO_EXAMPLE.md): Step-by-step usage example using the [Waymo Open Motion Dataset](https://waymo.com/open).
- [nuScenes Example](./docs/NUSCENES_EXAMPLE.md): Step-by-step usage example using the [nuScenes dataset](https://www.nuscenes.org/).

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

Run `uv sync --frozen --all-groups` to set up the environment.
Run `pre-commit run --all-files` to run all hooks on all files.
