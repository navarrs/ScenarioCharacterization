# ![WIP](https://img.shields.io/badge/status-WIP-orange) ScenarioCharacterization

>    **Note:** This project is a work in progress.

A generalizable, automated scenario characterization framework for trajectory datasets. This project is primarily a re-implementation of the scenario characterization approach introduced in [SafeShift](https://github.com/cmubig/SafeShift), as part of an internship project at **StackAV**.

## Visualization Examples

### Animated Scenarios

| | |
|:---:|:---:|
| <div align="center"><img width="500" alt="5c1f8d26c481e36d_2 43" src="https://github.com/user-attachments/assets/07688b7b-5252-4db7-9960-524761878dee" /> </div> <!-- pragma: allowlist secret --> | Result from our animated visualizer, Shows agents by color vehicle (gray), pedestrian (magenta), cyclist (green) and ego (blue), and the scenario's elapsed time throughout the episode.  |

### Static Scenarios

| | |
|:---:|:---:|
| <div align="center"><img width="700" alt="6e593bf6b9dbbf73" src="https://github.com/user-attachments/assets/06c0598f-3145-4b75-b2aa-a66cccde0638" /></div> <!-- pragma: allowlist secret --> | Result from our static scenario visualizer. Episode's time is shown by increasing trajectory opacity over time.  |

### Scenarios by Categorical Scores

| | |
|:---:|:---:|
| <div align="center"><img width="900" alt="5c1f8d26c481e36d_2 43" src="https://github.com/user-attachments/assets/2e078a15-34e3-40d8-b854-776c3cdbce3c" /> </div><!-- pragma: allowlist secret --> |  Result from our categorical profiler. Agents are visualized from dark green (low crit.) to dark red (high crit.) based on their criticality w.r.t. the ego-agent (blue). |

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
