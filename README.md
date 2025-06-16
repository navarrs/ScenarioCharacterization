# ![WIP](https://img.shields.io/badge/status-WIP-orange) ScenarioCharacterization

A generalizable, automated scenario characterization framework for trajectory datasets.
Currently, this is a re-implementation of the scenario characterization approach introduced in [SafeShift](https://github.com/cmubig/SafeShift).

Repository: [github.com/navarrs/ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization)

This repository currently uses:
- [uv](https://docs.astral.sh/uv/) as the package manager.
- [Hydra](https://hydra.cc/docs/intro/) for hierarchical configuration management.
- [Pydantic](https://docs.pydantic.dev/latest/) for input/output data validation.

## Installation

> **Note:** This project is a work in progress.

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

## Documentation

- [Organization](./docs/ORGANIZATION.md): Overview of the Hydra configuration structure.
- [Schemas](./docs/SCHEMAS.md): Guidelines for creating dataset adapters and processors that comply with the required input/output schemas.
- [Characterization](./docs/CHARACTERIZATION.md): Details on supported scenario characterization and visualization tools, and how to use them.
- [Example](./docs/EXAMPLE.md): Step-by-step usage example using the [Waymo Open Motion Dataset](https://waymo.com/open).
