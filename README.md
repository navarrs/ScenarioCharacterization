# ScenarioCharacterization (Work-in-Progress)

This repository is for automated scenario characterization for trajectory datasets. Currently, 
it builds from the scenario characterization approach introduced in [SafeShift](https://github.com/cmubig/SafeShift).

## Pre-requisites

This repository is using: 
- [uv](https://docs.astral.sh/uv/) as package manager.
- [Hydra](https://hydra.cc/docs/intro/) for hierarchical configurations. 
- **WIP: Pydantic**

## Repository Organization 

The main configuration file is `run_processor.yaml` which is utilized for computing scenario features
and scores. 

It is hierarchically constructed through the following configs:
- `characterizer`: specifies what type of characterization to run. For example: features, scores, etc.
- `dataset`: specifies which dataset adapter to use. 
- `paths`: specifies paths to the input and output data.
- `processor`: specifies what type of processor to run. Currently, we support feature and score processors. 

## Dataset Adapters

The dataset adapter class is intented for converting data from a dataset-specific format into a  structured representation. 

**WIP: Describe how it works.**

## Running a scenario processor 

The processor classes are designed to take a set of input scenarios and produce a specified characterization. 

### Feature processor

The feature processor takes in a feature class specified in the `characterizer` configuration and produces specialized features for a set of input scenarios specified in the `paths` configuration file. 

```
uv run src/run_processor.py processor=features characterizer=[feature_type]
```

Currently, we provide these feature groups, located under `config/characterizer`:
- `feature`: which computes a dummy random feature. Only used for testing purposes. 
- `individual_feature`: which computes a set of individual agent descriptors.
- **WIP: `interactive_feature`: which computes a set of interactive agent descriptors:**

#### Individual features

To run this characterizer: 
```
uv run src/run_processor.py processor=features characterizer=individual_features
```

Currently supported features:
- Agent speed
- **WIP: Agent speed limit diff**: Difference between the agent's speed and the speed limit
- Agent acceleration 
- Agent jerk
- Agent waiting period: interval an agent is waiting near a conflict point. 
- **WIP: Agent In-Lane**: deviation from a lane
- **WIP: Trajectory Anomaly**: distance to the closest behavior primitive.

#### WIP: Interactive Features

**WIP** To run this characterizer: 
```
uv run src/run_processor.py processor=features characterizer=interactive_features
```

Currently supported features:
 - **WIP: Time Headway**
 - **WIP: Time to Collision**
 - **WIP: Minimum Time to Conflict Point**
 - **WIP: Collisions**
 - **WIP: Trajectory-Pair Anomaly**


## TBD: Score processor


**WIP** The scorer takes in a list of features for a set of input scenarios and produces a specialized score. 
```
uv run src/run_processor.py processor=scores characterizer=score
```

## TBD: Scenario Probing

**WIP**

# TO-DOs

* Add probing
* Add score functions
* Add flexibility for specifying metrics of interest 