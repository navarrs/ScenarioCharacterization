# Input/Output Schemas

Input and output schemas are defined in [`./characterization/utils/schemas.py`](./characterization/utils/schemas.py) using [Pydantic](https://docs.pydantic.dev/latest/).
This repository currently uses the following schemas:
- [Scenario](#scenario-schema)
- [ScenarioFeatures](#scenario-features-schema)
- [ScenarioScores](#scenario-scores-schema)
- [FeatureDetections](#feature-detections-schema)
- [FeatureWeights](#feature-weights-schema)

## Scenario Schema

The dataset adapter class is responsible for converting data from a dataset-specific format into a structured representation.

The `Scenario` schema encapsulates:
* Agent information (e.g., trajectories, agent types, etc)
* Scenario metatada (e.g., scenario ID, scenario length, ego agent index, etc)
* Static map information (e.g., road layout, conflict points, etc)
* Dynamic map information (e.g., traffic lights and their states, etc)
* Tracks to predict, if downstream task is trajectory prediction

This model assumes that at least scenario metadata and agent information will be provided. All other fields are optional, in case information is unavailable.

See the [[schema](../src/characterization/schemas/scenario.py)] for more details and descriptions.

---

## Scenario Features Schema

The feature processor takes a `Scenario` as input and produces `ScenarioFeatures`.

The `ScenarioFeatures` schema encapsulates:
* **Metadata**: General information about the scenario, such as scenario ID and other relevant attributes. This should match the metadata from the scenario schema.
* **Individual Features**: Per-agent features, including:
    - Agent meta (e.g., valid indices, agent types)
    - Kinematic features (e.g., speed, acceleration, deceleration, jerk)
    - Behavioral features (e.g., waiting period, speed limit difference)
* **Interaction Features**: Features describing interactions between agents, such as:
    - Separation, intersection, and collision metrics
    - Time-based metrics (e.g., minimum time to collision point, time headway, time to collision, deceleration rate to avoid collision)
    - Interaction status and involved agent indices/types

See the [[schema](../src/characterization/schemas/scenario_features.py)] for more details and descriptions.

---

## Scenario Scores Schema

The score processor takes a `Scenario` and its corresponding `ScenarioFeatures` as input, and produces `ScenarioScores`.

The `ScenarioScores` schema encapsulates:

* **Metadata**: General information about the scenario, such as scenario ID and other relevant attributes. This matches the metadata from the scenario and features schemas.
* **Individual Scores**: Per-agent scores, which may include:
    - An array of scores for each agent in the scenario
    - An overall scene-level score summarizing agent performance
* **Interaction Scores**: Scores that quantify interactions between agents, such as:
    - Metrics for safety, efficiency, or other interaction-based criteria
    - Scene-level and per-agent interaction scores
* **SafeShift Scores**: Per-agent scores combining individual and interaction features.

See the [[schema](../src/characterization/schemas/scenario_scores.py)] for more details and descriptions.


## Feature Detections Schema

The `FeatureDetections` schema specifies configurable thresholds for detecting key driving behaviors or events from scenario features. It covers both individual and interaction features, referencing their sources and default values based on empirical or domain knowledge. These thresholds guide downstream processors in flagging notable events and can be adjusted for different datasets or applications.

See the [[schema](../src/characterization/schemas/detections.py)] for details.

## Feature Weights Schema

The `FeatureWeights` schema defines the relative importance (weights) assigned to each feature when aggregating or scoring scenario features. Each weight corresponds to a specific feature (e.g., speed, acceleration, collision) and can be tuned to emphasize or de-emphasize certain aspects of agent or interaction behavior in downstream metrics or composite scores. Default values are provided, but these can be customized to suit different evaluation criteria or application needs.

See the [[schema](../src/characterization/schemas/detections.py)] for details.
