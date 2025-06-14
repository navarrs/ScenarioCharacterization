# Input/Output Schemas

Input and output schemas are defined in [`./characterization/utils/schemas.py`](./characterization/utils/schemas.py) using [Pydantic](https://docs.pydantic.dev/latest/).
This repository currently uses three main schemas:
- [Scenario](#scenario-schema)
- [ScenarioFeatures](#scenario-features-schema)
- [ScenarioScores](#scenario-scores-schema)

---

## Scenario Schema

The dataset adapter class is responsible for converting data from a dataset-specific format into a structured representation, as defined by the schema below:

```python
class Scenario(BaseModel):
    # Scenario Information
    scenario_id: str
    last_observed_timestep: PositiveInt
    total_timesteps: PositiveInt
    timestamps: Float32NDArray1D

    # Agent Information
    num_agents: PositiveInt
    ego_index: NonNegativeInt
    ego_id: PositiveInt
    agent_ids: List[NonNegativeInt]
    agent_types: List[str]
    agent_valid: BooleanNDArray3D
    agent_positions: Float32NDArray3D
    agent_dimensions: Float32NDArray3D
    agent_velocities: Float32NDArray3D
    agent_headings: Float32NDArray3D
    agent_relevance: Float32NDArray1D

    # Map Information
    map_conflict_points: Float32NDArray2D | None
    agent_distances_to_conflict_points: Float32NDArray3D | None

    # Thresholds
    stationary_speed: float
    agent_to_agent_max_distance: float
    agent_to_conflict_point_max_distance: float
    agent_to_agent_distance_breach: float

    model_config = {"arbitrary_types_allowed": True}
```

---

## Scenario Features Schema

The feature processor takes a `Scenario` as input and produces `ScenarioFeatures` as output:

```python
class ScenarioFeatures(BaseModel):
    scenario_id: str
    num_agents: PositiveInt

    # Individual Features
    valid_idxs: Int32NDArray1D | None = None
    agent_types: List[str] | None = None
    speed: Float32NDArray1D | None = None
    speed_limit_diff: Float32NDArray1D | None = None
    acceleration: Float32NDArray1D | None = None
    deceleration: Float32NDArray1D | None = None
    jerk: Float32NDArray1D | None = None
    waiting_period: Float32NDArray1D | None = None
    waiting_interval: Float32NDArray1D | None = None
    waiting_distance: Float32NDArray1D | None = None

    # Interaction Features
    agent_to_agent_closest_dists: Float32NDArray2D | None = None
    separation: Float32NDArray1D | None = None
    intersection: Float32NDArray1D | None = None
    collision: Float32NDArray1D | None = None
    mttcp: Float32NDArray1D | None = None
    interaction_status: List[InteractionStatus] | None = None
    interaction_agent_indices: List[tuple[int, int]] | None = None
    interaction_agent_types: List[tuple[str, str]] | None = None

    model_config = {"arbitrary_types_allowed": True}
```

---

## Scenario Scores Schema

The score processor takes a `Scenario` and its corresponding `ScenarioFeatures` as input, and produces `ScenarioScores` as output:

```python
class ScenarioScores(BaseModel):
    scenario_id: str
    num_agents: PositiveInt

    # Individual Scores
    individual_agent_scores: Float32NDArray1D | None = None
    individual_scene_score: float | None = None

    # Interaction Scores
    interaction_agent_scores: Float32NDArray1D | None = None
    interaction_scene_score: float | None = None

    # Combined Scores
    combined_agent_scores: Float32NDArray1D | None = None
    combined_scene_score: float | None = None

    model_config = {"arbitrary_types_allowed": True}
```
