# Input/Output Schemas

Input and output schemas are defined in [`./characterization/utils/schemas.py`](./characterization/utils/schemas.py) using [Pydantic](https://docs.pydantic.dev/latest/).
This repository currently uses three main schemas:
- [Scenario](#scenario-schema)
- [ScenarioFeatures](#scenario-features-schema)
- [ScenarioScores](#scenario-scores-schema)

---

## Scenario Schema

The dataset adapter class is responsible for converting data from a dataset-specific format into a structured representation.

Schemas:
```python
class AgentData(BaseModel):
    agent_ids: list[NonNegativeInt]
    agent_types: list[AgentType]
    agent_trajectories: Float32NDArray3D
    agent_relevance: Float32NDArray2D | None = None


class ScenarioMetadata(BaseModel):
    scenario_id: str
    timestamps_seconds: list[float]
    current_time_index: int
    ego_vehicle_id: int
    ego_vehicle_index: int
    objects_of_interest: list[int]
    track_length: int
    dataset: str

    # Thresholds
    stationary_speed: float = 0.25  # m/s
    agent_to_agent_max_distance: float = 50.0  # meters
    agent_to_conflict_point_max_distance: float = 2.0  # meters
    agent_to_agent_distance_breach: float = 1.0  # meters
    heading_threshold: float = 45.0  # degrees

class TracksToPredict(BaseModel):
    track_index: list[NonNegativeInt]
    difficulty: list[NonNegativeInt]
    object_type: list[AgentType]


class StaticMapData(BaseModel):
    map_polylines: Float32NDArray2D | None = None
    lane_ids: Int32NDArray1D | None = None
    lane_speed_limits_mph: Float32NDArray1D | None = None
    lane_polyline_idxs: Int32NDArray2D | None = None
    road_line_ids: Int32NDArray1D | None = None
    road_line_polyline_idxs: Int32NDArray2D | None = None
    road_edge_ids: Int32NDArray1D | None = None
    road_edge_polyline_idxs: Int32NDArray2D | None = None
    crosswalk_ids: Int32NDArray1D | None = None
    crosswalk_polyline_idxs: Int32NDArray2D | None = None
    speed_bump_ids: Int32NDArray1D | None = None
    speed_bump_polyline_idxs: Int32NDArray2D | None = None
    stop_sign_ids: Int32NDArray1D | None = None
    stop_sign_polyline_idxs: Int32NDArray2D | None = None
    stop_sign_lane_ids: list[list[int]] | None = None

    # Optional information that can be derived from existing map information
    map_conflict_points: Float32NDArray2D | None = None
    agent_distances_to_conflict_points: Float32NDArray3D | None = None

class DynamicMapData(BaseModel):
    stop_points: list[Any] | None = None
    lane_ids: list[Any] | None = None
    states: list[Any] | None = None  # Placeholder for state information, can be more specific if needed


class Scenario(BaseModel):
    metadata: ScenarioMetadata
    agent_data: AgentData
    tracks_to_predict: TracksToPredict | None = None
    static_map_data: StaticMapData | None = None
    dynamic_map_data: DynamicMapData | None = None
```

See [[SOURCE](../src/characterization/schemas/scenario.py)] for more details and descriptions.

---

## Scenario Features Schema

The feature processor takes a `Scenario` as input and produces `ScenarioFeatures`.

Schemas:
```python
class Individual(BaseModel):
    # Agent meta
    valid_idxs: Int32NDArray1D | None = None
    agent_types: list[AgentType] | None = None

    # Individual Features
    speed: Float32NDArray1D | None = None
    speed_limit_diff: Float32NDArray1D | None = None
    acceleration: Float32NDArray1D | None = None
    deceleration: Float32NDArray1D | None = None
    jerk: Float32NDArray1D | None = None
    waiting_period: Float32NDArray1D | None = None


class Interaction(BaseModel):
    # Interaction Features
    separation: Float32NDArray1D | None = None
    intersection: Float32NDArray1D | None = None
    collision: Float32NDArray1D | None = None
    mttcp: Float32NDArray1D | None = None
    thw: Float32NDArray1D | None = None
    ttc: Float32NDArray1D | None = None
    drac: Float32NDArray1D | None = None

    interaction_status: list[InteractionStatus] | None = None
    interaction_agent_indices: list[tuple[int, int]] | None = None
    interaction_agent_types: list[tuple[AgentType, AgentType]] | None = None


class ScenarioFeatures(BaseModel):
    metadata: ScenarioMetadata
    individual_features: Individual | None = None
    interaction_features: Interaction | None = None
    agent_to_agent_closest_dists: Float32NDArray2D | None = None

```

See [[SOURCE](../src/characterization/schemas/scenario_features.py)] for more details and descriptions.

---

## Scenario Scores Schema

The score processor takes a `Scenario` and its corresponding `ScenarioFeatures` as input, and produces `ScenarioScores`.

Schemas:
```python
class Score(BaseModel):
    agent_scores: Float32NDArray1D | None = None
    scene_score: float | None = None

class ScenarioScores(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    metadata: ScenarioMetadata
    individual_scores: Score | None = None
    interaction_scores: Score | None = None
    safeshift_scores: Score | None = None
```

See [[SOURCE](../src/characterization/schemas/scenario_scores.py)] for more details and descriptions.
