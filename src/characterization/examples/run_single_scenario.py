import pickle  # nosec B403
import warnings

from omegaconf import DictConfig

from characterization.features.individual_features import IndividualFeatures
from characterization.scorer.individual_scorer import IndividualScorer
from characterization.utils.datasets.waymo import WaymoData

warnings.filterwarnings("ignore", category=DeprecationWarning)

SCENARIO_FILE = "sample_1a6893c4577732ff.pkl"
INPUT_SCENARIO_FILEPATH = f"./samples/scenarios/{SCENARIO_FILE}"
print("Loading scenario from:", INPUT_SCENARIO_FILEPATH)
with open(INPUT_SCENARIO_FILEPATH, "rb") as f:
    scenario_data = pickle.load(f)  # nosec B301

dataset_config = DictConfig(
    {
        "load": False,
        "scenario_type": "gt",
        "scenario_base_path": None,
        "scenario_meta_path": None,
        "conflict_points_path": None,
    },
)
dataset = WaymoData(dataset_config)

# This will return the Scenario Schema object
scenario = dataset.transform_scenario_data(scenario_data)
print("\nTransformed scenario (fields):", scenario.model_fields.keys())

# Compute features
feature_config = DictConfig(
    {
        "return_criterion": "critical",  # Can be 'critical' or 'average'
    },
)
feature_processor = IndividualFeatures(feature_config)
features = feature_processor.compute(scenario)
print("\nComputed features (fields):\n", features.model_fields.keys())

# Compute the scenario scores
scorer_config = DictConfig(
    {
        "individual_score_function": "simple",
        "score_clip": {
            "min": 0.0,
            "max": 150.0,
        },
        "weights": {
            "speed": 0.1,
            "acceleration": 1.0,
            "deceleration": 1.0,
            "jerk": 1.0,
            "waiting_period": 1.0,
            "waiting_intervals": 1.0,
            "waiting_distances": 1.0,
        },
        "detections": {
            "speed": 10,
            "acceleration": 10,
            "deceleration": 10,
            "jerk": 10,
            "waiting_period": 8,
            "waiting_intervals": 8,
            "waiting_distances": 8,
        },
    },
)
scorer = IndividualScorer(scorer_config)
scores = scorer.compute(scenario=scenario, scenario_features=features)
print("\nComputed scenario score:\n", scores)
