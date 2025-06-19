from characterization.scorer.score_utils import (
    simple_individual_score,
    simple_interaction_score,
)

SUPPORTED_SCORERS = ["individual", "interaction", "safeshift"]

INDIVIDUAL_SCORE_FUNCTIONS = {
    "simple": simple_individual_score,
}
INTERACTION_SCORE_FUNCTIONS = {
    "simple": simple_interaction_score,
}
