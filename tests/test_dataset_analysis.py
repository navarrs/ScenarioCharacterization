"""Tests for per-dataset trajectory and interaction counting."""

import numpy as np

from characterization.schemas.scenario import ScenarioMetadata
from characterization.schemas.scenario_features import Individual, Interaction, ScenarioFeatures
from characterization.utils.analysis import count_scenario_features
from characterization.utils.common import InteractionStatus, TrajectoryType
from characterization.utils.scenario_types import AgentType

# Agents 0..3: ego, vehicle, pedestrian, cyclist. Agent 2 (pedestrian) fails the validity gate.
AGENT_TYPES = [
    AgentType.TYPE_EGO_AGENT,
    AgentType.TYPE_VEHICLE,
    AgentType.TYPE_PEDESTRIAN,
    AgentType.TYPE_CYCLIST,
]
VALID_IDXS = np.asarray([0, 1, 3], dtype=np.int32)
AGENT_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PAIR_STATUSES = [
    InteractionStatus.COMPUTED_OK,
    InteractionStatus.PARTIAL_INVALID_HEADING,
    InteractionStatus.DISTANCE_TOO_FAR,
    InteractionStatus.MASK_NOT_VALID,
    InteractionStatus.COMPUTED_OK,
    InteractionStatus.COMPUTED_OK,
]


def _metadata() -> ScenarioMetadata:
    return ScenarioMetadata(
        scenario_id="counts-test",
        timestamps_seconds=[0.0, 1.0],
        frequency_hz=1.0,
        current_time_index=0,
        ego_vehicle_id=0,
        ego_vehicle_index=0,
        objects_of_interest=[],
        track_length=2,
        dataset="test",
    )


def _features(*, with_individual: bool = True, with_interaction: bool = True) -> ScenarioFeatures:
    individual = None
    if with_individual:
        individual = Individual(
            valid_idxs=VALID_IDXS,
            agent_types=AGENT_TYPES,
            agent_trajectory_types=[TrajectoryType.TYPE_STRAIGHT] * len(VALID_IDXS),
        )

    interaction = None
    if with_interaction:
        interaction = Interaction(
            interaction_agent_indices=AGENT_PAIRS,
            interaction_agent_types=[(AGENT_TYPES[i], AGENT_TYPES[j]) for i, j in AGENT_PAIRS],
            interaction_status=PAIR_STATUSES,
        )

    return ScenarioFeatures(metadata=_metadata(), individual_features=individual, interaction_features=interaction)


def _nonzero(counts: dict[str, int], prefix: str) -> dict[str, int]:
    """Returns the non-zero counts under *prefix*, with the prefix stripped from each key."""
    return {key.removeprefix(prefix): value for key, value in counts.items() if key.startswith(prefix) and value}


def test_agents_split_by_type_with_ego_separate() -> None:
    """Every agent is counted under its own type, and the ego keeps its own bucket."""
    counts = count_scenario_features(_features())

    assert _nonzero(counts, "agents_total_") == {
        "TYPE_EGO_AGENT": 1,
        "TYPE_VEHICLE": 1,
        "TYPE_PEDESTRIAN": 1,
        "TYPE_CYCLIST": 1,
    }


def test_valid_agents_follow_valid_idxs() -> None:
    """Valid counts subset the full agent list through valid_idxs, dropping agent 2."""
    counts = count_scenario_features(_features())

    assert _nonzero(counts, "agents_valid_") == {
        "TYPE_EGO_AGENT": 1,
        "TYPE_VEHICLE": 1,
        "TYPE_CYCLIST": 1,
    }


def test_pairs_bucket_by_pair_type_with_ego_folded_into_vehicle() -> None:
    """Total pair counts cover every enumerated pair, with ego pairs typed as vehicle pairs."""
    counts = count_scenario_features(_features())

    assert _nonzero(counts, "pairs_total_") == {
        "TYPE_VEHICLE_VEHICLE": 1,  # (ego, vehicle)
        "TYPE_VEHICLE_PEDESTRIAN": 2,  # (ego, ped), (vehicle, ped)
        "TYPE_VEHICLE_CYCLIST": 2,  # (ego, cyclist), (vehicle, cyclist)
        "TYPE_PEDESTRIAN_CYCLIST": 1,  # (ped, cyclist)
    }


def test_only_computed_statuses_count_as_valid_pairs() -> None:
    """COMPUTED_OK and PARTIAL_INVALID_HEADING count; TOO_FAR and MASK_NOT_VALID do not."""
    counts = count_scenario_features(_features())

    assert _nonzero(counts, "pairs_valid_") == {
        "TYPE_VEHICLE_VEHICLE": 1,  # (ego, vehicle) OK
        "TYPE_VEHICLE_PEDESTRIAN": 1,  # (ego, ped) PARTIAL; (vehicle, ped) MASK_NOT_VALID
        "TYPE_VEHICLE_CYCLIST": 1,  # (ego, cyclist) TOO_FAR; (vehicle, cyclist) OK
        "TYPE_PEDESTRIAN_CYCLIST": 1,  # (ped, cyclist) OK
    }


def test_ego_pairs_tracked_separately() -> None:
    """Ego involvement is reported alongside the pair-type buckets that hide it."""
    counts = count_scenario_features(_features())

    assert _nonzero(counts, "pairs_with_ego_") == {"total": 3, "valid": 2}


def test_missing_halves_count_as_zero() -> None:
    """A feature artifact holding only one half still counts, with zeros for the other."""
    individual_only = count_scenario_features(_features(with_interaction=False))
    assert _nonzero(individual_only, "agents_total_") == {
        "TYPE_EGO_AGENT": 1,
        "TYPE_VEHICLE": 1,
        "TYPE_PEDESTRIAN": 1,
        "TYPE_CYCLIST": 1,
    }
    assert _nonzero(individual_only, "pairs_") == {}

    interaction_only = count_scenario_features(_features(with_individual=False))
    assert _nonzero(interaction_only, "pairs_total_") == {
        "TYPE_VEHICLE_VEHICLE": 1,
        "TYPE_VEHICLE_PEDESTRIAN": 2,
        "TYPE_VEHICLE_CYCLIST": 2,
        "TYPE_PEDESTRIAN_CYCLIST": 1,
    }
    assert _nonzero(interaction_only, "agents_") == {}
