"""Tests for compact ego-distance weighting."""

import unittest

import numpy as np
from omegaconf import DictConfig

from characterization.schemas.scenario import AgentData, Scenario, ScenarioMetadata
from characterization.schemas.scenario_features import Interaction, ScenarioFeatures
from characterization.scorer.interaction_scorer import InteractionScorer
from characterization.utils.common import InteractionStatus
from characterization.utils.geometric_utils import (
    compute_agent_to_agent_closest_dists,
    compute_agent_to_ego_closest_dists,
)
from characterization.utils.scenario_types import AgentType


class EgoDistanceWeightingTest(unittest.TestCase):
    """Tests that compact ego distances preserve scoring weights."""

    def test_compact_distances_preserve_ego_weights(self) -> None:
        """Full and compact distance features produce identical ego weights."""
        positions = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0]],
                [[10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 2.0, 0.0]],
            ],
            dtype=np.float32,
        )
        trajectory = np.zeros((3, 3, 10), dtype=np.float32)
        trajectory[:, :, :3] = positions
        trajectory[:, :, 9] = 1.0
        metadata = ScenarioMetadata(
            scenario_id="ego-weight-test",
            timestamps_seconds=[0.0, 1.0, 2.0],
            frequency_hz=1.0,
            current_time_index=0,
            ego_vehicle_id=1,
            ego_vehicle_index=1,
            objects_of_interest=[],
            track_length=3,
            dataset="test",
        )
        scenario = Scenario(
            metadata=metadata,
            agent_data=AgentData(
                agent_ids=[0, 1, 2],
                agent_types=[AgentType.TYPE_VEHICLE] * 3,
                agent_trajectories=trajectory,
            ),
        )
        full_features = ScenarioFeatures(
            metadata=metadata,
            agent_to_agent_closest_dists=compute_agent_to_agent_closest_dists(positions, chunk_size=1),
        )
        compact_features = ScenarioFeatures(
            metadata=metadata,
            agent_to_ego_closest_dists=compute_agent_to_ego_closest_dists(positions, ego_agent_index=1),
        )

        scorer = InteractionScorer(
            DictConfig(
                {
                    "interaction_score_function": "simple",
                    "score_weighting_method": "distance_to_ego_agent",
                    "score_clip": {"min": 0.0, "max": 200.0},
                }
            )
        )
        full_weights = scorer.get_weights(scenario, full_features)
        compact_weights = scorer.get_weights(scenario, compact_features)

        np.testing.assert_array_equal(compact_weights, full_weights)

        interaction = Interaction(
            collision=np.asarray([1.0, 0.0], dtype=np.float32),
            inv_mttcp=np.asarray([1.0, 0.2], dtype=np.float32),
            inv_thw=np.asarray([0.4, 0.1], dtype=np.float32),
            inv_ttc=np.asarray([0.3, 0.05], dtype=np.float32),
            drac=np.asarray([2.0, 0.5], dtype=np.float32),
            interaction_status=[InteractionStatus.COMPUTED_OK, InteractionStatus.COMPUTED_OK],
            interaction_agent_indices=[(0, 1), (1, 2)],
        )
        full_score = scorer.compute_interaction_score(
            scenario,
            ScenarioFeatures(
                metadata=metadata,
                interaction_features=interaction,
                agent_to_agent_closest_dists=full_features.agent_to_agent_closest_dists,
            ),
        )
        compact_score = scorer.compute_interaction_score(
            scenario,
            ScenarioFeatures(
                metadata=metadata,
                interaction_features=interaction,
                agent_to_ego_closest_dists=compact_features.agent_to_ego_closest_dists,
            ),
        )

        assert compact_score.agent_scores is not None
        assert full_score.agent_scores is not None
        np.testing.assert_array_equal(compact_score.agent_scores, full_score.agent_scores)
        assert compact_score.scene_score == full_score.scene_score
