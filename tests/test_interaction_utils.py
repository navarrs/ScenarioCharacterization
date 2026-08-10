import unittest

import numpy as np
from numpy.typing import NDArray

from characterization.features.interaction_utils import (
    compute_drac,
    compute_mttcp,
    compute_separation,
    compute_thw,
    compute_ttc,
)
from characterization.utils.common import InteractionAgent


def _dense_mttcp_reference(
    agent_i: InteractionAgent,
    agent_j: InteractionAgent,
    agent_to_agent_max_distance: float,
) -> NDArray[np.float32]:
    position_i, position_j = agent_i.position, agent_j.position
    vel_i, vel_j = agent_i.speed, agent_j.speed

    dists = np.linalg.norm(position_i[:, None, :] - position_j, axis=-1)
    i_idx, _ = np.where(dists <= agent_to_agent_max_distance)
    _, i_unique = np.unique(i_idx, return_index=True)
    ti = i_idx[i_unique]
    if len(ti) == 0:
        return np.array([np.inf], dtype=np.float32)

    conflict_points = position_i[ti]
    mttcp = np.inf * np.ones(conflict_points.shape[0], dtype=np.float32)
    cp_to_position_i = np.linalg.norm(position_i - conflict_points[:, None], axis=-1)
    cp_to_position_j = np.linalg.norm(position_j - conflict_points[:, None], axis=-1)
    tj = cp_to_position_j.argmin(axis=-1)

    t_min = np.minimum(ti, tj) + 1
    for n, t in enumerate(t_min):
        ttcp_i = cp_to_position_i[n, :t] / vel_i[:t]
        ttcp_j = cp_to_position_j[n, :t] / vel_j[:t]
        mttcp[n] = np.abs(ttcp_i - ttcp_j).min()

    return mttcp


class SeparationReuseTest(unittest.TestCase):
    """Test reuse of precomputed pair separation distances."""

    def test_metric_helpers_match_without_precomputed_separations(self) -> None:
        """Ensure precomputed separations preserve all metric-helper outputs."""
        agent_i = InteractionAgent()
        agent_j = InteractionAgent()
        agent_i.position = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        agent_j.position = np.asarray(
            [[2.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        agent_i.speed = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        agent_j.speed = np.asarray([2.0, 1.0, 2.0, 1.0], dtype=np.float32)
        agent_i.length = np.full(4, 1.0, dtype=np.float32)
        agent_j.length = np.full(4, 1.0, dtype=np.float32)

        valid_headings = np.asarray([0, 1, 3], dtype=np.intp)
        leading_agent = np.asarray([0, 1, 0], dtype=np.intp)
        separations = compute_separation(agent_i, agent_j)

        np.testing.assert_array_equal(
            compute_thw(agent_i, agent_j, leading_agent, valid_headings),
            compute_thw(agent_i, agent_j, leading_agent, valid_headings, separations=separations),
        )
        np.testing.assert_array_equal(
            compute_ttc(agent_i, agent_j, leading_agent, valid_headings),
            compute_ttc(agent_i, agent_j, leading_agent, valid_headings, separations=separations),
        )
        np.testing.assert_array_equal(
            compute_drac(agent_i, agent_j, leading_agent, valid_headings),
            compute_drac(agent_i, agent_j, leading_agent, valid_headings, separations=separations),
        )

    def test_mttcp_chunking_matches_dense_reference(self) -> None:
        """Ensure chunked mTTCP processing preserves the dense implementation for several chunk sizes."""
        agent_i = InteractionAgent()
        agent_j = InteractionAgent()
        agent_i.position = np.asarray(
            [[float(index), 0.0, 0.0] for index in range(7)],
            dtype=np.float32,
        )
        agent_j.position = np.asarray(
            [[6.0 - float(index), 0.25, 0.0] for index in range(7)],
            dtype=np.float32,
        )
        agent_i.speed = np.full(7, 2.0, dtype=np.float32)
        agent_j.speed = np.full(7, 1.5, dtype=np.float32)
        threshold = 0.5
        expected = _dense_mttcp_reference(agent_i, agent_j, threshold)

        for chunk_size in (1, 2, 256):
            np.testing.assert_array_equal(
                compute_mttcp(agent_i, agent_j, threshold, chunk_size=chunk_size),
                expected,
            )

    def test_mttcp_rejects_non_positive_chunk_size(self) -> None:
        """Ensure invalid mTTCP chunk sizes fail before allocating work buffers."""
        raised = False
        try:
            compute_mttcp(InteractionAgent(), InteractionAgent(), chunk_size=0)
        except ValueError:
            raised = True

        assert raised


if __name__ == "__main__":
    unittest.main()
