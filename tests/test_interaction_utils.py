import unittest

import numpy as np

from characterization.features.interaction_utils import (
    compute_drac,
    compute_separation,
    compute_thw,
    compute_ttc,
)
from characterization.utils.common import InteractionAgent


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


if __name__ == "__main__":
    unittest.main()
