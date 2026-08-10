import unittest

import numpy as np
from numpy.typing import NDArray
from shapely import LineString

from characterization.features.interaction_utils import compute_intersections
from characterization.utils.common import InteractionAgent


def _reference_intersections(positions_i: NDArray[np.float32], positions_j: NDArray[np.float32]) -> NDArray[np.bool_]:
    segments_i = np.stack([positions_i[:-1], positions_i[1:]], axis=1)
    segments_j = np.stack([positions_j[:-1], positions_j[1:]], axis=1)
    intersections = [
        LineString(segment_i).intersects(LineString(segment_j))
        for segment_i, segment_j in zip(segments_i, segments_j, strict=False)
    ]
    return np.asarray([intersections[0], *intersections], dtype=bool)


class IntersectionCullingTest(unittest.TestCase):
    """Test that AABB rejection preserves Shapely intersection results."""

    def test_intersections_match_reference_geometry(self) -> None:
        """Ensure culling does not change crossing, touching, or disjoint segment results."""
        cases = (
            (
                np.asarray([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32),
                np.asarray([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [4.0, 2.0, 0.0]], dtype=np.float32),
            ),
            (
                np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
                np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
            ),
            (
                np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
                np.asarray([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float32),
            ),
        )

        for positions_i, positions_j in cases:
            agent_i = InteractionAgent()
            agent_j = InteractionAgent()
            agent_i.position = positions_i
            agent_j.position = positions_j

            np.testing.assert_array_equal(
                compute_intersections(agent_i, agent_j),
                _reference_intersections(positions_i, positions_j),
            )


if __name__ == "__main__":
    unittest.main()
