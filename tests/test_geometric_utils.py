"""Tests for geometric utility functions."""

import unittest

import numpy as np

from characterization.utils.geometric_utils import (
    compute_agent_to_agent_closest_dists,
    compute_agent_to_ego_closest_dists,
)


class ClosestDistanceTest(unittest.TestCase):
    """Tests for chunked closest-distance computation."""

    def test_chunking_preserves_dense_result(self) -> None:
        """Chunked computation matches the previous dense result for multiple chunk sizes."""
        positions = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0]],
                [[10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 2.0, 0.0]],
            ],
            dtype=np.float32,
        )
        dense_distances = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
        expected = np.nan_to_num(np.nanmin(dense_distances, axis=-1), nan=np.inf).astype(np.float32)

        actual_default = compute_agent_to_agent_closest_dists(positions)
        np.testing.assert_array_equal(actual_default, expected)

        for chunk_size in (1, 2, 256):
            actual = compute_agent_to_agent_closest_dists(positions, chunk_size=chunk_size)
            np.testing.assert_array_equal(actual, expected)

    def test_non_positive_chunk_size_is_rejected(self) -> None:
        """Invalid chunk sizes fail explicitly."""
        positions = np.zeros((2, 2, 3), dtype=np.float32)

        with self.assertRaises(ValueError):  # noqa: PT027
            compute_agent_to_agent_closest_dists(positions, chunk_size=0)

    def test_ego_distances_match_full_matrix_column(self) -> None:
        """Compact ego distances match the corresponding full-matrix values."""
        positions = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0]],
                [[10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 2.0, 0.0]],
            ],
            dtype=np.float32,
        )
        full_distances = compute_agent_to_agent_closest_dists(positions, chunk_size=1)
        ego_distances = compute_agent_to_ego_closest_dists(positions, ego_agent_index=1)

        np.testing.assert_array_equal(ego_distances, full_distances[:, 1])
