"""Unit tests for pipeline.data.utils — collect_arrays."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.data.utils import collect_arrays
from pipeline.protocols import Batch


class FakeStream:
    """Minimal DataStream yielding two batches."""

    def __init__(self):
        self._batches = [
            Batch(inputs=np.array([[1.0, 2.0], [3.0, 4.0]]), targets=np.array([0, 1])),
            Batch(inputs=np.array([[5.0, 6.0]]), targets=np.array([0])),
        ]

    def __iter__(self):
        yield from self._batches

    def __len__(self):
        return len(self._batches)


class TestCollectArrays:
    def test_collects_all_batches(self):
        """Happy Path: drains entire stream into (X, y)."""
        stream = FakeStream()
        X, y = collect_arrays(stream)
        assert X.shape == (3, 2)
        assert y.shape == (3,)
        np.testing.assert_array_equal(X, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        np.testing.assert_array_equal(y, np.array([0, 1, 0]))

    def test_single_batch(self):
        """Boundary: stream with exactly one batch."""
        stream = FakeStream()
        stream._batches = [Batch(inputs=np.array([[1.0]]), targets=np.array([0]))]
        X, y = collect_arrays(stream)
        assert X.shape == (1, 1)
        assert y.shape == (1,)

    def test_empty_stream_raises(self):
        """Boundary: empty stream should raise ValueError."""
        stream = FakeStream()
        stream._batches = []

        with pytest.raises(ValueError, match="empty"):
            collect_arrays(stream)
