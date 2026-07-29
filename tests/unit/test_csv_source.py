"""Unit tests for pipeline.data.csv_source -- CsvDataSource."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pipeline.data.csv_source import CsvDataSource
from pipeline.protocols import Batch


@pytest.fixture
def tiny_csv() -> str:
    """Write a temporary CSV for testing."""
    import pandas as pd

    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [0.1, 0.2, 0.3, 0.4, 0.5],
        "label": [0, 1, 0, 1, 0],
    })
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        df.to_csv(f, index=False)
        return f.name


class TestCsvDataSourceBasic:
    """Basic functionality tests."""

    def test_iter_yields_batches(self, tiny_csv):
        """Happy Path: __iter__ yields Batch objects."""
        source = CsvDataSource(tiny_csv, batch_size=2)
        batches = list(source)
        assert len(batches) == 3  # 5 samples / 2 = 3 batches
        assert all(isinstance(b, Batch) for b in batches)

    def test_batch_shapes(self, tiny_csv):
        """Happy Path: batch inputs/targets have correct shapes."""
        source = CsvDataSource(tiny_csv, batch_size=2, shuffle=False)
        batches = list(source)
        # First 2 batches: 2 samples each; last batch: 1 sample
        assert batches[0].inputs.shape == (2, 2)  # 2 features (a, b)
        assert batches[0].targets.shape == (2,)
        assert batches[1].inputs.shape == (2, 2)
        assert batches[1].targets.shape == (2,)
        assert batches[2].inputs.shape == (1, 2)
        assert batches[2].targets.shape == (1,)

    def test_len_returns_num_batches(self, tiny_csv):
        """Happy Path: __len__ returns ceil(n_samples / batch_size)."""
        source = CsvDataSource(tiny_csv, batch_size=2)
        assert len(source) == 3

    def test_target_column_default_last(self, tiny_csv):
        """Happy Path: by default, last column is treated as target."""
        source = CsvDataSource(tiny_csv, batch_size=5, shuffle=False)
        batch = next(iter(source))
        # targets should be the 'label' column
        np.testing.assert_array_equal(batch.targets, np.array([0, 1, 0, 1, 0]))

    def test_explicit_target_column(self, tiny_csv):
        """Happy Path: explicit target_column uses that column as target."""
        source = CsvDataSource(
            tiny_csv, batch_size=5, target_column="label", shuffle=False
        )
        batch = next(iter(source))
        # inputs should have only 'a' and 'b' columns
        assert batch.inputs.shape == (5, 2)


class TestCsvDataSourceShuffle:
    """Reproducibility and shuffle tests."""

    def test_same_seed_same_order(self, tiny_csv):
        """Reproducibility: same seed yields same batch order."""
        s1 = CsvDataSource(tiny_csv, batch_size=2, seed=42)
        s2 = CsvDataSource(tiny_csv, batch_size=2, seed=42)
        batches1 = list(s1)
        batches2 = list(s2)
        for b1, b2 in zip(batches1, batches2):
            np.testing.assert_array_equal(b1.inputs, b2.inputs)

    def test_different_seed_different_order(self, tiny_csv):
        """Reproducibility: different seeds give different order."""
        s1 = CsvDataSource(tiny_csv, batch_size=5, seed=1)
        s2 = CsvDataSource(tiny_csv, batch_size=5, seed=999)
        b1 = next(iter(s1))
        b2 = next(iter(s2))
        # With 5 samples, likely different order (tiny prob of collision)
        assert not np.array_equal(b1.inputs, b2.inputs)

    def test_no_shuffle_preserves_order(self, tiny_csv):
        """Happy Path: shuffle=False preserves CSV row order."""
        source = CsvDataSource(tiny_csv, batch_size=5, shuffle=False)
        batch = next(iter(source))
        expected = np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4], [5.0, 0.5]])
        np.testing.assert_array_equal(batch.inputs, expected)


class TestCsvDataSourceEdgeCases:
    """Edge case and error handling tests."""

    def test_missing_file_raises(self):
        """Error: FileNotFoundError for nonexistent CSV."""
        source = CsvDataSource("/nonexistent/path.csv", batch_size=2)
        with pytest.raises(FileNotFoundError):
            list(source)

    def test_lazy_load_file_not_read_on_init(self):
        """Happy Path: CSV is not read during __init__ (lazy loading)."""
        source = CsvDataSource("/nonexistent/path.csv", batch_size=2)
        # No error during construction -- file not accessed yet
        assert source is not None

    def test_len_consistent(self, tiny_csv):
        """Happy Path: len() is consistent across multiple calls."""
        source = CsvDataSource(tiny_csv, batch_size=2)
        assert len(source) == 3
        assert len(source) == 3  # second call same result

    def test_single_batch_exact(self, tiny_csv):
        """Boundary: batch_size equal to n_samples yields 1 batch."""
        source = CsvDataSource(tiny_csv, batch_size=5)
        assert len(source) == 1

    def test_iter_twice(self, tiny_csv):
        """Happy Path: iterating twice produces data both times (re-shuffles)."""
        source = CsvDataSource(tiny_csv, batch_size=5, seed=42)
        list(source)  # first pass
        batches = list(source)  # second pass
        assert len(batches) == 1
