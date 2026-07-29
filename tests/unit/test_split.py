"""Unit tests for pipeline.data.split -- train_test_split."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.data.split import train_test_split
from pipeline.protocols import Batch, DataStream

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_csv_path():
    """Path to the committed test fixture."""
    from pathlib import Path

    return str(Path(__file__).parent.parent / "fixtures" / "tiny_titanic.csv")


# ── Helpers ─────────────────────────────────────────────────────────────


class _StreamFromBatches(DataStream):
    """Test double: DataStream backed by an in-memory list of Batches."""

    def __init__(self, batches: list[Batch]) -> None:
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


@pytest.fixture
def sample_stream():
    """Return a DataStream with 10 samples (batch_size=5)."""
    batches = [
        Batch(
            inputs=np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]),
            targets=np.array([0, 0, 1, 1, 0]),
        ),
        Batch(
            inputs=np.array([[5.0], [6.0], [7.0], [8.0], [9.0]]),
            targets=np.array([1, 1, 0, 0, 1]),
        ),
    ]
    return _StreamFromBatches(batches)


# ── Tests ───────────────────────────────────────────────────────────────


class TestTrainTestSplit:
    """Tests for train_test_split()."""

    # -- Happy path basics ------------------------------------------------

    def test_returns_two_streams(self, tiny_csv_path):
        """Happy Path: returns (train_stream, val_stream) tuple."""
        from pipeline.data.csv_source import CsvDataSource

        source = CsvDataSource(tiny_csv_path, batch_size=2, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.8)
        assert train is not None
        assert val is not None
        assert train is not val

    def test_train_ratio_split(self, tiny_csv_path):
        """Happy Path: split respects train_ratio."""
        from pipeline.data.csv_source import CsvDataSource

        source = CsvDataSource(tiny_csv_path, batch_size=1, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.7, shuffle=False)
        # 10 samples: ~7 train, ~3 val
        train_samples = sum(len(b.inputs) for b in train)
        val_samples = sum(len(b.inputs) for b in val)
        assert train_samples == 7
        assert val_samples == 3

    def test_both_streams_iterable(self, tiny_csv_path):
        """Happy Path: both returned streams can be independently iterated."""
        from pipeline.data.csv_source import CsvDataSource

        source = CsvDataSource(tiny_csv_path, batch_size=2, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.8)
        train_batches = list(train)
        val_batches = list(val)
        assert len(train_batches) > 0
        assert len(val_batches) > 0

    # -- Shuffle reproducibility ------------------------------------------

    def test_shuffle_reproducibility(self, tiny_csv_path):
        """Reproducibility: same seed yields same split."""
        from pipeline.data.csv_source import CsvDataSource

        s1 = CsvDataSource(tiny_csv_path, batch_size=2, shuffle=False)
        s2 = CsvDataSource(tiny_csv_path, batch_size=2, shuffle=False)
        t1, v1 = train_test_split(s1, seed=42)
        t2, v2 = train_test_split(s2, seed=42)
        t1_data = np.concatenate([b.inputs for b in t1])
        t2_data = np.concatenate([b.inputs for b in t2])
        np.testing.assert_array_equal(t1_data, t2_data)

    # -- Non-overlap -----------------------------------------------------

    def test_no_overlap_between_splits(self, tiny_csv_path):
        """Happy Path: train and val sets have no overlapping samples."""
        from pipeline.data.csv_source import CsvDataSource

        source = CsvDataSource(tiny_csv_path, batch_size=1, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.8, shuffle=False)
        train_data = np.concatenate([b.inputs for b in train])
        val_data = np.concatenate([b.inputs for b in val])
        for row in val_data:
            assert not any(np.array_equal(row, t_row) for t_row in train_data)

    # -- __len__ consistency ---------------------------------------------

    def test_len_consistent(self, tiny_csv_path):
        """Happy Path: __len__ returns number of batches."""
        from pipeline.data.csv_source import CsvDataSource

        source = CsvDataSource(tiny_csv_path, batch_size=3, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.7, shuffle=False)
        assert len(train) > 0
        assert len(val) > 0

    # -- Boundary cases --------------------------------------------------

    def test_edge_ratio_near_one(self, tiny_csv_path):
        """Boundary: train_ratio=0.99 -> almost all data in train."""
        from pipeline.data.csv_source import CsvDataSource

        source = CsvDataSource(tiny_csv_path, batch_size=1, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.99, shuffle=False)
        train_samples = sum(len(b.inputs) for b in train)
        val_samples = sum(len(b.inputs) for b in val)
        assert train_samples == 9
        assert val_samples == 1

    def test_all_data_preserved(self, tiny_csv_path):
        """Happy Path: no samples lost during split."""
        from pipeline.data.csv_source import CsvDataSource

        source = CsvDataSource(tiny_csv_path, batch_size=1, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.5, shuffle=False)
        train_count = sum(len(b.inputs) for b in train)
        val_count = sum(len(b.inputs) for b in val)
        assert train_count + val_count == 10

    # -- Generic DataStream (not CsvDataSource) ---------------------------

    def test_generic_data_stream(self, sample_stream):
        """Happy Path: works with any DataStream, not just CsvDataSource."""
        train, val = train_test_split(sample_stream, train_ratio=0.6, shuffle=False)
        # 10 samples, 60% -> 6 train, 4 val
        train_count = sum(len(b.inputs) for b in train)
        val_count = sum(len(b.inputs) for b in val)
        assert train_count == 6
        assert val_count == 4

    def test_generic_data_stream_shuffle(self, sample_stream):
        """Happy Path: shuffle works with generic DataStream."""
        train, val = train_test_split(sample_stream, train_ratio=0.5, shuffle=True, seed=42)
        train_data = np.concatenate([b.inputs for b in train])
        val_data = np.concatenate([b.inputs for b in val])
        # Convert each row to tuple for set-based uniqueness check
        train_hashes = {tuple(row) for row in train_data.tolist()}
        val_hashes = {tuple(row) for row in val_data.tolist()}
        # All train rows should be unique
        assert len(train_hashes) == len(train_data), "Duplicate rows in train set"
        assert len(val_hashes) == len(val_data), "Duplicate rows in val set"
        # No overlap between train and val
        assert train_hashes.isdisjoint(val_hashes)
        # Total count preserved
        total = len(train_data) + len(val_data)
        assert total == 10

    # -- Empty stream ----------------------------------------------------

    def test_empty_source(self):
        """Edge: empty DataStream returns two empty streams."""
        empty = _StreamFromBatches([])
        train, val = train_test_split(empty, train_ratio=0.8)
        assert len(list(train)) == 0
        assert len(list(val)) == 0

    # -- Default parameters ----------------------------------------------

    def test_default_params(self, tiny_csv_path):
        """Boundary: uses defaults (train_ratio=0.8, shuffle=True, seed=42)."""
        from pipeline.data.csv_source import CsvDataSource

        source = CsvDataSource(tiny_csv_path, batch_size=2, shuffle=False)
        train, val = train_test_split(source)  # no explicit args
        train_count = sum(len(b.inputs) for b in train)
        val_count = sum(len(b.inputs) for b in val)
        assert train_count + val_count == 10
        assert train_count <= 8  # 80% of 10
