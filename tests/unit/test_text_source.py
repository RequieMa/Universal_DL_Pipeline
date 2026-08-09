"""Unit tests for pipeline.data.text_source -- TextDataSource."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from pipeline.data.text_source import TextDataSource
from pipeline.protocols import Batch


@pytest.fixture
def tiny_text_csv() -> str:
    """Write a temporary labeled-text CSV for testing."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "text": [
                "the quick brown fox",
                "jumps over the lazy dog",
                "the fox is quick",
                "brown dog sleeps",
                "hello world",
            ],
            "label": [0, 1, 0, 1, 0],
        }
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        return f.name


class TestTextDataSourceBasic:
    """Basic functionality tests."""

    def test_construction(self, tiny_text_csv):
        """Happy Path: __init__ stores config and defers file I/O."""
        source = TextDataSource(
            tiny_text_csv,
            text_column="text",
            label_column="label",
            batch_size=2,
            seed=42,
        )
        assert source.batch_size == 2
        assert source.text_column == "text"
        assert source.label_column == "label"
        assert source._df is None  # not loaded yet (lazy)

    def test_n_samples(self, tiny_text_csv):
        """Happy Path: n_samples returns row count."""
        source = TextDataSource(tiny_text_csv, batch_size=2)
        assert source.n_samples == 5

    def test_vocab(self, tiny_text_csv):
        """Happy Path: vocab returns union of whitespace-split tokens."""
        source = TextDataSource(tiny_text_csv, batch_size=5, shuffle=False)
        vocab = source.vocab
        expected = {
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "is",
            "sleeps",
            "hello",
            "world",
        }
        assert vocab == expected

    def test_len_returns_num_batches(self, tiny_text_csv):
        """Happy Path: __len__ returns ceil(n_samples / batch_size)."""
        source = TextDataSource(tiny_text_csv, batch_size=2)
        assert len(source) == 3  # 5 / 2 -> ceil = 3


class TestTextDataSourceIteration:
    """Iteration, typing, and shuffle tests."""

    def test_iter_yields_str_inputs(self, tiny_text_csv):
        """Happy Path: inputs are lists of raw strings."""
        source = TextDataSource(tiny_text_csv, batch_size=5, shuffle=False)
        batch = next(iter(source))
        assert isinstance(batch, Batch)
        assert isinstance(batch.inputs, list)
        assert all(isinstance(t, str) for t in batch.inputs)
        assert batch.inputs[0] == "the quick brown fox"

    def test_iter_yields_int64_targets(self, tiny_text_csv):
        """Happy Path: targets are int64 numpy arrays."""
        source = TextDataSource(tiny_text_csv, batch_size=5, shuffle=False)
        batch = next(iter(source))
        assert isinstance(batch.targets, np.ndarray)
        assert batch.targets.dtype == np.int64
        np.testing.assert_array_equal(batch.targets, np.array([0, 1, 0, 1, 0]))

    def test_shuffle_deterministic(self, tiny_text_csv):
        """Reproducibility: same seed yields same batch order."""
        s1 = TextDataSource(tiny_text_csv, batch_size=2, seed=42)
        s2 = TextDataSource(tiny_text_csv, batch_size=2, seed=42)
        batches1 = list(s1)
        batches2 = list(s2)
        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2, strict=False):
            assert b1.inputs == b2.inputs
            np.testing.assert_array_equal(b1.targets, b2.targets)

    def test_no_shuffle_preserves_order(self, tiny_text_csv):
        """Happy Path: shuffle=False preserves CSV row order."""
        source = TextDataSource(tiny_text_csv, batch_size=5, shuffle=False)
        batch = next(iter(source))
        assert batch.inputs[0] == "the quick brown fox"
        assert batch.inputs[-1] == "hello world"

    def test_empty_csv_yields_no_batches(self):
        """Empty: a CSV with no rows iterates to zero batches."""
        import pandas as pd

        df = pd.DataFrame({"text": [], "label": []})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            path = f.name
        source = TextDataSource(path, batch_size=2)
        # __len__ must agree with the number of batches __iter__ yields.
        assert len(source) == 0
        assert list(source) == []
        assert len(source) == len(list(source))


class TestTextDataSourceEdgeCases:
    """Error handling tests."""

    def test_missing_file_raises(self):
        """Error: FileNotFoundError for nonexistent CSV."""
        source = TextDataSource("/nonexistent/path.csv", batch_size=2)
        with pytest.raises(FileNotFoundError):
            list(source)

    def test_lazy_load_file_not_read_on_init(self):
        """Happy Path: CSV is not read during __init__."""
        source = TextDataSource("/nonexistent/path.csv", batch_size=2)
        assert source._df is None

    def test_iter_twice(self, tiny_text_csv):
        """Happy Path: iterating twice produces data both times (re-shuffles)."""
        source = TextDataSource(tiny_text_csv, batch_size=5, seed=42)
        first = list(source)
        second = list(source)
        assert len(first) == 1
        assert len(second) == 1
