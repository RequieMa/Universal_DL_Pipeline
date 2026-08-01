"""Unit tests for pipeline.export.to_csv — to_csv()."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from pipeline.export.to_csv import to_csv


class TestToCsv:
    """Tests for to_csv()."""

    def test_writes_file(self):
        """Happy Path: to_csv creates a file at the given path."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            to_csv(data, path)
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_round_trip(self):
        """Happy Path: written CSV can be read back correctly."""
        import pandas as pd

        data = np.array([[1.0, 0.5], [2.0, 0.3], [3.0, 0.1]])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            to_csv(data, path, columns=["pred_0", "pred_1"])
            df = pd.read_csv(path)
            assert list(df.columns) == ["pred_0", "pred_1"]
            assert df.shape == (3, 2)
            np.testing.assert_array_almost_equal(df.values, data)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_no_columns_default_header(self):
        """Happy Path: when columns=None, default numeric headers are written."""
        import pandas as pd

        data = np.array([[1.0, 2.0]])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            to_csv(data, path)
            df = pd.read_csv(path)
            assert df.shape == (1, 2)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_single_row(self):
        """Boundary: single-row array."""
        data = np.array([[0.1, 0.9]])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            to_csv(data, path)
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_1d_array(self):
        """Happy Path: 1D array is reshaped for CSV writing."""
        data = np.array([0.1, 0.2, 0.3])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            to_csv(data, path)
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_creates_parent_directory(self):
        """Happy Path: parent directories are created if needed."""
        import os
        import tempfile

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "subdir", "output.csv")
        data = np.array([[1.0]])
        try:
            to_csv(data, path)
            assert Path(path).exists()
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
