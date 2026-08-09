"""CSV file data source.

Provides :class:`CsvDataSource`, a :class:`DataStream` implementation
that reads tabular data from CSV files and yields :class:`Batch` objects.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from pipeline.protocols import Batch, DataStream


class CsvDataSource(DataStream):
    """CSV file -> Batch iterator.

    Reads a CSV into a pandas DataFrame, optionally shuffles rows,
    and yields batches as :class:`Batch` objects. The last column is
    used as the target unless ``target_column`` is specified.

    Lazy-loading: the CSV is read on the first call to :meth:`__iter__`
    or :meth:`__len__`, not during ``__init__``. This keeps construction
    cheap and defers I/O.

    Usage::

        stream = CsvDataSource("train.csv", batch_size=32)
        print(len(stream))  # -> number of batches
        for batch in stream:
            print(batch.inputs.shape, batch.targets.shape)
    """

    def __init__(
        self,
        file_path: str | Path,
        batch_size: int = 32,
        target_column: str | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """Create a CSV data source.

        Args:
            file_path: Path to a ``.csv`` file.
            batch_size: Number of samples per batch.
            target_column: Column name for labels. Default: last column.
            shuffle: If True, shuffle rows before batching.
            seed: Random seed for reproducible shuffling.
        """
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self.target_column = target_column
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._df: pd.DataFrame | None = None
        self._n_samples: int = 0

    def _load(self) -> pd.DataFrame:
        """Lazy-load the CSV on first access.

        Returns:
            The loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if self._df is None:
            import pandas as pd

            self._df = pd.read_csv(self.file_path)
            if self.target_column is None:
                self.target_column = self._df.columns[-1]
            self._n_samples = len(self._df)
        return self._df

    def __iter__(self) -> Iterator[Batch]:
        """Yield one :class:`Batch` per batch window.

        If ``shuffle=True``, row order is randomized on each call to
        ``__iter__``, so repeated iteration produces different orders.
        """
        df = self._load()
        indices = np.arange(self._n_samples)
        if self.shuffle:
            self.rng.shuffle(indices)

        feature_cols = [c for c in df.columns if c != self.target_column]
        for start in range(0, self._n_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch_df = df.iloc[batch_idx]
            inputs = batch_df[feature_cols].to_numpy(dtype=np.float64)
            targets = batch_df[self.target_column].to_numpy()
            yield Batch(inputs=inputs, targets=targets)

    def __len__(self) -> int:
        """Number of batches (``ceil(n_samples / batch_size)``)."""
        self._load()
        return math.ceil(self._n_samples / self.batch_size)
