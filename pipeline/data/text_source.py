"""Text file data source.

Provides :class:`TextDataSource`, a :class:`DataStream` implementation
that reads labeled text from a CSV file and yields :class:`Batch` objects
whose inputs are lists of strings.
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


class TextDataSource(DataStream):
    """Labeled-text CSV -> Batch iterator.

    Reads a CSV with a text column and a label column into a pandas
    DataFrame, optionally shuffles rows, and yields batches as
    :class:`Batch` objects. ``inputs`` is a ``list[str]`` of raw texts;
    ``targets`` is an ``np.ndarray[int64]`` of labels.

    The pattern mirrors :class:`~pipeline.data.csv_source.CsvDataSource`.

    Lazy-loading: the CSV is read on the first call to :meth:`__iter__`,
    :meth:`__len__`, or the :attr:`n_samples`/:attr:`vocab` properties, not
    during ``__init__``. This keeps construction cheap and defers I/O.

    Usage::

        stream = TextDataSource("train.csv", text_column="text",
                                label_column="label", batch_size=32)
        for batch in stream:
            print(batch.inputs)      # -> list of raw strings
            print(batch.targets)     # -> np.ndarray of int labels
    """

    def __init__(
        self,
        file_path: str | Path,
        text_column: str = "text",
        label_column: str = "label",
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """Create a text data source.

        Args:
            file_path: Path to a ``.csv`` file.
            text_column: Name of the column holding raw text.
            label_column: Name of the column holding integer labels.
            batch_size: Number of samples per batch.
            shuffle: If True, shuffle rows before batching.
            seed: Random seed for reproducible shuffling.
        """
        self.file_path = Path(file_path)
        self.text_column = text_column
        self.label_column = label_column
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._df: pd.DataFrame | None = None
        self._n_samples = 0

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
            self._n_samples = len(self._df)
        return self._df

    @property
    def n_samples(self) -> int:
        """Number of rows in the source CSV."""
        self._load()
        return self._n_samples

    @property
    def vocab(self) -> set[str]:
        """The set of all whitespace-separated tokens across all texts."""
        df = self._load()
        words: set[str] = set()
        for text in df[self.text_column]:
            words.update(str(text).split())
        return words

    def __iter__(self) -> Iterator[Batch]:
        """Yield one :class:`Batch` per batch window with ``list[str]`` inputs."""
        df = self._load()
        indices = np.arange(self._n_samples)
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, self._n_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch_df = df.iloc[batch_idx]
            texts = [str(t) for t in batch_df[self.text_column]]
            labels = batch_df[self.label_column].to_numpy(dtype=np.int64)
            yield Batch(inputs=texts, targets=labels)

    def __len__(self) -> int:
        """Number of batches (``ceil(n_samples / batch_size)``)."""
        self._load()
        return max(1, math.ceil(self._n_samples / self.batch_size))
