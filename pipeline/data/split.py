"""Train/validation split utilities.

Splits a :class:`DataStream` into two independent streams
for training and validation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from pipeline.protocols import Batch, DataStream


def _collect_data(source: DataStream) -> tuple[np.ndarray, np.ndarray]:
    """Collect all data from a DataStream into contiguous arrays.

    Iterates over ``source``, converts each batch to arrays,
    and concatenates them.

    Args:
        source: A :class:`DataStream` to read from.

    Returns:
        ``(inputs, targets)`` where each is a :class:`np.ndarray`.
        Both are empty arrays when the stream is empty.
    """
    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for batch in source:
        all_inputs.append(np.asarray(batch.inputs))
        all_targets.append(np.asarray(batch.targets))

    if not all_inputs:
        return np.array([]), np.array([])

    return np.concatenate(all_inputs, axis=0), np.concatenate(all_targets)


def train_test_split(
    source: DataStream,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[DataStream, DataStream]:
    """Split a DataStream into (train_stream, val_stream).

    Collects all data from ``source`` into memory, shuffles if requested,
    then splits by ``train_ratio``. Returns two :class:`_InMemoryDataStream`
    instances backed by the split data.

    For large datasets that don't fit in memory, a streaming split
    can be added in a later phase.

    Args:
        source: Any :class:`DataStream` to split.
        train_ratio: Fraction of data for training (in ``(0, 1)``).
        shuffle: If True, shuffle samples before splitting.
        seed: Random seed for reproducible shuffling.

    Returns:
        ``(train_stream, val_stream)`` tuple of :class:`DataStream`.
    """
    inputs, targets = _collect_data(source)
    if len(inputs) == 0:
        return (
            _InMemoryDataStream(np.array([]), np.array([])),
            _InMemoryDataStream(np.array([]), np.array([])),
        )
    n_samples = len(inputs)

    # Shuffle indices
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)

    # Split
    n_train = math.floor(n_samples * train_ratio)
    # Ensure at least 1 sample in each set when possible
    n_train = max(1, min(n_train, n_samples - 1))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    return (
        _InMemoryDataStream(inputs[train_idx], targets[train_idx]),
        _InMemoryDataStream(inputs[val_idx], targets[val_idx]),
    )


class _InMemoryDataStream(DataStream):
    """A :class:`DataStream` backed by pre-split numpy arrays.

    Internal helper for :func:`train_test_split`. Not part of the public API.
    Stores the arrays for one side of the split and yields them as a single
    batch on iteration.

    For Phase 1 this is sufficient: batch sizing is handled upstream by
    :class:`CsvDataSource` before the split, or consumers split the single
    batch themselves.
    """

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Store the arrays.

        Args:
            inputs: Feature array of shape ``(n_samples, *feature_dims)``.
            targets: Target array of shape ``(n_samples,)``.
        """
        self._inputs = inputs
        self._targets = targets

    def __iter__(self) -> Iterator[Batch]:
        """Yield one :class:`Batch` containing all stored data."""
        if len(self._inputs) > 0:
            yield Batch(inputs=self._inputs, targets=self._targets)

    def __len__(self) -> int:
        """Number of batches -- returns 1 when there is data, 0 otherwise."""
        return 1 if len(self._inputs) > 0 else 0
