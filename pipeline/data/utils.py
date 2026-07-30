"""Utility functions for data handling."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pipeline.protocols import DataStream


def collect_arrays(stream: DataStream) -> tuple[np.ndarray, np.ndarray]:
    """Drain a DataStream into (X, y) numpy arrays.

    Iterates through all batches, concatenating inputs and targets.
    Useful for sklearn models that need the full dataset at once,
    and for evaluation loops that aggregate predictions.

    Args:
        stream: Any DataStream implementation.

    Returns:
        (X, y) tuple where X has shape ``(n_samples, n_features)``
        and y has shape ``(n_samples,)`` or ``(n_samples, n_targets)``.

    Raises:
        ValueError: If the stream yields no batches.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for batch in stream:
        xs.append(np.asarray(batch.inputs))
        ys.append(np.asarray(batch.targets))
    if not xs:
        raise ValueError("Cannot collect arrays from an empty DataStream")
    return np.concatenate(xs), np.concatenate(ys)
