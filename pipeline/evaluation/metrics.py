"""Pure numpy-based metric functions and the Metrics container.

Provides accuracy, precision, recall, F1 score, and confusion matrix
as stateless pure functions, plus a :class:`Metrics` container that
decouples the evaluate stage from specific metric choices.
"""
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

ArrayLike = np.ndarray | Any


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of correct predictions.

    Args:
        y_true: Ground-truth labels of shape ``(n_samples,)``.
        y_pred: Predicted labels of shape ``(n_samples,)``.

    Returns:
        Accuracy in ``[0.0, 1.0]``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
    pos_label: int | None = None,
) -> float:
    """Precision: TP / (TP + FP).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: ``"binary"`` or ``"macro"``. Macro averages per-class precision.
        pos_label: Positive class label for binary mode. When ``None`` (default)
            and ``average="binary"``, auto-detects from the unique sorted labels
            using ``classes[-1]``.

    Returns:
        Precision in ``[0.0, 1.0]``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary" and len(classes) <= 2:
        if pos_label is None:
            pos_label = classes[-1]
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    # macro
    scores: list[float] = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(scores))


def recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
    pos_label: int | None = None,
) -> float:
    """Recall: TP / (TP + FN).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: ``"binary"`` or ``"macro"``. Macro averages per-class recall.
        pos_label: Positive class label for binary mode. When ``None`` (default)
            and ``average="binary"``, auto-detects from the unique sorted labels
            using ``classes[-1]``.

    Returns:
        Recall in ``[0.0, 1.0]``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary" and len(classes) <= 2:
        if pos_label is None:
            pos_label = classes[-1]
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    scores: list[float] = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(scores))


def f1_score(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary"
) -> float:
    """F1 score: harmonic mean of precision and recall.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: ``"binary"`` or ``"macro"``.

    Returns:
        F1 score in ``[0.0, 1.0]``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary" and len(classes) <= 2:
        p = precision(y_true, y_pred, average="binary")
        r = recall(y_true, y_pred, average="binary")
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    scores: list[float] = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(scores))


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    num_classes: int | None = None,
) -> np.ndarray:
    """Confusion matrix C where C[i,j] = count of true=i predicted=j.

    Args:
        y_true: Ground-truth labels of shape ``(n_samples,)``.
        y_pred: Predicted labels of shape ``(n_samples,)``.
        num_classes: Number of classes. Auto-detected from data if None.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=False):
        cm[t, p] += 1
    return cm


class Metrics:
    """Collection of metric functions with cached results.

    Decouples the evaluate stage from knowledge of which specific
    metrics are being used. Holds both the functions and their
    computed values.

    Usage::

        state.metrics = Metrics(accuracy=accuracy, f1=f1_score)
        results = state.metrics.compute(y_true, y_pred)
        print(state.metrics["accuracy"])  # -> 0.92
    """

    def __init__(
        self, **named_metrics: Callable[[ArrayLike, ArrayLike], float]
    ) -> None:
        """Register named metric functions.

        Args:
            **named_metrics: ``name=function`` pairs (e.g., ``accuracy=accuracy``).
        """
        self._metrics: dict[str, Callable[[ArrayLike, ArrayLike], float]] = (
            named_metrics
        )
        self._values: dict[str, float] = {}

    def compute(
        self, y_true: ArrayLike, y_pred: ArrayLike
    ) -> dict[str, float]:
        """Run all registered metrics and cache results.

        Args:
            y_true: Ground-truth labels of shape ``(n_samples,)``.
            y_pred: Predicted labels of shape ``(n_samples,)``.

        Returns:
            ``{name: value}`` dict with one entry per registered metric.
        """
        self._values = {
            name: fn(y_true, y_pred) for name, fn in self._metrics.items()
        }
        return self._values

    def __getitem__(self, name: str) -> float:
        """Access a computed metric value by name.

        Args:
            name: Metric name as registered.

        Returns:
            Computed float value.

        Raises:
            KeyError: If name was not registered or compute() not called.
        """
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        """Yield registered metric names."""
        return iter(self._metrics)

    def __len__(self) -> int:
        """Number of registered metrics."""
        return len(self._metrics)

    def __contains__(self, name: str) -> bool:
        """Check if a metric name is registered."""
        return name in self._metrics
