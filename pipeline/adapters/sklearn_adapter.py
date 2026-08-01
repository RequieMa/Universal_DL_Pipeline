"""Sklearn estimator adapter and stub components."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from pipeline.protocols import (
    ArrayLike,
    Loss,
    LossProtocol,
    ModelProtocol,
    OptimizerProtocol,
    Parameter,
)


class StubLoss(LossProtocol):
    """No-op loss for non-gradient models. Always returns 0.0.

    Used with :class:`SklearnModel` so ``build_model()`` always has
    a loss function to assign to ``state.loss_fn``.
    """

    def forward(self, predictions: ArrayLike, targets: ArrayLike) -> Loss:
        """Return a zero-valued Loss.

        Args:
            predictions: Ignored.
            targets: Ignored.

        Returns:
            A :class:`Loss` with ``value=0.0`` and no backward function.
        """
        return Loss(value=0.0)


class StubOptimizer(OptimizerProtocol):
    """No-op optimizer for non-gradient models.

    Used with :class:`SklearnModel` so ``build_model()`` always has
    an optimizer to assign to ``state.optimizer``.
    """

    def step(self) -> None:
        """No-op: sklearn models update via ``.fit()``, not gradients."""

    def zero_grad(self) -> None:
        """No-op: no gradients to zero."""


class SklearnModel(ModelProtocol):
    """Wraps any sklearn estimator as a :class:`ModelProtocol`.

    ``forward()`` delegates to ``predict_proba()`` for classifiers
    (returning class probabilities) or ``predict()`` for regressors.
    ``parameters()`` returns an empty list — sklearn models don't
    expose gradient parameters.

    The estimator is **not** fit by this adapter. The pipeline
    subclass calls ``estimator.fit()`` in an overridden ``train()``.

    Usage::

        from sklearn.linear_model import LogisticRegression
        model = SklearnModel(LogisticRegression(max_iter=1000))
        # Pipeline's train() calls model._estimator.fit(X, y)

    Attributes:
        _estimator: The wrapped sklearn estimator. Public for pipeline
            subclasses that need to call ``.fit()`` directly.
    """

    def __init__(self, estimator: Any) -> None:
        """Wrap an sklearn estimator.

        Args:
            estimator: Any sklearn-compatible estimator with ``fit()``
                and ``predict()``/``predict_proba()`` methods.
        """
        self._estimator = estimator

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        """Run inference through the wrapped estimator.

        For classifiers: delegates to ``predict_proba()`` (returns
        class probabilities). For regressors: delegates to
        ``predict()``.

        Args:
            inputs: Feature matrix of shape ``(batch_size, n_features)``.

        Returns:
            Predictions — class probabilities ``(batch_size, n_classes)``
            for classifiers, or scalar predictions ``(batch_size,)``
            for regressors.
        """
        inputs_arr = np.asarray(inputs)
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(inputs_arr)
        return self._estimator.predict(inputs_arr)

    def parameters(self) -> Iterable[Parameter]:
        """Return an empty iterable — sklearn has no gradient parameters.

        Returns:
            Empty list. The pipeline skips gradient updates when
            ``parameters()`` is empty.
        """
        return []

    def train_mode(self) -> None:
        """No-op. Sklearn handles train/eval internally."""

    def eval_mode(self) -> None:
        """No-op. Sklearn handles train/eval internally."""
