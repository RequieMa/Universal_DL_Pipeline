"""Pure-numpy loss functions.

Each loss function satisfies :class:`LossProtocol` and returns a
:class:`Loss` object whose ``_backward_fn`` closure computes the
gradient of the loss with respect to the predictions and propagates
it through the attached model (if any).

For use without a model (standalone loss computation), omit the
``model`` argument — ``backward()`` becomes a no-op.

Usage::

    loss_fn = MSELoss(model=numpy_model)
    loss = loss_fn(predictions, targets)  # Loss with _backward_fn
    loss.backward()  # propagates to model params
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pipeline.protocols import ArrayLike, Loss, LossProtocol

if TYPE_CHECKING:
    from pipeline.adapters.numpy_adapter import NumpyModel


class MSELoss(LossProtocol):
    """Mean squared error: :math:`\\frac{1}{N}\\sum(\\hat{y} - y)^2`.

    The ``_backward_fn`` computes ``dL/dpred = 2 * (pred - target) / N``
    and propagates through the attached model via ``model.backward()``.

    Args:
        model: Optional :class:`NumpyModel`. When provided, ``backward()``
            propagates gradients through the model's layers and populates
            ``param.grad`` on every parameter. When ``None`` (default),
            ``backward()`` is a no-op — useful for standalone loss
            computation.
    """

    def __init__(self, model: NumpyModel | None = None) -> None:
        self._model = model

    def forward(self, predictions: ArrayLike, targets: ArrayLike) -> Loss:
        """Compute MSE loss.

        Args:
            predictions: Model output, shape ``(batch_size, *dims)``.
            targets: Ground truth, same shape as predictions.

        Returns:
            :class:`Loss` with ``value`` = mean squared error and
            ``_backward_fn`` that propagates gradients.
        """
        pred = np.asarray(predictions, dtype=np.float64)
        targ = np.asarray(targets, dtype=np.float64)
        diff = pred - targ
        value = float(np.mean(diff**2))
        n = diff.size
        model = self._model

        def _backward() -> None:
            if model is None:
                return
            dL_dpred = 2.0 * diff / n
            model.backward(dL_dpred)

        return Loss(value=value, _backward_fn=_backward)


class CrossEntropyLoss(LossProtocol):
    """Cross-entropy loss: softmax + negative log-likelihood.

    For integer targets: ``-mean(log(softmax(logits)[correct_class]))``.
    The ``_backward_fn`` computes ``dL/dlogits = (probs - one_hot)/N``
    and propagates through the attached model.

    Args:
        model: Optional :class:`NumpyModel`. When provided, ``backward()``
            propagates gradients. When ``None``, ``backward()`` is a no-op.
    """

    def __init__(self, model: NumpyModel | None = None) -> None:
        self._model = model

    def forward(self, predictions: ArrayLike, targets: ArrayLike) -> Loss:
        """Compute cross-entropy loss.

        Args:
            predictions: Raw logits, shape ``(batch_size, n_classes)``.
            targets: Integer class labels, shape ``(batch_size,)``.

        Returns:
            :class:`Loss` with ``value`` = cross-entropy and
            ``_backward_fn`` that computes ``dL/dlogits``.
        """
        logits = np.asarray(predictions, dtype=np.float64)
        targets_arr = np.asarray(targets)

        # Numerically stable softmax
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        batch_size = logits.shape[0]

        # Negative log-likelihood for correct class
        target_probs = probs[np.arange(batch_size), targets_arr.astype(np.int64)]
        value = float(-np.mean(np.log(target_probs + 1e-8)))

        # One-hot for gradient computation
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(batch_size), targets_arr.astype(np.int64)] = 1.0

        model = self._model

        def _backward() -> None:
            if model is None:
                return
            dL_dlogits = (probs - y_onehot) / batch_size
            model.backward(dL_dlogits)

        return Loss(value=value, _backward_fn=_backward)
