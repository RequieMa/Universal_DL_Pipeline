"""Pure-numpy optimizer update rules.

Each rule is a standalone class with an :meth:`update` method
that mutates a :class:`Parameter` in-place. They are used by
:class:`NumpyOptimizer` which handles the iteration over parameters
and the ``zero_grad``/``step`` pipeline contract.

Usage::

    from pipeline.training.optimizers import SGD
    from pipeline.adapters.numpy_adapter import NumpyOptimizer

    opt = NumpyOptimizer(model.parameters(), SGD(lr=0.01))
    opt.zero_grad()
    # ... backward pass populates param.grad ...
    opt.step()
"""

from __future__ import annotations

import numpy as np

from pipeline.protocols import Parameter


class SGD:
    """Vanilla stochastic gradient descent: ``w = w - lr * grad``.

    The simplest possible optimizer. Students start here (M2) before
    meeting momentum and adaptive methods.

    Usage::

        sgd = SGD(lr=0.01)
        sgd.update(param)  # param.data -= lr * param.grad

    Args:
        lr: Learning rate. Must be positive.

    Raises:
        ValueError: If ``lr <= 0``.
    """

    def __init__(self, lr: float = 0.01) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = lr

    def update(self, param: Parameter) -> None:
        """Apply the SGD update to one parameter.

        ``param.data = param.data - lr * param.grad``

        Args:
            param: Parameter with ``.data`` and ``.grad`` populated.
        """
        assert param.grad is not None, f"Parameter {param.name} has no gradient"
        param.data = param.data - self.lr * param.grad


class Adam:
    """Adam optimizer: momentum + adaptive per-parameter learning rates.

    ``m = beta1 * m + (1 - beta1) * grad`` (first moment)
    ``v = beta2 * v + (1 - beta2) * grad^2`` (second moment)
    ``m_hat = m / (1 - beta1^t)``, ``v_hat = v / (1 - beta2^t)``
    ``w = w - lr * m_hat / (sqrt(v_hat) + eps)``

    Each parameter gets its own ``m`` and ``v`` buffers (keyed by
    ``id(param)``). The ``t`` counter is shared across all parameters.

    Usage::

        adam = Adam(lr=0.001)
        adam.update(param)

    Args:
        lr: Learning rate. Must be positive.
        betas: ``(beta1, beta2)`` decay rates for first/second moments.
        eps: Small constant for numerical stability.

    Raises:
        ValueError: If ``lr <= 0``.
    """

    def __init__(
        self,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}
        self._t = 0

    def update(self, param: Parameter) -> None:
        """Apply one Adam update step.

        Args:
            param: Parameter with ``.data`` and ``.grad`` populated.
        """
        self._t += 1
        assert param.grad is not None, f"Parameter {param.name} has no gradient"
        key = id(param)
        if key not in self._m:
            self._m[key] = np.zeros_like(param.data)
            self._v[key] = np.zeros_like(param.data)

        grad = param.grad
        self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * grad
        self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * grad**2

        m_hat = self._m[key] / (1 - self.beta1**self._t)
        v_hat = self._v[key] / (1 - self.beta2**self._t)

        param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
