"""Pure-numpy model and optimizer adapters.

Provides :class:`NumpyModel` — a :class:`ModelProtocol` backed by
numpy arrays with explicit :class:`Parameter` objects — and
:class:`NumpyOptimizer` — an :class:`OptimizerProtocol` that
delegates to update rules (SGD, Adam).
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from pipeline.protocols import ArrayLike, ModelProtocol, OptimizerProtocol, Parameter


class NumpyModel(ModelProtocol):
    """Model backed by numpy arrays with explicit Parameter objects.

    Supports configurable layer stacks for both single-neuron (M2)
    and multi-layer (M3) use. ``parameters()`` returns live
    :class:`Parameter` references — the optimizer mutates them
    in-place via ``param.data`` and ``param.grad``.

    ``forward()`` caches intermediate activations in ``self._cache``
    so that ``backward()`` can propagate gradients through all layers
    using the chain rule.

    Usage::

        # Single neuron (M2): one (W, b) pair
        model = NumpyModel([(W1, b1)])

        # Two-layer MLP (M3)
        model = NumpyModel([(W1, b1), (W2, b2)], activation="relu")

    Args:
        layers: List of ``(weight, bias)`` tuples. Weight shape is
            ``(out_features, in_features)``, bias shape is
            ``(out_features,)``. One tuple = one layer.
        activation: Activation for hidden layers. ``"relu"`` only
            for now. Last layer always outputs raw logits.

    Raises:
        ValueError: If ``layers`` is empty.
    """

    def __init__(
        self,
        layers: list[tuple[np.ndarray, np.ndarray]],
        activation: str = "relu",
    ) -> None:
        if not layers:
            raise ValueError("layers must not be empty")
        self._params: list[Parameter] = []
        for i, (W, b) in enumerate(layers):
            self._params.append(
                Parameter(data=W, grad=np.zeros_like(W), name=f"W{i}")
            )
            self._params.append(
                Parameter(data=b, grad=np.zeros_like(b), name=f"b{i}")
            )
        self._activation = activation
        self._training = True
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        """Forward pass through all layers.

        Caches intermediate activations (``a{i}`` = input to layer i,
        ``z{i}`` = pre-activation of layer i) for use by
        :meth:`backward`.

        Args:
            inputs: Feature matrix, shape ``(batch_size, in_features)``.

        Returns:
            Output logits, shape ``(batch_size, out_features)``.
        """
        x = np.asarray(inputs, dtype=np.float64)
        self._cache = {}
        n_layers = len(self._params) // 2

        for i in range(n_layers):
            W = self._params[2 * i].data
            b = self._params[2 * i + 1].data
            self._cache[f"a{i}"] = x  # input to this layer
            z = x @ W.T + b
            if i < n_layers - 1:
                self._cache[f"z{i}"] = z  # pre-activation (for relu grad)
                x = self._apply_activation(z)
            else:
                x = z  # last layer: no activation (raw logits)

        return x

    def backward(self, dL_doutput: np.ndarray) -> None:
        """Backpropagate gradient through all layers.

        Populates ``param.grad`` for every parameter. Call after
        :meth:`forward` so that ``self._cache`` is populated.

        Args:
            dL_doutput: Gradient of loss w.r.t. model output,
                shape ``(batch_size, out_features)``.
        """
        grad = dL_doutput
        n_layers = len(self._params) // 2

        for i in range(n_layers - 1, -1, -1):
            W_param = self._params[2 * i]
            b_param = self._params[2 * i + 1]
            a_prev = self._cache[f"a{i}"]

            # Apply activation gradient for hidden layers
            if i < n_layers - 1:
                z = self._cache[f"z{i}"]
                grad = grad * (z > 0)  # ReLU derivative

            # dL/db = sum of incoming gradients over batch
            b_param.grad = grad.sum(axis=0)
            # dL/dW = grad.T @ a_prev
            W_param.grad = grad.T @ a_prev

            # Propagate to previous layer
            if i > 0:
                grad = grad @ W_param.data

    def parameters(self) -> Iterable[Parameter]:
        """Yield all trainable parameters.

        Returns live references — the optimizer mutates them in-place.
        """
        yield from self._params

    def train_mode(self) -> None:
        """Switch to training mode."""
        self._training = True

    def eval_mode(self) -> None:
        """Switch to evaluation mode."""
        self._training = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function element-wise.

        Args:
            z: Pre-activation array.

        Returns:
            Activated array, same shape as ``z``.

        Raises:
            ValueError: If activation is unknown.
        """
        if self._activation == "relu":
            return np.maximum(0, z)
        raise ValueError(f"Unknown activation: {self._activation}")


class NumpyOptimizer(OptimizerProtocol):
    """Optimizer that mutates :class:`Parameter` objects in-place.

    Satisfies :class:`OptimizerProtocol` for the pipeline while
    delegating the actual update formula to a rule object (SGD,
    Adam, etc.). The rule must have an ``update(param)`` method.

    Usage::

        from pipeline.training.optimizers import SGD
        opt = NumpyOptimizer(model.parameters(), SGD(lr=0.01))
        opt.zero_grad()
        # ... backward pass ...
        opt.step()

    Args:
        parameters: Live Parameter references from a model.
        rule: An optimizer rule with an ``update(param)`` method
            that mutates ``param.data`` based on ``param.grad``.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        rule: object,
    ) -> None:
        self._params = list(parameters)
        self._rule = rule

    def step(self) -> None:
        """Update all parameters using the rule.

        Calls ``rule.update(param)`` for each parameter.
        """
        for p in self._params:
            self._rule.update(p)

    def zero_grad(self) -> None:
        """Reset all parameter gradients to zero.

        Skips parameters where ``grad`` is ``None``.
        """
        for p in self._params:
            if p.grad is not None:
                p.grad.fill(0.0)
