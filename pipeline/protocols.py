"""Abstract protocols for the Universal DL Pipeline.

Defines the framework-agnostic interfaces that every component must satisfy.
Any framework (numpy, PyTorch, JAX) can satisfy these interfaces via an adapter.

Protocols defined here:
    - :class:`ArrayLike` — type alias for array-like data
    - :class:`Parameter` — a trainable parameter
    - :class:`Batch` — one batch of data
    - :class:`DataStream` — iterable producing batches
    - :class:`ModelProtocol` — a model with forward pass and parameters
    - :class:`LossProtocol` — a loss function
    - :class:`OptimizerProtocol` — a parameter optimizer
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

# ── Type alias ──────────────────────────────────────────────────────────
# WHY: ArrayLike lets the pipeline accept numpy arrays, torch tensors, or
# any duck-typed array without binding to a specific framework.
ArrayLike = np.ndarray | Any


# ── Data structures ─────────────────────────────────────────────────────
@dataclass(eq=False)
class Parameter:
    """A trainable model parameter.

    Wraps a framework-specific tensor. The optimizer reads/writes
    ``data`` and ``grad`` through the parameter reference.

    Attributes:
        data: The parameter value (weight or bias tensor).
        grad: Accumulated gradient for this parameter, or None.
        name: Human-readable label (e.g. ``"layer1.weight"``).
    """

    data: ArrayLike
    grad: ArrayLike | None = None
    name: str = ""

    def __eq__(self, other: object) -> bool:
        """Compare two Parameters by value.

        Uses :func:`numpy.array_equal` so that array-valued fields
        compare correctly without raising ``ValueError`` on multi-
        element arrays.
        """
        if not isinstance(other, Parameter):
            return NotImplemented
        return (
            self.name == other.name
            and _array_equal(self.data, other.data)
            and _array_equal(self.grad, other.grad)
        )


def _array_equal(a: object, b: object) -> bool:
    """Check array equality, handling None."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return bool(np.array_equal(a, b))


@dataclass(eq=False)
class Batch:
    """One batch of data — inputs and targets.

    Framework-agnostic: the arrays inside can be numpy ndarrays,
    torch Tensors, or any duck-typed array-like object.

    Attributes:
        inputs: Feature matrix of shape ``(batch_size, *feature_dims)``.
        targets: Label tensor of shape ``(batch_size, *target_dims)``.
    """

    inputs: ArrayLike
    targets: ArrayLike

    def __eq__(self, other: object) -> bool:
        """Compare two Batches by value.

        Uses :func:`numpy.array_equal` so that array-valued fields
        compare correctly without raising ``ValueError`` on multi-
        element arrays.
        """
        if not isinstance(other, Batch):
            return NotImplemented
        return (
            _array_equal(self.inputs, other.inputs)
            and _array_equal(self.targets, other.targets)
        )


@dataclass
class Loss:
    """Loss computation result.

    Wraps the scalar loss value and an optional backward hook.
    ``float(loss)`` extracts the scalar; ``loss.backward()`` computes
    gradients into :attr:`Parameter.grad`. The backward pass is a no-op
    for non-gradient models (sklearn, Phase 1 fake models).

    Usage::

        loss = loss_fn(predictions, targets)
        print(f"{loss:.4f}")         # __float__ → scalar
        loss.backward()              # compute gradients (no-op for non-gradient models)
        optimizer.step()             # apply gradients
    """

    value: float
    _backward_fn: Callable[[], None] | None = None

    def backward(self) -> None:
        """Compute gradients for all trainable parameters.

        No-op for non-gradient models. For numpy/torch adapters,
        the concrete implementation provides the framework-specific
        gradient computation via ``_backward_fn``.
        """
        if self._backward_fn is not None:
            self._backward_fn()

    def __float__(self) -> float:
        """Extract the scalar loss value."""
        return self.value


# ── Abstract interfaces ──────────────────────────────────────────────────
class DataStream(ABC):
    """Iterable producing :class:`Batch` objects one at a time.

    The standard way to feed data into the pipeline. One iteration yields
    one batch. Concrete implementations wrap CSV files, image folders,
    or framework-specific DataLoaders.

    Usage::

        stream = CsvDataSource("train.csv", batch_size=32)
        for batch in stream:
            predictions = model.forward(batch.inputs)

    NOTE: :meth:`__len__` should return the number of *batches*, not samples.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Batch]:
        """Yield one :class:`Batch` per iteration step."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Total number of batches in this stream."""
        ...


class ModelProtocol(ABC):
    """A model: forward pass plus trainable parameters.

    The minimum contract a model must satisfy to work with the pipeline.
    Concrete implementations wrap sklearn estimators, numpy functions,
    or PyTorch ``nn.Module`` objects.

    Usage::

        class MyModel(ModelProtocol):
            def forward(self, inputs):
                return self._w @ inputs.T
            def parameters(self):
                return [Parameter(data=self._w, name="w")]
            def train_mode(self): ...
            def eval_mode(self): ...
    """

    @abstractmethod
    def forward(self, inputs: ArrayLike) -> ArrayLike:
        """Run a forward pass and return predictions.

        Args:
            inputs: Feature matrix of shape ``(batch_size, *feature_dims)``.

        Returns:
            Predictions of shape ``(batch_size, *output_dims)``.
        """
        ...

    @abstractmethod
    def parameters(self) -> Iterable[Parameter]:
        """Return all trainable parameters.

        NOTE: May return an empty iterable for models that don't expose
        parameters (e.g., sklearn estimators). The pipeline skips the
        gradient update (backward + optimizer step) in that case.
        """
        ...

    @abstractmethod
    def train_mode(self) -> None:
        """Switch the model to training mode.

        Affects layers like dropout and batch normalization.
        """
        ...

    @abstractmethod
    def eval_mode(self) -> None:
        """Switch the model to evaluation (inference) mode.

        Disables dropout and uses running statistics for batch norm.
        """
        ...


class LossProtocol(ABC):
    """A loss function: :math:`\\mathcal{L}(\\hat{y}, y) \\to \\mathbb{R}`.

    Callable interface: ``loss_fn(predictions, targets)`` returns a
    scalar float. Concrete implementations provide the specific loss
    formula (MSE, cross-entropy, etc.).

    Usage::

        loss_fn = MSELoss()
        value = loss_fn(model.forward(x), y)
    """

    @abstractmethod
    def forward(self, predictions: ArrayLike, targets: ArrayLike) -> Loss:
        """Compute the loss.

        Args:
            predictions: Model output of shape ``(batch_size, *dims)``.
            targets: Ground truth of shape ``(batch_size, *dims)``.

        Returns:
            Loss value object.
        """
        ...

    def __call__(self, predictions: ArrayLike, targets: ArrayLike) -> Loss:
        """Delegate to :meth:`forward`. The standard calling convention."""
        return self.forward(predictions, targets)


class OptimizerProtocol(ABC):
    """Updates model parameters using accumulated gradients.

    After each backward pass, gradients accumulate in each
    :class:`Parameter.grad`. The optimizer consumes those gradients
    to update :attr:`Parameter.data`.

    Usage::

        optimizer.zero_grad()
        loss = loss_fn(model.forward(x), y)
        loss.backward()   # framework-specific gradient computation
        optimizer.step()
    """

    @abstractmethod
    def step(self) -> None:
        """Update all parameters using their accumulated gradients.

        Called once per batch (or per gradient accumulation step).
        """
        ...

    @abstractmethod
    def zero_grad(self) -> None:
        """Reset all parameter gradients to zero.

        Must be called before each backward pass to prevent gradient
        accumulation across batches.
        """
        ...
