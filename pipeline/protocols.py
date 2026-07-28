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


@dataclass
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
