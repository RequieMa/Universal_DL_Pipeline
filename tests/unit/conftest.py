"""Shared test doubles (fakes) and fixtures for unit tests.

All test files under tests/unit/ can import these via conftest — pytest
makes the names available as module-level imports.

Fakes defined here:
    - :class:`FakeDataStream` — minimal DataStream implementation
    - :class:`FakeModel` — minimal ModelProtocol (linear: x -> 2*x)
    - :class:`FakeModelWithParams` — linear model with explicit, mutable
      parameters for TrainLoop gradient-update testing
    - :class:`FakeLoss` — MSE loss via LossProtocol
    - :class:`FakeOptimizer` — counter-based OptimizerProtocol

Fixtures (pytest fixtures):
    - :func:`tiny_titanic_path` — path to the tiny_titanic.csv fixture
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pipeline.protocols import (
    ArrayLike,
    Batch,
    DataStream,
    LossProtocol,
    ModelProtocol,
    OptimizerProtocol,
    Parameter,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# ── FakeDataStream ──────────────────────────────────────────────────────


class FakeDataStream(DataStream):
    """Minimal DataStream implementation for testing.

    Wraps a pre-built list of :class:`Batch` objects. Useful when a
    test needs to control exactly what data the pipeline receives.

    Parameters:
        batches: The batches to yield when iterated.
    """

    def __init__(self, batches: list[Batch]) -> None:
        self._batches = batches

    def __iter__(self) -> Iterator[Batch]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


# ── FakeModel ───────────────────────────────────────────────────────────


class FakeModel(ModelProtocol):
    """Minimal ModelProtocol implementation for testing.

    ``forward(x)`` returns ``x * 2``. Has one trainable parameter
    (``w = [1.0]``). Tracks train/eval mode via ``_mode`` attribute.

    This is the simplest possible fake — prefer it when the test does not
    inspect or mutate parameters.
    """

    def __init__(self) -> None:
        self._params = [Parameter(data=np.array([1.0]), name="w")]
        self._mode = "train"

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        return np.asarray(inputs) * 2.0

    def parameters(self) -> Iterator[Parameter]:
        return iter(self._params)

    def train_mode(self) -> None:
        self._mode = "train"

    def eval_mode(self) -> None:
        self._mode = "eval"


# ── FakeModelWithParams ─────────────────────────────────────────────────


class FakeModelWithParams(ModelProtocol):
    """Linear model ``y = W @ x + b`` with explicit, mutable parameters.

    Designed for TrainLoop gradient-update tests: the optimizer mutates
    ``param.data`` and ``param.grad`` in-place, and the forward pass
    reads those values, so a test can verify that parameters changed after
    ``optimizer.step()``.

    Parameters:
        in_features: Number of input features.
        out_features: Number of output targets.
        seed: Random seed for reproducible weight/bias initialisation.
            Defaults to 42.
    """

    def __init__(self, in_features: int = 4, out_features: int = 2, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self._W = Parameter(
            data=rng.standard_normal((out_features, in_features)).astype(np.float64),
            grad=np.zeros((out_features, in_features), dtype=np.float64),
            name="weight",
        )
        self._b = Parameter(
            data=rng.standard_normal(out_features).astype(np.float64),
            grad=np.zeros(out_features, dtype=np.float64),
            name="bias",
        )
        self._mode = "train"

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        """Compute ``W @ x + b`` (rows = samples, columns = features)."""
        x = np.asarray(inputs, dtype=np.float64)
        return x @ self._W.data.T + self._b.data

    def parameters(self) -> Iterable[Parameter]:
        """Yield the weight and bias Parameters (mutable references)."""
        yield self._W
        yield self._b

    def train_mode(self) -> None:
        self._mode = "train"

    def eval_mode(self) -> None:
        self._mode = "eval"


# ── FakeLoss ────────────────────────────────────────────────────────────


class FakeLoss(LossProtocol):
    """Minimal LossProtocol implementation — returns MSE as a plain float.

    NOTE: Returns a ``float``, not a :class:`Loss` object, for backward
    compatibility with Phase-1 tests. TrainLoop tests that need
    ``loss.backward()`` should use a concrete loss that returns a
    :class:`Loss` instance.
    """

    def forward(self, predictions: ArrayLike, targets: ArrayLike) -> float:
        preds = np.asarray(predictions)
        targs = np.asarray(targets)
        return float(np.mean((preds - targs) ** 2))


# ── FakeOptimizer ───────────────────────────────────────────────────────


class FakeOptimizer(OptimizerProtocol):
    """Minimal OptimizerProtocol implementation for testing.

    Counts calls to ``step()`` and ``zero_grad()``. Does *not* actually
    update parameters — use a concrete subclass or mock when you need
    parameter-mutation verification.
    """

    def __init__(self) -> None:
        self.step_count = 0
        self.zero_count = 0
        self._params: list[Parameter] = []

    def step(self) -> None:
        self.step_count += 1

    def zero_grad(self) -> None:
        self.zero_count += 1


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_titanic_path() -> Path:
    """Return the absolute path to ``tiny_titanic.csv``.

    The CSV contains 10 rows of Titanic passenger data with columns:
    PassengerId, Survived, Pclass, Sex, Age, Fare.  Suitable for
    DataSource tests that need a small real-world-ish dataset.

    Returns:
        Path to the fixture CSV file.
    """
    return Path(__file__).resolve().parent.parent / "fixtures" / "tiny_titanic.csv"
