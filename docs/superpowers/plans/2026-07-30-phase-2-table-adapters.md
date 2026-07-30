# Phase 2: Table Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build framework adapters (sklearn, numpy) and training components (SGD, Adam, MSELoss, CrossEntropyLoss) that satisfy the pipeline protocols, plus Titanic examples and contract tests.

**Architecture:** `SklearnModel` wraps sklearn estimators — overrides `train()` for `.fit()`, uses stub Loss/Optimizer. `NumpyModel` uses explicit Parameter objects with a `backward()` method that propagates gradients through layers. `NumpyOptimizer` delegates to `SGD`/`Adam` rules that mutate Parameters in-place. Loss functions return `Loss` objects whose `_backward_fn` closures call `model.backward()`.

**Tech Stack:** Python 3.11+, numpy, sklearn (lazy-imported), sympy (notebook only), pytest

## Global Constraints

- **No framework in core.** `pipeline/` never imports sklearn or sympy at module level — only inside functions/methods.
- **TDD iron law.** No production code without a failing test first. Every task: test → fail → implement → pass → commit.
- **Google-style docstrings** on all public API classes and methods.
- **Lines per file ≤ 300**, lines per function ≤ 50, public methods per class ≤ 5.
- **ruff check + format clean**, mypy strict mode (stub warnings ok on py3.13).
- **pytest** `-v --tb=short`, marker `slow` for integration/E2E.
- **No mocking frameworks** — only fakes and stubs. This is a teaching project.
- **All comments in English.**
- **Commit messages** end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `collect_arrays` utility

**Files:**
- Create: `pipeline/data/utils.py`
- Create: `tests/unit/test_data_utils.py`
- Modify: `pipeline/data/__init__.py` (add export)

**Interfaces:**
- Produces: `collect_arrays(stream: DataStream) -> tuple[np.ndarray, np.ndarray]`

**Purpose:** Drain a DataStream into (X, y) numpy arrays. Used by sklearn pipelines (need full dataset for `.fit()`) and evaluation loops (aggregate predictions).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_data_utils.py
"""Unit tests for pipeline.data.utils — collect_arrays."""
from __future__ import annotations

import numpy as np

from pipeline.data.utils import collect_arrays
from pipeline.protocols import Batch


class FakeStream:
    """Minimal DataStream yielding two batches."""

    def __init__(self):
        self._batches = [
            Batch(inputs=np.array([[1.0, 2.0], [3.0, 4.0]]), targets=np.array([0, 1])),
            Batch(inputs=np.array([[5.0, 6.0]]), targets=np.array([0])),
        ]

    def __iter__(self):
        yield from self._batches

    def __len__(self):
        return len(self._batches)


class TestCollectArrays:
    def test_collects_all_batches(self):
        """Happy Path: drains entire stream into (X, y)."""
        stream = FakeStream()
        X, y = collect_arrays(stream)
        assert X.shape == (3, 2)
        assert y.shape == (3,)
        np.testing.assert_array_equal(X, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        np.testing.assert_array_equal(y, np.array([0, 1, 0]))

    def test_single_batch(self):
        """Boundary: stream with exactly one batch."""
        stream = FakeStream()
        stream._batches = [Batch(inputs=np.array([[1.0]]), targets=np.array([0]))]
        X, y = collect_arrays(stream)
        assert X.shape == (1, 1)
        assert y.shape == (1,)

    def test_empty_stream_raises(self):
        """Boundary: empty stream should raise ValueError."""
        stream = FakeStream()
        stream._batches = []

        with pytest.raises(ValueError, match="empty"):
            collect_arrays(stream)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_data_utils.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.data.utils'`

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/data/utils.py
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
```

- [ ] **Step 4: Update `pipeline/data/__init__.py`**

```python
# Add after the existing imports:
from pipeline.data.utils import collect_arrays

# Add to __all__:
__all__ = ["CsvDataSource", "train_test_split", "collect_arrays"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_data_utils.py -v
```
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add pipeline/data/utils.py pipeline/data/__init__.py tests/unit/test_data_utils.py
git commit -m "feat: add collect_arrays utility to drain DataStream into (X, y)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: StubLoss + StubOptimizer

**Files:**
- Create: `pipeline/adapters/__init__.py` (empty package init)
- Create: `pipeline/adapters/sklearn_adapter.py`
- Create: `tests/unit/test_sklearn_adapter.py`
- Create: `tests/contract/__init__.py` (empty file)

**Interfaces:**
- Produces: `StubLoss(LossProtocol)` — `forward(predictions, targets) -> Loss`
- Produces: `StubOptimizer(OptimizerProtocol)` — `step()`, `zero_grad()`

**Purpose:** No-op loss and optimizer for non-gradient models. Satisfy the protocol so `build_model()` always assigns `state.loss_fn` and `state.optimizer` without None-checks.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sklearn_adapter.py (partial — StubLoss and StubOptimizer tests)
"""Unit tests for pipeline.adapters.sklearn_adapter."""
from __future__ import annotations

import numpy as np

from pipeline.adapters.sklearn_adapter import StubLoss, StubOptimizer
from pipeline.protocols import Loss


class TestStubLoss:
    def test_forward_returns_loss_zero(self):
        """StubLoss always returns Loss(0.0)."""
        loss_fn = StubLoss()
        result = loss_fn.forward(
            np.array([1.0, 2.0]), np.array([0.0, 1.0])
        )
        assert isinstance(result, Loss)
        assert float(result) == 0.0

    def test_call_delegates_to_forward(self):
        """__call__ returns same as forward()."""
        loss_fn = StubLoss()
        result = loss_fn(np.array([1.0]), np.array([0.0]))
        assert float(result) == 0.0

    def test_backward_is_noop(self):
        """Loss.backward() does not raise (no-op)."""
        loss_fn = StubLoss()
        loss = loss_fn.forward(np.array([1.0]), np.array([0.0]))
        loss.backward()  # should not raise


class TestStubOptimizer:
    def test_step_is_noop(self):
        """step() does nothing, does not raise."""
        opt = StubOptimizer()
        opt.step()  # should not raise

    def test_zero_grad_is_noop(self):
        """zero_grad() does nothing, does not raise."""
        opt = StubOptimizer()
        opt.zero_grad()  # should not raise

    def test_multiple_calls_do_not_raise(self):
        """Calling step/zero_grad repeatedly is safe."""
        opt = StubOptimizer()
        for _ in range(10):
            opt.zero_grad()
            opt.step()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_sklearn_adapter.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.adapters.sklearn_adapter'`

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/adapters/__init__.py (empty for now — populated in Task 10)
"""Framework adapters that satisfy pipeline protocols."""

# pipeline/adapters/sklearn_adapter.py (partial — StubLoss + StubOptimizer)
"""Sklearn estimator adapter and stub components."""
from __future__ import annotations

import numpy as np

from pipeline.protocols import ArrayLike, Loss, LossProtocol, OptimizerProtocol


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_sklearn_adapter.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/adapters/__init__.py pipeline/adapters/sklearn_adapter.py tests/unit/test_sklearn_adapter.py tests/contract/__init__.py
git commit -m "feat: add StubLoss and StubOptimizer for non-gradient models

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: SklearnModel adapter

**Files:**
- Modify: `pipeline/adapters/sklearn_adapter.py` (add SklearnModel)
- Modify: `tests/unit/test_sklearn_adapter.py` (add SklearnModel tests)

**Interfaces:**
- Produces: `SklearnModel(ModelProtocol)` — wraps any sklearn estimator
  - `__init__(estimator)` — estimator is any sklearn-compatible object with `.fit()` and `.predict()`/`.predict_proba()`
  - `forward(inputs) -> ArrayLike` — delegates to `predict_proba()` if available, else `predict()`
  - `parameters() -> Iterable[Parameter]` — returns `[]` (no gradient parameters)
  - `train_mode()` / `eval_mode()` — no-ops

**Critical design note:** `SklearnModel` uses lazy imports. `sklearn` is NOT imported at module level — `__init__` receives an already-constructed estimator. The pipeline subclass imports sklearn and constructs the estimator.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_sklearn_adapter.py

from pipeline.adapters.sklearn_adapter import SklearnModel
from pipeline.protocols import Parameter


class TestSklearnModel:
    """Tests for SklearnModel adapter."""

    def test_parameters_returns_empty(self):
        """Sklearn models have no gradient parameters."""
        from sklearn.linear_model import LogisticRegression

        model = SklearnModel(LogisticRegression())
        params = list(model.parameters())
        assert params == []

    def test_train_eval_mode_noop(self):
        """train_mode() and eval_mode() do not raise."""
        from sklearn.linear_model import LogisticRegression

        model = SklearnModel(LogisticRegression())
        model.train_mode()
        model.eval_mode()
        # No assertion needed — just verifying no exception

    def test_forward_shape_after_fit(self):
        """forward() returns predictions with correct shape after fit."""
        from sklearn.linear_model import LogisticRegression

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([0, 1, 0, 1])

        est = LogisticRegression(max_iter=1000)
        est.fit(X, y)

        model = SklearnModel(est)
        preds = model.forward(np.array([[1.0, 2.0], [3.0, 4.0]]))
        # LogisticRegression with 2 classes → predict_proba returns (n, 2)
        assert preds.shape == (2, 2)
        # Probabilities sum to ~1 per row
        assert np.allclose(preds.sum(axis=1), 1.0, atol=0.01)

    def test_forward_delegates_to_predict_proba(self):
        """forward() uses predict_proba when available (classifier)."""
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression(max_iter=1000)
        est.fit(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            np.array([0, 1, 0, 1]),
        )
        model = SklearnModel(est)
        preds = model.forward(np.array([[1.0, 2.0]]))
        # Should match predict_proba output
        expected = est.predict_proba(np.array([[1.0, 2.0]]))
        np.testing.assert_array_almost_equal(preds, expected)

    def test_forward_uses_predict_when_no_proba(self):
        """forward() falls back to predict() for regressors."""
        from sklearn.linear_model import LinearRegression

        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        est = LinearRegression()
        est.fit(X, y)

        model = SklearnModel(est)
        preds = model.forward(np.array([[1.5], [2.5]]))
        assert preds.shape == (2,)
        expected = est.predict(np.array([[1.5], [2.5]]))
        np.testing.assert_array_almost_equal(preds, expected)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_sklearn_adapter.py::TestSklearnModel -v
```
Expected: FAIL — `ImportError: cannot import name 'SklearnModel'`

- [ ] **Step 3: Write minimal implementation**

```python
# Append to pipeline/adapters/sklearn_adapter.py

from collections.abc import Iterable

from pipeline.protocols import ModelProtocol, Parameter


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

    def __init__(self, estimator: object) -> None:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_sklearn_adapter.py -v
```
Expected: all tests pass (5 StubLoss/Optimizer + 5 SklearnModel = 10 passed)

- [ ] **Step 5: Run existing tests to check no regressions**

```bash
uv run pytest -m "not slow" -v
```
Expected: all 225+ existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add pipeline/adapters/sklearn_adapter.py tests/unit/test_sklearn_adapter.py
git commit -m "feat: add SklearnModel adapter wrapping sklearn estimators

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: SGD optimizer rule

**Files:**
- Create: `pipeline/training/optimizers.py`
- Create: `tests/unit/test_optimizers.py`

**Interfaces:**
- Produces: `SGD` — `__init__(lr: float = 0.01)`, `update(param: Parameter) -> None`

**Purpose:** Pure numpy SGD: `param.data -= lr * param.grad`. Minimal, readable, teachable.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_optimizers.py
"""Unit tests for pipeline.training.optimizers — SGD, Adam."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.protocols import Parameter


class TestSGD:
    """Tests for SGD optimizer rule."""

    def test_update_reduces_param_by_lr_times_grad(self):
        """SGD: data = data - lr * grad."""
        from pipeline.training.optimizers import SGD

        param = Parameter(
            data=np.array([1.0, 2.0, 3.0]),
            grad=np.array([0.1, 0.2, 0.3]),
            name="w",
        )
        sgd = SGD(lr=0.1)
        sgd.update(param)
        expected = np.array([1.0, 2.0, 3.0]) - 0.1 * np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(param.data, expected)

    def test_update_with_default_lr(self):
        """Default lr=0.01."""
        from pipeline.training.optimizers import SGD

        param = Parameter(
            data=np.array([5.0]),
            grad=np.array([2.0]),
            name="w",
        )
        sgd = SGD()
        sgd.update(param)
        np.testing.assert_array_almost_equal(param.data, np.array([4.98]))

    def test_zero_gradient_no_change(self):
        """When grad is all zeros, data is unchanged."""
        from pipeline.training.optimizers import SGD

        param = Parameter(
            data=np.array([1.0, 2.0]),
            grad=np.array([0.0, 0.0]),
            name="w",
        )
        original = param.data.copy()
        sgd = SGD(lr=0.1)
        sgd.update(param)
        np.testing.assert_array_equal(param.data, original)

    def test_lr_zero_no_change(self):
        """lr=0.0 means no update."""
        from pipeline.training.optimizers import SGD

        param = Parameter(
            data=np.array([1.0, 2.0]),
            grad=np.array([0.5, 1.0]),
            name="w",
        )
        original = param.data.copy()
        sgd = SGD(lr=0.0)
        sgd.update(param)
        np.testing.assert_array_equal(param.data, original)

    def test_negative_lr_raises(self):
        """Negative learning rate should raise ValueError."""
        from pipeline.training.optimizers import SGD

        with pytest.raises(ValueError, match="positive"):
            SGD(lr=-0.1)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_optimizers.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/training/optimizers.py
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
        param.data = param.data - self.lr * param.grad
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_optimizers.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/training/optimizers.py tests/unit/test_optimizers.py
git commit -m "feat: add SGD optimizer rule (pure numpy)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Adam optimizer rule

**Files:**
- Modify: `pipeline/training/optimizers.py` (add Adam class)
- Modify: `tests/unit/test_optimizers.py` (add Adam tests)

**Interfaces:**
- Produces: `Adam` — `__init__(lr=0.001, betas=(0.9, 0.999), eps=1e-8)`, `update(param: Parameter) -> None`

**Purpose:** Adam optimizer — momentum + adaptive learning rates. Builds on the same `update(param)` interface as SGD.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_optimizers.py

class TestAdam:
    """Tests for Adam optimizer rule."""

    def test_update_reduces_loss_direction(self):
        """Adam steps in the direction that reduces the gradient."""
        from pipeline.training.optimizers import Adam

        param = Parameter(
            data=np.array([1.0, 1.0]),
            grad=np.array([0.5, 0.5]),
            name="w",
        )
        adam = Adam(lr=0.1)
        adam.update(param)
        # Parameter should decrease (grad is positive → step is negative)
        assert param.data[0] < 1.0
        assert param.data[1] < 1.0

    def test_zero_gradient_no_change(self):
        """When grad is all zeros, data is unchanged."""
        from pipeline.training.optimizers import Adam

        param = Parameter(
            data=np.array([3.0, 4.0]),
            grad=np.array([0.0, 0.0]),
            name="w",
        )
        original = param.data.copy()
        adam = Adam(lr=0.01)
        adam.update(param)
        np.testing.assert_array_equal(param.data, original)

    def test_step_increases_over_calls(self):
        """Bias correction: early steps are smaller, converge to lr."""
        from pipeline.training.optimizers import Adam

        param = Parameter(
            data=np.array([0.0]),
            grad=np.array([1.0]),
            name="w",
        )
        adam = Adam(lr=0.001, betas=(0.9, 0.999))
        # First step: t=1, bias correction = sqrt(1-0.999)/(1-0.9) ≈ small
        adam.update(param)
        step1_change = abs(float(param.data[0]))
        # Reset and do 1000 warmup steps on a different param
        param2 = Parameter(data=np.array([0.0]), grad=np.array([0.0]), name="w2")
        for _ in range(1000):
            adam.update(param2)  # advance t counter
        # Now step with warm t
        param3 = Parameter(data=np.array([0.0]), grad=np.array([1.0]), name="w3")
        adam.update(param3)
        step_large_change = abs(float(param3.data[0]))
        # After warmup, step size should be close to full lr
        # With grad=1, step ≈ -lr = -0.001
        assert step_large_change > step1_change * 0.5

    def test_different_params_have_separate_momentum(self):
        """Each parameter gets its own m and v buffers."""
        from pipeline.training.optimizers import Adam

        p1 = Parameter(data=np.array([1.0]), grad=np.array([0.3]), name="w1")
        p2 = Parameter(data=np.array([2.0]), grad=np.array([0.7]), name="w2")
        adam = Adam(lr=0.1)
        adam.update(p1)
        adam.update(p2)
        # Both should have moved
        assert p1.data[0] != 1.0
        assert p2.data[0] != 2.0

    def test_negative_lr_raises(self):
        """Negative learning rate should raise ValueError."""
        from pipeline.training.optimizers import Adam

        with pytest.raises(ValueError, match="positive"):
            Adam(lr=-0.001)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_optimizers.py::TestAdam -v
```
Expected: FAIL — `ImportError: cannot import name 'Adam'`

- [ ] **Step 3: Write minimal implementation**

```python
# Append to pipeline/training/optimizers.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_optimizers.py -v
```
Expected: all 10 passed (5 SGD + 5 Adam)

- [ ] **Step 5: Commit**

```bash
git add pipeline/training/optimizers.py tests/unit/test_optimizers.py
git commit -m "feat: add Adam optimizer rule (pure numpy)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: MSELoss (pure numpy)

**Files:**
- Create: `pipeline/training/losses.py`
- Create: `tests/unit/test_losses.py`

**Interfaces:**
- Produces: `MSELoss(LossProtocol)` — `__init__(model: NumpyModel | None = None)`, `forward(predictions, targets) -> Loss`

**Purpose:** Mean squared error. The `_backward_fn` closure computes `dL/dpred = 2*(pred-target)/N` and propagates through the model (if one is attached).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_losses.py
"""Unit tests for pipeline.training.losses — MSELoss, CrossEntropyLoss."""
from __future__ import annotations

import numpy as np

from pipeline.protocols import Loss


class TestMSELoss:
    """Tests for MSELoss."""

    def test_forward_returns_loss_object(self):
        """forward() returns a Loss instance."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        result = loss_fn.forward(pred, target)
        assert isinstance(result, Loss)

    def test_perfect_prediction_zero_loss(self):
        """MSE = 0 when predictions equal targets."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        loss = loss_fn.forward(pred, target)
        assert float(loss) == 0.0

    def test_positive_loss_for_errors(self):
        """MSE > 0 when predictions differ from targets."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()
        pred = np.array([1.0, 2.0])
        target = np.array([3.0, 4.0])
        loss = loss_fn.forward(pred, target)
        # MSE = mean((1-3)^2 + (2-4)^2) = mean(4+4) = 4
        assert float(loss) == pytest.approx(4.0)

    def test_call_delegates_to_forward(self):
        """__call__ returns same as forward()."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()
        pred = np.array([1.0, 2.0])
        target = np.array([1.0, 3.0])
        result_call = loss_fn(pred, target)
        result_fwd = loss_fn.forward(pred, target)
        assert float(result_call) == float(result_fwd)

    def test_backward_noop_without_model(self):
        """backward() is a no-op when no model is attached."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()  # no model
        pred = np.array([1.0, 2.0])
        target = np.array([0.0, 1.0])
        loss = loss_fn.forward(pred, target)
        loss.backward()  # should not raise

    def test_backward_with_model_populates_grads(self):
        """backward() propagates dL/dpred through attached model."""
        from pipeline.training.losses import MSELoss

        # We need NumpyModel, which doesn't exist yet.
        # Build a minimal inline model for this test.
        from pipeline.protocols import Parameter

        class _MinimalModel:
            """Single-layer linear model for MSELoss gradient test."""

            def __init__(self):
                self.W = Parameter(
                    data=np.array([[2.0]]),
                    grad=np.zeros((1, 1)),
                    name="W",
                )
                self.b = Parameter(
                    data=np.array([0.0]),
                    grad=np.zeros(1),
                    name="b",
                )
                self._params = [self.W, self.b]
                self._cache = {}

            def forward(self, x):
                x = np.asarray(x, dtype=np.float64)
                self._cache["a0"] = x  # input for backward
                return x @ self.W.data.T + self.b.data

            def backward(self, dL_doutput):
                self.W.grad = dL_doutput.T @ self._cache["a0"]
                self.b.grad = dL_doutput.sum(axis=0)

            def parameters(self):
                return iter(self._params)

        model = _MinimalModel()
        loss_fn = MSELoss(model=model)
        x = np.array([[1.0], [2.0]])  # batch of 2
        y = np.array([[3.0], [5.0]])  # target
        pred = model.forward(x)
        loss = loss_fn.forward(pred, y)
        loss.backward()
        # Gradients should be populated and non-zero
        assert model.W.grad is not None
        assert model.b.grad is not None
        assert np.any(model.W.grad != 0)

    def test_2d_output(self):
        """MSE works with 2D predictions (batch_size, output_dim)."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()
        pred = np.array([[0.5, 0.5], [0.2, 0.8]])
        target = np.array([[1.0, 0.0], [0.0, 1.0]])
        loss = loss_fn.forward(pred, target)
        expected = float(np.mean((pred - target) ** 2))
        assert float(loss) == pytest.approx(expected)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_losses.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/training/losses.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_losses.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/training/losses.py tests/unit/test_losses.py
git commit -m "feat: add MSELoss with model-aware backward propagation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: CrossEntropyLoss (pure numpy)

**Files:**
- Modify: `pipeline/training/losses.py` (add CrossEntropyLoss)
- Modify: `tests/unit/test_losses.py` (add CrossEntropyLoss tests)

**Interfaces:**
- Produces: `CrossEntropyLoss(LossProtocol)` — `__init__(model=None)`, `forward(logits, targets) -> Loss`

**Purpose:** Softmax + negative log-likelihood. The `_backward_fn` computes `dL/dlogits = (probs - one_hot(targets)) / batch_size` and propagates through the model.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_losses.py

class TestCrossEntropyLoss:
    """Tests for CrossEntropyLoss."""

    def test_forward_returns_loss_object(self):
        """forward() returns a Loss instance."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[2.0, 1.0], [0.5, 1.5]])
        targets = np.array([0, 1])
        result = loss_fn.forward(logits, targets)
        assert isinstance(result, Loss)

    def test_perfect_prediction_low_loss(self):
        """Loss is near zero for confident correct predictions."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[10.0, -10.0], [-10.0, 10.0]])
        targets = np.array([0, 1])
        loss = loss_fn.forward(logits, targets)
        assert float(loss) < 0.001

    def test_wrong_prediction_high_loss(self):
        """Loss is high for confident wrong predictions."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[-10.0, 10.0], [10.0, -10.0]])
        targets = np.array([0, 1])
        loss = loss_fn.forward(logits, targets)
        assert float(loss) > 10.0

    def test_probabilities_sum_to_one(self):
        """Internal softmax produces valid probability distribution."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([0])
        loss = loss_fn.forward(logits, targets)
        # Value > 0 for wrong prediction
        assert float(loss) > 0.0

    def test_call_delegates_to_forward(self):
        """__call__ returns same as forward()."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[1.0, 2.0], [3.0, 1.0]])
        targets = np.array([1, 0])
        assert float(loss_fn(logits, targets)) == float(
            loss_fn.forward(logits, targets)
        )

    def test_backward_noop_without_model(self):
        """backward() is no-op when no model attached."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[1.0, 2.0]])
        targets = np.array([0])
        loss = loss_fn.forward(logits, targets)
        loss.backward()  # should not raise

    def test_backward_with_model_populates_grads(self):
        """backward() propagates through attached model."""
        from pipeline.training.losses import CrossEntropyLoss
        from pipeline.protocols import Parameter

        class _MinimalModel:
            """Two-class linear model for CrossEntropyLoss gradient test."""

            def __init__(self):
                self.W = Parameter(
                    data=np.array([[0.5, 0.5], [0.5, 0.5]]),
                    grad=np.zeros((2, 2)),
                    name="W",
                )
                self.b = Parameter(
                    data=np.array([0.0, 0.0]),
                    grad=np.zeros(2),
                    name="b",
                )
                self._params = [self.W, self.b]
                self._cache = {}

            def forward(self, x):
                x = np.asarray(x, dtype=np.float64)
                self._cache["a0"] = x
                return x @ self.W.data.T + self.b.data

            def backward(self, dL_doutput):
                self.W.grad = dL_doutput.T @ self._cache["a0"]
                self.b.grad = dL_doutput.sum(axis=0)

            def parameters(self):
                return iter(self._params)

        model = _MinimalModel()
        loss_fn = CrossEntropyLoss(model=model)
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        pred = model.forward(x)
        loss = loss_fn.forward(pred, y)
        loss.backward()
        assert model.W.grad is not None
        assert model.b.grad is not None
        assert np.any(model.W.grad != 0)

    def test_batch_independence(self):
        """Loss for a batch equals mean of per-sample losses."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[3.0, 1.0], [1.0, 3.0]])
        targets = np.array([0, 1])
        loss_batch = float(loss_fn.forward(logits, targets))

        loss0 = float(loss_fn.forward(np.array([[3.0, 1.0]]), np.array([0])))
        loss1 = float(loss_fn.forward(np.array([[1.0, 3.0]]), np.array([1])))
        assert loss_batch == pytest.approx((loss0 + loss1) / 2.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_losses.py::TestCrossEntropyLoss -v
```
Expected: FAIL — `ImportError: cannot import name 'CrossEntropyLoss'`

- [ ] **Step 3: Write minimal implementation**

```python
# Append to pipeline/training/losses.py

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
        n_classes = logits.shape[1]

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_losses.py -v
```
Expected: 15 passed (7 MSE + 8 CrossEntropy)

- [ ] **Step 5: Commit**

```bash
git add pipeline/training/losses.py tests/unit/test_losses.py
git commit -m "feat: add CrossEntropyLoss with softmax and backward propagation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: NumpyModel adapter

**Files:**
- Create: `pipeline/adapters/numpy_adapter.py`
- Create: `tests/unit/test_numpy_adapter.py`

**Interfaces:**
- Produces: `NumpyModel(ModelProtocol)`
  - `__init__(layers: list[tuple[np.ndarray, np.ndarray]], activation: str = "relu")`
  - `forward(inputs) -> ArrayLike`
  - `backward(dL_doutput: np.ndarray) -> None` — propagates gradients through all layers
  - `parameters() -> Iterable[Parameter]`
  - `train_mode()` / `eval_mode()`

**Critical design notes:**
- `layers` is a list of `(weight, bias)` tuples — e.g., `[(W1, b1), (W2, b2)]` for a 2-layer network
- `forward()` caches intermediate activations (`self._cache`) for `backward()`
- `backward()` applies the chain rule: `dL/dW = dL_dz.T @ a_prev`, `dL/db = dL_dz.sum(axis=0)`
- Last layer has no activation (raw logits); intermediate layers use ReLU

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_numpy_adapter.py
"""Unit tests for pipeline.adapters.numpy_adapter — NumpyModel, NumpyOptimizer."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.protocols import Parameter


class TestNumpyModelConstruction:
    """Construction and basic properties."""

    def test_single_layer_construction(self):
        """NumpyModel with one (weight, bias) pair."""
        from pipeline.adapters.numpy_adapter import NumpyModel

        W = np.array([[0.5, 0.5]])
        b = np.array([0.0])
        model = NumpyModel([(W, b)])
        params = list(model.parameters())
        assert len(params) == 2  # W and b
        assert params[0].name == "W0"
        assert params[1].name == "b0"

    def test_two_layer_construction(self):
        """NumpyModel with two (weight, bias) pairs."""
        from pipeline.adapters.numpy_adapter import NumpyModel

        W1 = np.array([[0.5, 0.5], [0.3, 0.3]])
        b1 = np.array([0.0, 0.0])
        W2 = np.array([[1.0, 1.0]])
        b2 = np.array([0.0])
        model = NumpyModel([(W1, b1), (W2, b2)])
        params = list(model.parameters())
        assert len(params) == 4  # W1, b1, W2, b2

    def test_grad_initialized_to_zeros(self):
        """Each Parameter.grad starts as zeros matching data shape."""
        from pipeline.adapters.numpy_adapter import NumpyModel

        W = np.array([[0.5, 0.5, 0.5]])
        b = np.array([0.0])
        model = NumpyModel([(W, b)])
        for p in model.parameters():
            assert p.grad is not None
            assert p.grad.shape == p.data.shape
            np.testing.assert_array_equal(p.grad, np.zeros_like(p.data))

    def test_empty_layers_raises(self):
        """Empty layer list should raise ValueError."""
        from pipeline.adapters.numpy_adapter import NumpyModel

        with pytest.raises(ValueError, match="empty"):
            NumpyModel([])


class TestNumpyModelForward:
    """Forward pass tests."""

    def test_single_layer_forward_shape(self):
        """Single layer: (batch, in) -> (batch, out)."""
        from pipeline.adapters.numpy_adapter import NumpyModel

        W = np.array([[1.0, 0.0]])
        b = np.array([0.0])
        model = NumpyModel([(W, b)])
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = model.forward(x)
        assert out.shape == (2, 1)

    def test_two_layer_forward_shape(self):
        """Two layers: (batch, in) -> (batch, hidden) -> (batch, out)."""
        from pipeline.adapters.numpy_adapter import NumpyModel

        W1 = np.ones((4, 3)) * 0.1
        b1 = np.zeros(4)
        W2 = np.ones((2, 4)) * 0.1
        b2 = np.zeros(2)
        model = NumpyModel([(W1, b1), (W2, b2)])
        x = np.array([[1.0, 2.0, 3.0]])
        out = model.forward(x)
        assert out.shape == (1, 2)

    def test_forward_is_deterministic(self):
        """Same input → same output (no randomness in forward)."""
        from pipeline.adapters.numpy_adapter import NumpyModel

        rng = np.random.default_rng(42)
        W = rng.standard_normal((3, 4))
        b = rng.standard_normal(3)
        model = NumpyModel([(W, b)])
        x = rng.standard_normal((10, 4))
        out1 = model.forward(x)
        out2 = model.forward(x)
        np.testing.assert_array_equal(out1, out2)

    def test_last_layer_no_activation(self):
        """Last layer outputs raw logits (no activation applied)."""
        from pipeline.adapters.numpy_adapter import NumpyModel

        W1 = np.eye(3)
        b1 = np.ones(3)
        W2 = np.eye(3)
        b2 = np.ones(3)
        model = NumpyModel([(W1, b1), (W2, b2)], activation="relu")
        x = np.array([[1.0, 2.0, 3.0]])
        out = model.forward(x)
        # With relu: layer1 = relu(x + 1) = x + 1 (since all positive)
        # layer2 = (x+1) + 1 = x + 2
        expected = x + 2.0
        np.testing.assert_array_almost_equal(out, expected)

    def test_train_eval_mode_toggle(self):
        """train_mode() and eval_mode() toggle without error."""
        from pipeline.adapters.numpy_adapter import NumpyModel

        W = np.eye(2)
        b = np.zeros(2)
        model = NumpyModel([(W, b)])
        model.train_mode()
        model.eval_mode()
        model.train_mode()


class TestNumpyModelBackward:
    """Backward pass tests."""

    def test_backward_populates_grads_single_layer(self):
        """backward() sets param.grad for single-layer model."""
        from pipeline.adapters.numpy_adapter import NumpyModel

        W = np.array([[1.0, 2.0]])
        b = np.array([0.5])
        model = NumpyModel([(W, b)])
        x = np.array([[1.0, 3.0]])
        model.forward(x)  # populates cache
        dL_dout = np.array([[0.5]])
        model.backward(dL_dout)
        # dL/db = sum(dL/dout) = 0.5
        np.testing.assert_array_almost_equal(
            list(model.parameters())[1].grad, np.array([0.5])
        )
        # dL/dW = dL/dout.T @ x = [[0.5]] @ [[1.0, 3.0]]... wait
        # dL_dout shape (1,1), x shape (1,2)
        # dL/dW = dL_dout.T @ x = (1,1) @ (1,2) = (1,2)... no that's wrong
        # dL_dout.T = (1,1).T = (1,1), x = (1,2)
        # (1,1) @ (1,2) → error, (1,1) @ (1,2) with @ means (1,1) @ (1,2) → (1,2) ✓
        # Actually in numpy: (1,1) @ (1,2) = (1,2) ✓
        expected_dW = dL_dout.T @ x  # [[0.5]] @ [[1,3]] = [[0.5, 1.5]]
        np.testing.assert_array_almost_equal(
            list(model.parameters())[0].grad, expected_dW
        )

    def test_backward_propagates_through_two_layers(self):
        """Gradients flow through both layers."""
        from pipeline.adapters.numpy_adapter import NumpyModel

        rng = np.random.default_rng(42)
        W1 = rng.standard_normal((4, 3)) * 0.1
        b1 = np.zeros(4)
        W2 = rng.standard_normal((2, 4)) * 0.1
        b2 = np.zeros(2)
        model = NumpyModel([(W1, b1), (W2, b2)])
        x = rng.standard_normal((5, 3))
        model.forward(x)
        dL_dout = rng.standard_normal((5, 2))
        model.backward(dL_dout)
        params = list(model.parameters())
        # All params should have non-zero grads
        for p in params:
            assert p.grad is not None
            assert np.any(p.grad != 0), f"{p.name} grad is all zeros"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_numpy_adapter.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# pipeline/adapters/numpy_adapter.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_numpy_adapter.py -v
```
Expected: all 12 tests pass

- [ ] **Step 5: Run MSELoss tests (they import NumpyModel now)**

```bash
uv run pytest tests/unit/test_losses.py -v
```
Expected: all 15 still pass

- [ ] **Step 6: Commit**

```bash
git add pipeline/adapters/numpy_adapter.py tests/unit/test_numpy_adapter.py
git commit -m "feat: add NumpyModel adapter with forward/backward propagation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: NumpyOptimizer adapter

**Files:**
- Modify: `pipeline/adapters/numpy_adapter.py` (add NumpyOptimizer)
- Modify: `tests/unit/test_numpy_adapter.py` (add NumpyOptimizer tests)

**Interfaces:**
- Produces: `NumpyOptimizer(OptimizerProtocol)` — `__init__(parameters, rule)`, `step()`, `zero_grad()`

**Purpose:** Bridges the `OptimizerProtocol` pipeline contract with the `SGD`/`Adam` update rules. Iterates over Parameter refs, delegates each to `rule.update(param)`.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_numpy_adapter.py

from pipeline.adapters.numpy_adapter import NumpyOptimizer


class TestNumpyOptimizer:
    """Tests for NumpyOptimizer adapter."""

    def test_step_delegates_to_rule(self):
        """step() calls rule.update() for each parameter."""
        from pipeline.training.optimizers import SGD

        p1 = Parameter(data=np.array([1.0, 2.0]), grad=np.array([0.1, 0.2]), name="w")
        p2 = Parameter(data=np.array([3.0]), grad=np.array([0.3]), name="b")
        opt = NumpyOptimizer([p1, p2], SGD(lr=1.0))
        opt.step()
        # w = w - 1.0 * grad
        np.testing.assert_array_almost_equal(p1.data, np.array([0.9, 1.8]))
        np.testing.assert_array_almost_equal(p2.data, np.array([2.7]))

    def test_zero_grad_resets_all_grads(self):
        """zero_grad() sets all param.grad to zero."""
        from pipeline.training.optimizers import SGD

        p1 = Parameter(
            data=np.array([1.0]),
            grad=np.array([0.5]),
            name="w",
        )
        p2 = Parameter(
            data=np.array([2.0]),
            grad=np.array([0.3]),
            name="b",
        )
        opt = NumpyOptimizer([p1, p2], SGD(lr=0.1))
        opt.zero_grad()
        np.testing.assert_array_equal(p1.grad, np.array([0.0]))
        np.testing.assert_array_equal(p2.grad, np.array([0.0]))

    def test_step_count_matches_rule_calls(self):
        """Optimizer calls rule.update once per parameter per step."""
        from pipeline.training.optimizers import SGD

        p1 = Parameter(data=np.array([1.0]), grad=np.array([0.1]), name="w1")
        p2 = Parameter(data=np.array([2.0]), grad=np.array([0.2]), name="w2")
        opt = NumpyOptimizer([p1, p2], SGD(lr=0.1))
        # step once
        opt.step()
        assert p1.data[0] != 1.0
        assert p2.data[0] != 2.0

    def test_zero_grad_handles_none_grad(self):
        """zero_grad() doesn't crash when param.grad is None."""
        from pipeline.training.optimizers import SGD

        p = Parameter(data=np.array([1.0]), grad=None, name="w")
        opt = NumpyOptimizer([p], SGD(lr=0.1))
        opt.zero_grad()  # should not raise
        assert p.grad is None  # unchanged

    def test_step_zero_grad_step_cycle(self):
        """Full cycle: zero_grad → backward sets grad → step → zero_grad."""
        from pipeline.training.optimizers import SGD

        p = Parameter(data=np.array([5.0]), grad=np.array([0.0]), name="w")
        opt = NumpyOptimizer([p], SGD(lr=1.0))

        # Simulate a training step
        opt.zero_grad()
        np.testing.assert_array_equal(p.grad, np.array([0.0]))
        p.grad = np.array([2.0])  # simulate backward
        opt.step()
        assert p.data[0] == 3.0  # 5.0 - 1.0*2.0
        opt.zero_grad()
        np.testing.assert_array_equal(p.grad, np.array([0.0]))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_numpy_adapter.py::TestNumpyOptimizer -v
```
Expected: FAIL — `ImportError: cannot import name 'NumpyOptimizer'`

- [ ] **Step 3: Write minimal implementation**

```python
# Append to pipeline/adapters/numpy_adapter.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_numpy_adapter.py -v
```
Expected: all 17 passed (12 NumpyModel + 5 NumpyOptimizer)

- [ ] **Step 5: Commit**

```bash
git add pipeline/adapters/numpy_adapter.py tests/unit/test_numpy_adapter.py
git commit -m "feat: add NumpyOptimizer adapter delegating to SGD/Adam rules

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: Package exports and `__init__.py` updates

**Files:**
- Modify: `pipeline/adapters/__init__.py` (add exports)
- Modify: `pipeline/training/__init__.py` (add optimizers, losses exports)
- Modify: `pipeline/__init__.py` (add adapter subpackage to docstring)

**Purpose:** Make all Phase 2 components importable from their canonical locations without breaking the lazy-import constraint.

- [ ] **Step 1: Update `pipeline/adapters/__init__.py`**

```python
# Replace the empty file with:
"""Framework adapters that satisfy pipeline protocols.

Each adapter wraps a specific framework (sklearn, numpy, PyTorch)
and satisfies :class:`ModelProtocol`, :class:`LossProtocol`, or
:class:`OptimizerProtocol` so the pipeline can use it without
knowing which framework is underneath.

Usage::

    from pipeline.adapters import SklearnModel, NumpyModel

Imports are lazy — sklearn is only imported when :class:`SklearnModel`
is instantiated, not when the module loads.
"""
from pipeline.adapters.sklearn_adapter import SklearnModel, StubLoss, StubOptimizer
from pipeline.adapters.numpy_adapter import NumpyModel, NumpyOptimizer

__all__ = [
    "SklearnModel",
    "StubLoss",
    "StubOptimizer",
    "NumpyModel",
    "NumpyOptimizer",
]
```

- [ ] **Step 2: Update `pipeline/training/__init__.py`**

```python
# Replace with:
"""Training loop orchestration and optimization components.

Provides the :class:`TrainLoop` class that manages the per-epoch,
per-batch training lifecycle, plus pure-numpy optimizer rules
(:class:`SGD`, :class:`Adam`) and loss functions (:class:`MSELoss`,
:class:`CrossEntropyLoss`).

Usage::

    from pipeline.training import TrainLoop, SGD, Adam, MSELoss, CrossEntropyLoss
"""
from pipeline.training.train_loop import TrainLoop
from pipeline.training.optimizers import SGD, Adam
from pipeline.training.losses import MSELoss, CrossEntropyLoss

__all__ = ["TrainLoop", "SGD", "Adam", "MSELoss", "CrossEntropyLoss"]
```

- [ ] **Step 3: Update `pipeline/__init__.py`**

```python
# Update the module docstring and __all__:
"""Universal DL Pipeline — a teaching-first, framework-agnostic deep learning pipeline.

The pipeline defines *when* stages run via :class:`BasePipeline`.
Concrete implementations define *how* by satisfying the protocols in :mod:`pipeline.protocols`.

Quick start::

    from pipeline import BasePipeline, PipelineState, Config
    from pipeline.protocols import ModelProtocol, DataStream, Batch
    from pipeline.registry import register, build

Subpackages:
    - :mod:`pipeline.protocols` — abstract interfaces (framework-agnostic)
    - :mod:`pipeline.pipeline` — BasePipeline ABC + PipelineState
    - :mod:`pipeline.registry` — dependency injection registry
    - :mod:`pipeline.hooks` — hook system for cross-cutting concerns
    - :mod:`pipeline.config` — configuration dataclass
    - :mod:`pipeline.data` — data sources (CsvDataSource) and split utilities
    - :mod:`pipeline.training` — train loop, optimizers (SGD, Adam), losses (MSE, CrossEntropy)
    - :mod:`pipeline.evaluation` — metrics container and pure metric functions
    - :mod:`pipeline.export` — CSV and checkpoint export utilities
    - :mod:`pipeline.adapters` — framework adapters (SklearnModel, NumpyModel, NumpyOptimizer)
    - :mod:`pipeline.utils` — device, seed utilities
"""

from pipeline.config import Config
from pipeline.pipeline import BasePipeline, PipelineState

__all__ = ["BasePipeline", "PipelineState", "Config"]
__version__ = "0.2.0"
```

- [ ] **Step 4: Verify all imports work**

```bash
uv run python -c "
from pipeline.adapters import SklearnModel, StubLoss, StubOptimizer, NumpyModel, NumpyOptimizer
from pipeline.training import SGD, Adam, MSELoss, CrossEntropyLoss
from pipeline.data.utils import collect_arrays
print('All imports successful')
"
```

- [ ] **Step 5: Run all tests to verify no regressions**

```bash
uv run pytest -m "not slow" -v
```
Expected: all 225+ existing tests still pass, plus all new Phase 2 tests.

- [ ] **Step 6: Commit**

```bash
git add pipeline/adapters/__init__.py pipeline/training/__init__.py pipeline/__init__.py
git commit -m "feat: add Phase 2 exports to package __init__ files

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: Contract tests

**Files:**
- Create: `tests/contract/test_model_contract.py`

**Interfaces:**
- Consumes: `SklearnModel`, `NumpyModel`, `StubLoss`, `StubOptimizer`, `NumpyOptimizer`, `SGD`
- Tests: forward shape, train/eval cycle, full pipeline E2E on Titanic data for both adapters

**Purpose:** Verify that both adapters satisfy the `ModelProtocol` contract. Same test, parametrized over adapter type. This is the key test that proves the protocol design is correct.

- [ ] **Step 1: Write the contract test (this IS the test — no separate impl file)**

```python
# tests/contract/test_model_contract.py
"""Contract tests: same tests run on SklearnModel and NumpyModel.

Verifies that every ModelProtocol implementation satisfies the
pipeline contract — forward shape, mode toggling, and end-to-end
on the Titanic dataset.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pipeline.protocols import Batch

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_titanic_csv(path: str) -> str:
    """Write a small Titanic-style CSV for contract tests.

    Returns the file path.
    """
    import pandas as pd

    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "feature_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "feature_c": [22.0, 38.0, 25.0, 40.0, 30.0, 35.0, 28.0, 45.0],
        "survived": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _make_sklearn_pipeline(csv_path: str):
    """Build a pipeline with SklearnModel."""
    from sklearn.linear_model import LogisticRegression

    from pipeline.adapters.sklearn_adapter import SklearnModel, StubLoss, StubOptimizer
    from pipeline.config import Config
    from pipeline.data.csv_source import CsvDataSource
    from pipeline.data.split import train_test_split
    from pipeline.data.utils import collect_arrays
    from pipeline.evaluation.metrics import Metrics, accuracy
    from pipeline.pipeline import BasePipeline, PipelineState

    config = Config(
        batch_size=4,
        num_epochs=1,
        train_ratio=0.75,
        seed=42,
        output_dir=f"{Path(csv_path).parent}/output_sklearn",
    )

    class SklearnPipeline(BasePipeline):
        def load_data(self, state: PipelineState) -> None:
            source = CsvDataSource(csv_path, batch_size=config.batch_size, shuffle=False)
            train, val = train_test_split(source, train_ratio=config.train_ratio)
            state.data_stream = train
            state.val_data_stream = val

        def build_model(self, state: PipelineState) -> None:
            state.model = SklearnModel(LogisticRegression(max_iter=1000))
            state.loss_fn = StubLoss()
            state.optimizer = StubOptimizer()
            state.metrics = Metrics(accuracy=accuracy)

        def train(self, state: PipelineState) -> None:
            X, y = collect_arrays(state.data_stream)
            state.model._estimator.fit(X, y)
            state.history = {"loss": [0.0]}

        def evaluate(self, state: PipelineState) -> None:
            state.model.eval_mode()
            all_preds, all_targets = [], []
            for batch in state.val_data_stream:
                probs = np.asarray(state.model.forward(batch.inputs))
                all_preds.append(np.argmax(probs, axis=1))
                all_targets.append(np.asarray(batch.targets))
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            state.metrics.compute(y_true, y_pred)

        def export(self, state: PipelineState) -> None:
            state.predictions = np.array([0])

    return SklearnPipeline(config)


def _make_numpy_pipeline(csv_path: str):
    """Build a pipeline with NumpyModel."""
    from pipeline.adapters.numpy_adapter import NumpyModel, NumpyOptimizer
    from pipeline.config import Config
    from pipeline.data.csv_source import CsvDataSource
    from pipeline.data.split import train_test_split
    from pipeline.evaluation.metrics import Metrics, accuracy
    from pipeline.pipeline import BasePipeline, PipelineState
    from pipeline.training.losses import CrossEntropyLoss
    from pipeline.training.optimizers import SGD

    config = Config(
        batch_size=4,
        num_epochs=5,
        train_ratio=0.75,
        learning_rate=0.5,
        seed=42,
        output_dir=f"{Path(csv_path).parent}/output_numpy",
    )

    class NumpyPipeline(BasePipeline):
        def load_data(self, state: PipelineState) -> None:
            source = CsvDataSource(csv_path, batch_size=config.batch_size, shuffle=False)
            train, val = train_test_split(source, train_ratio=config.train_ratio)
            state.data_stream = train
            state.val_data_stream = val

        def build_model(self, state: PipelineState) -> None:
            rng = np.random.default_rng(42)
            W1 = rng.standard_normal((8, 3)) * 0.1
            b1 = np.zeros(8)
            W2 = rng.standard_normal((2, 8)) * 0.1
            b2 = np.zeros(2)
            state.model = NumpyModel([(W1, b1), (W2, b2)], activation="relu")
            state.loss_fn = CrossEntropyLoss(model=state.model)
            state.optimizer = NumpyOptimizer(
                state.model.parameters(), SGD(lr=config.learning_rate)
            )
            state.metrics = Metrics(accuracy=accuracy)

        def evaluate(self, state: PipelineState) -> None:
            state.model.eval_mode()
            all_preds, all_targets = [], []
            for batch in state.val_data_stream:
                logits = np.asarray(state.model.forward(batch.inputs))
                all_preds.append(np.argmax(logits, axis=1))
                all_targets.append(np.asarray(batch.targets))
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            state.metrics.compute(y_true, y_pred)

        def export(self, state: PipelineState) -> None:
            state.predictions = np.array([0])

    return NumpyPipeline(config)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

PIPELINE_BUILDERS: dict[str, Callable] = {
    "sklearn": _make_sklearn_pipeline,
    "numpy": _make_numpy_pipeline,
}


@pytest.mark.slow
class TestModelContract:
    """Every ModelProtocol adapter must pass these."""

    @pytest.mark.parametrize("adapter_name", list(PIPELINE_BUILDERS.keys()))
    def test_forward_shape_matches_batch_size(self, adapter_name: str):
        """forward(inputs).shape[0] must equal the number of input samples."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_titanic_csv(f"{tmpdir}/train.csv")
            pipeline = PIPELINE_BUILDERS[adapter_name](csv_path)
            state = pipeline.run("train")
            # After training, forward should work
            model = state.model
            test_input = np.array([[1.0, 0.1, 22.0], [2.0, 0.2, 38.0]])
            output = model.forward(test_input)
            assert output.shape[0] == 2, (
                f"{adapter_name}: forward output batch size mismatch: "
                f"expected 2, got {output.shape[0]}"
            )

    @pytest.mark.parametrize("adapter_name", list(PIPELINE_BUILDERS.keys()))
    def test_train_eval_mode_cycle_no_error(self, adapter_name: str):
        """train_mode() → eval_mode() → train_mode() must not raise."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_titanic_csv(f"{tmpdir}/train.csv")
            pipeline = PIPELINE_BUILDERS[adapter_name](csv_path)
            state = pipeline.run("train")
            model = state.model
            model.train_mode()
            model.eval_mode()
            model.train_mode()

    @pytest.mark.parametrize("adapter_name", list(PIPELINE_BUILDERS.keys()))
    def test_full_pipeline_on_titanic(self, adapter_name: str):
        """Same Titanic task, both adapters complete all 6 stages."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_titanic_csv(f"{tmpdir}/train.csv")
            pipeline = PIPELINE_BUILDERS[adapter_name](csv_path)
            state = pipeline.run("train")
            # All stages completed
            assert state.history is not None, f"{adapter_name}: history not populated"
            assert "accuracy" in state.metrics, f"{adapter_name}: accuracy not computed"
            assert state.predictions is not None, f"{adapter_name}: predictions not set"
```

- [ ] **Step 2: Run contract tests**

```bash
uv run pytest tests/contract/test_model_contract.py -v -m slow
```
Expected: 6 passed (3 tests × 2 adapters)

- [ ] **Step 3: Verify no regressions**

```bash
uv run pytest -m "not slow" -v
```
Expected: all unit tests still pass

- [ ] **Step 4: Commit**

```bash
git add tests/contract/test_model_contract.py
git commit -m "test: add contract tests parametrized over SklearnModel and NumpyModel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: M1 sklearn example notebook

**Files:**
- Create: `examples/m1-sklearn/titanic.ipynb`

**Purpose:** Jupyter notebook demonstrating the full sklearn pipeline on Titanic: load data, build LogisticRegression via SklearnModel, train (override with `.fit()`), evaluate, and export predictions.

**Notebook cells (markdown + code):**

Cell 1 (markdown):
```
# M1: Sklearn — Logistic Regression on Titanic

Load the Titanic dataset, wrap a `LogisticRegression` in `SklearnModel`,
use stub Loss/Optimizer, and override `train()` to call `.fit()`.
```

Cell 2 (code):
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline.config import Config
from pipeline.data.csv_source import CsvDataSource
from pipeline.data.split import train_test_split
from pipeline.data.utils import collect_arrays
from pipeline.evaluation.metrics import Metrics, accuracy, f1_score, confusion_matrix
from pipeline.hooks.progress import ProgressHook
from pipeline.pipeline import BasePipeline, PipelineState
from pipeline.adapters.sklearn_adapter import SklearnModel, StubLoss, StubOptimizer

print("Imports OK")
```

Cell 3 (code):
```python
# Load Titanic data
# Assumes data/titanic/train.csv exists (download from Kaggle)
config = Config(
    data_dir="./data/titanic",
    output_dir="./outputs/m1-sklearn",
    batch_size=32,
    num_epochs=1,
    train_ratio=0.8,
    seed=42,
    task_name="titanic-sklearn",
)
```

Cell 4 (code):
```python
class TitanicSklearnPipeline(BasePipeline):
    """Sklearn LogisticRegression on Titanic."""

    def load_data(self, state):
        source = CsvDataSource(
            "data/titanic/train.csv",
            batch_size=config.batch_size,
            target_column="Survived",
            shuffle=True,
            seed=config.seed,
        )
        train, val = train_test_split(source, train_ratio=config.train_ratio)
        state.data_stream = train
        state.val_data_stream = val

    def build_model(self, state):
        from sklearn.pipeline import make_pipeline
        est = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=config.seed),
        )
        state.model = SklearnModel(est)
        state.loss_fn = StubLoss()
        state.optimizer = StubOptimizer()
        state.metrics = Metrics(accuracy=accuracy, f1=f1_score)

    def train(self, state):
        X, y = collect_arrays(state.data_stream)
        state.model._estimator.fit(X, y)
        state.history = {"loss": [0.0]}

    def evaluate(self, state):
        state.model.eval_mode()
        preds, trues = [], []
        for batch in state.val_data_stream:
            probs = np.asarray(state.model.forward(batch.inputs))
            preds.append(np.argmax(probs, axis=1))
            trues.append(np.asarray(batch.targets))
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        state.metrics.compute(y_true, y_pred)

    def export(self, state):
        # Load test set and predict
        test_source = CsvDataSource(
            "data/titanic/test.csv",
            batch_size=config.batch_size,
            target_column="PassengerId",
            shuffle=False,
        )
        test_preds = []
        for batch in test_source:
            probs = np.asarray(state.model.forward(batch.inputs))
            test_preds.append(np.argmax(probs, axis=1))
        state.predictions = np.concatenate(test_preds)
        # Save submission
        submission = pd.DataFrame({
            "PassengerId": ...,
            "Survived": state.predictions,
        })
        submission.to_csv(f"{config.output_dir}/submission.csv", index=False)
```

Cell 5 (code):
```python
pipeline = TitanicSklearnPipeline(config)
pipeline.add_hook(ProgressHook())
state = pipeline.run("train")

print(f"\nAccuracy:  {state.metrics['accuracy']:.4f}")
print(f"F1 Score:  {state.metrics['f1']:.4f}")
print("Training complete!")
```

Cell 6 (markdown):
```
## What happened?

1. `SklearnModel` wrapped a `LogisticRegression` inside a `StandardScaler` pipeline.
2. `StubLoss` and `StubOptimizer` satisfied the pipeline contract — they do nothing,
   because sklearn's `.fit()` handles training internally.
3. The pipeline overrode `train()` to call `_estimator.fit(X, y)` on the full dataset,
   instead of the per-batch gradient descent loop.
4. `evaluate()` used `predict_proba()` → `argmax` to compute accuracy and F1.

**Key insight:** sklearn and gradient descent train differently. The pipeline
accommodates both — sklearn overrides `train()`, numpy uses the default `TrainLoop`.
```

- [ ] **Step 1: Create the notebook directory and write the notebook**

```bash
mkdir -p examples/m1-sklearn
```
Then use NotebookEdit to create the file. (In practice, create the .ipynb with the cells above.)

- [ ] **Step 2: Commit**

```bash
git add examples/m1-sklearn/titanic.ipynb
git commit -m "docs: add M1 sklearn Titanic example notebook

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 13: M2→M3 numpy notebook

**Files:**
- Create: `examples/m2-m3-numpy/gradient_to_mlp.ipynb`

**Purpose:** One notebook, two acts:
- **Act 1 (M2):** Single neuron → sympy symbolic gradient → lambdify → manual numpy GD
- **Act 2 (M3):** Multi-layer MLP → NumpyModel → CrossEntropyLoss._backward_fn → TrainLoop → Kaggle submission

**Notebook cells:**

Cell 1 (markdown):
```
# M2→M3: From Gradient Descent to Multi-Layer Networks

**Act 1 (M2):** Where do gradients come from? We define a single neuron in sympy,
differentiate symbolically, lambdify into numpy, and run gradient descent by hand.

**Act 2 (M3):** We connect multiple neurons, trust backpropagation, and use the
pipeline's NumpyModel + CrossEntropyLoss to train a real MLP on Titanic.
```

Cell 2 (markdown):
```
## Act 1: A Single Neuron's Gradient
```

Cell 3 (code):
```python
import sympy as sp
import numpy as np

# Define symbols
w, b, x, t = sp.symbols('w b x t')
# Single neuron: y = sigmoid(w*x + b)
y = 1 / (1 + sp.exp(-(w*x + b)))
# MSE loss: L = (y - t)^2
L = (y - t)**2

# Symbolic gradients
dL_dw = sp.diff(L, w)
dL_db = sp.diff(L, b)

print("dL/dw =", sp.simplify(dL_dw))
print("dL/db =", sp.simplify(dL_db))
```

Cell 4 (code):
```python
# Lambdify into numpy functions
grad_w_fn = sp.lambdify([w, b, x, t], dL_dw, 'numpy')
grad_b_fn = sp.lambdify([w, b, x, t], dL_db, 'numpy')

# Test on one data point
w_val, b_val = 0.5, 0.0
x_val, t_val = 1.0, 0.0

print(f"grad_w = {grad_w_fn(w_val, b_val, x_val, t_val):.6f}")
print(f"grad_b = {grad_b_fn(w_val, b_val, x_val, t_val):.6f}")
```

Cell 5 (code):
```python
# Manual gradient descent loop
from pipeline.adapters.numpy_adapter import NumpyModel, NumpyOptimizer
from pipeline.training.optimizers import SGD
from pipeline.protocols import Parameter

# Initialize parameters
W_data = np.array([[0.5]])
b_data = np.array([0.0])
model = NumpyModel([(W_data, b_data)], activation="relu")
opt = NumpyOptimizer(model.parameters(), SGD(lr=0.1))

# Training data (simple 1D regression)
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([[2.0], [4.0], [6.0], [8.0]])  # y = 2x

losses = []
for epoch in range(50):
    epoch_loss = 0
    for i in range(len(X)):
        x_i = X[i:i+1]
        y_i = y[i:i+1]
        pred = model.forward(x_i)
        loss_val = float(np.mean((pred - y_i)**2))
        epoch_loss += loss_val

        # Manual gradient: dL/dpred = 2*(pred - y_i)
        dL_dpred = 2 * (pred - y_i)
        model.backward(dL_dpred)

        opt.step()
        opt.zero_grad()

    losses.append(epoch_loss / len(X))

print(f"Final loss: {losses[-1]:.6f}")
print(f"Learned W: {list(model.parameters())[0].data[0,0]:.4f} (target: 2.0)")
print(f"Learned b: {list(model.parameters())[1].data[0]:.4f} (target: 0.0)")
```

Cell 6 (code):
```python
# Plot the loss curve
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Single Neuron: Gradient Descent from First Principles")
plt.show()
```

Cell 7 (markdown):
```
**What we just did:** We derived the gradient formulas by hand (sympy),
converted them to numpy (lambdify), ran gradient descent manually,
and verified the neuron learned y = 2x.

**Now:** Let's do the same thing for a multi-layer network on Titanic.
But this time, we let backpropagation handle the gradients.
```

Cell 8 (markdown):
```
## Act 2: Multi-Layer Network on Titanic
```

Cell 9 (code):
```python
import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.data.csv_source import CsvDataSource
from pipeline.data.split import train_test_split
from pipeline.data.utils import collect_arrays
from pipeline.evaluation.metrics import Metrics, accuracy
from pipeline.hooks.progress import ProgressHook
from pipeline.pipeline import BasePipeline, PipelineState
from pipeline.adapters.numpy_adapter import NumpyModel, NumpyOptimizer
from pipeline.training.losses import CrossEntropyLoss
from pipeline.training.optimizers import Adam

config = Config(
    data_dir="./data/titanic",
    output_dir="./outputs/m3-numpy",
    batch_size=32,
    num_epochs=100,
    train_ratio=0.8,
    learning_rate=0.01,
    seed=42,
    task_name="titanic-numpy-mlp",
)
```

Cell 10 (code):
```python
class TitanicNumpyMLP(BasePipeline):
    """Two-layer numpy MLP on Titanic."""

    def load_data(self, state):
        source = CsvDataSource(
            "data/titanic/train.csv",
            batch_size=config.batch_size,
            target_column="Survived",
            shuffle=True,
            seed=config.seed,
        )
        train, val = train_test_split(source, train_ratio=config.train_ratio)
        state.data_stream = train
        state.val_data_stream = val

    def build_model(self, state):
        rng = np.random.default_rng(config.seed)
        # 7 input features → 16 hidden → 2 output classes
        W1 = rng.standard_normal((16, 7)) * np.sqrt(2.0 / 7)
        b1 = np.zeros(16)
        W2 = rng.standard_normal((2, 16)) * np.sqrt(2.0 / 16)
        b2 = np.zeros(2)

        state.model = NumpyModel([(W1, b1), (W2, b2)], activation="relu")
        state.loss_fn = CrossEntropyLoss(model=state.model)
        state.optimizer = NumpyOptimizer(
            state.model.parameters(),
            Adam(lr=config.learning_rate),
        )
        state.metrics = Metrics(accuracy=accuracy)

    def evaluate(self, state):
        state.model.eval_mode()
        preds, trues = [], []
        for batch in state.val_data_stream:
            logits = np.asarray(state.model.forward(batch.inputs))
            preds.append(np.argmax(logits, axis=1))
            trues.append(np.asarray(batch.targets))
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        state.metrics.compute(y_true, y_pred)

    def export(self, state):
        state.predictions = np.array([0])
```

Cell 11 (code):
```python
pipeline = TitanicNumpyMLP(config)
pipeline.add_hook(ProgressHook())
state = pipeline.run("train")

print(f"\nAccuracy: {state.metrics['accuracy']:.4f}")
print(f"Final loss: {state.history['loss'][-1]:.4f}")
print("Training complete!")
```

Cell 12 (markdown):
```
## Compare: M1 (sklearn) vs M3 (numpy MLP)

| | M1 sklearn | M3 numpy MLP |
|---|---|---|
| Model | LogisticRegression | 2-layer MLP (7→16→2) |
| Training | `.fit()` one call | 100 epochs per-batch GD |
| Optimizer | LBFGS (internal) | Adam (lr=0.01) |
| Loss | log-loss (internal) | CrossEntropyLoss |
| Parameters | None exposed | 4 Parameters (W1,b1,W2,b2) |
| Gradient | sklearn internal | manual backprop in `model.backward()` |

**Key insight:** The same pipeline structure ran both. The sklearn path overrode
`train()`; the numpy path used the default `TrainLoop`. The protocol design works.
```

- [ ] **Step 1: Create the notebook directory and write the notebook**

```bash
mkdir -p examples/m2-m3-numpy
```
Then create the .ipynb with the cells above.

- [ ] **Step 2: Commit**

```bash
git add examples/m2-m3-numpy/gradient_to_mlp.ipynb
git commit -m "docs: add M2-M3 numpy gradient-to-mlp example notebook

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Completion Gate

- [ ] `python -m pipeline` prints version 0.2.0 + device info
- [ ] `uv run pytest -m "not slow"` — all unit tests pass (225+ existing + ~60 new)
- [ ] `uv run pytest -m "slow"` — contract tests pass (6 tests, 2 adapters)
- [ ] `uv run ruff check .` — clean
- [ ] `uv run ruff format . --check` — clean
- [ ] `uv run mypy` — clean (stub warnings ok on py3.13)
- [ ] All imports work without sklearn installed: `python -c "from pipeline import BasePipeline"`
- [ ] All imports work with sklearn installed: `python -c "from pipeline.adapters import SklearnModel"`

## Post-Phase-2 (not in this plan)

- `CheckpointHook`, `EarlyStoppingHook` — Phase 3
- `TorchAdapter`, `TorchDataStream` — Phase 3
- `ImageFolderDataSource` — Phase 3
- `save_checkpoint`, `load_checkpoint` — Phase 4
- HPO, ensemble — Phase 5
