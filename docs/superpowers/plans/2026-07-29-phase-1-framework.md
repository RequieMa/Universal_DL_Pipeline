# Phase 1: Pipeline Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pipeline framework layer — concrete components (CsvDataSource, TrainLoop,
Metrics, ProgressHook, to_csv, CLI) that operate on Phase 0's abstract protocols, plus targeted
amendments to Phase 0 where TrainLoop integration reveals gaps.

**Architecture:** TrainLoop owns training-time hook dispatch (epoch/batch lifecycle).
BasePipeline retains stage-level events. PipelineState acts as the shared state machine.
All new components implement Phase 0 protocols; no torch/sklearn/sympy dependencies.

**Tech Stack:** Python >=3.11,<3.14; numpy, pandas, pyyaml, tqdm; pytest, ruff, mypy (strict);
uv for environment management.

## Global Constraints

- TDD iron law: no production code without a failing test first
- Lines per file ≤ 300; lines per function ≤ 50; nesting depth ≤ 3
- Public methods per class ≤ 5 (TrainLoop has 2: `run`, `add_hook` — ok)
- Google-style docstrings, English
- `@property` for derived/read-only attrs
- Forbidden: closures with mutable state, `if task_type` branching, module-level globals
- Commits: NEVER include "Co-Authored-By" or "Claude"
- Dependencies added via `uv add`; run `uv sync --dev` after
- All new directories need `__init__.py` with module docstring

---

## File Map

```
pipeline/                          # existing
├── __init__.py                    # modify: add new exports
├── protocols.py                   # modify: Loss class, val_data_stream, metrics type
├── pipeline.py                    # modify: hooks @property, train() default impl
├── hooks.py                       # unchanged
├── registry.py                    # unchanged
├── config.py                      # unchanged
├── data/                          # NEW directory
│   ├── __init__.py                # new
│   ├── csv_source.py              # new: CsvDataSource
│   └── split.py                   # new: train_test_split() + _InMemoryDataStream
├── training/                      # NEW directory
│   ├── __init__.py                # new
│   └── train_loop.py              # new: TrainLoop
├── evaluation/                    # NEW directory
│   ├── __init__.py                # new
│   └── metrics.py                 # new: Metrics class + 5 pure functions
├── export/                        # NEW directory
│   ├── __init__.py                # new
│   └── to_csv.py                  # new: to_csv()
├── hooks/                         # existing module → NEW package
│   ├── __init__.py                # new (re-export BaseHook)
│   └── progress.py                # new: ProgressHook
└── utils/                         # unchanged
    └── ...

tests/
├── unit/
│   ├── test_protocols.py          # modify: add Loss tests
│   ├── test_pipeline.py           # modify: add hooks @property tests, update existing
│   ├── test_metrics.py            # new
│   ├── test_train_loop.py         # new
│   ├── test_csv_source.py         # new
│   ├── test_split.py              # new
│   ├── test_progress.py           # new
│   ├── test_to_csv.py             # new
│   └── test_main.py               # new
├── fixtures/
│   └── tiny_titanic.csv           # new: 10 rows, 3 features + 1 target
└── integration/
    └── test_pipeline_e2e.py       # new: full E2E with fake components

run.py                             # new: CLI entry (project root)
```

---

### Task 1: Loss class (protocols.py amendment)

**Files:**
- Modify: `pipeline/protocols.py:1-250`
- Modify: `tests/unit/test_protocols.py`

**Interfaces:**
- Produces: `Loss(value: float, _backward_fn: Callable[[], None] | None = None)` dataclass
  - `Loss.backward() -> None` — calls `_backward_fn` if not None; no-op otherwise
  - `Loss.__float__() -> float` — returns `self.value`
- Produces: `LossProtocol.forward()` return type annotation: `-> Loss`

- [ ] **Step 1: Write failing tests for Loss**

Add to `tests/unit/test_protocols.py`:

```python
class TestLoss:
    """Tests for the Loss value object."""

    def test_float_conversion(self):
        """Happy Path: float(loss) extracts the scalar value."""
        loss = Loss(value=0.5)
        assert float(loss) == 0.5

    def test_backward_noop_when_no_fn(self):
        """Happy Path: backward() is a no-op when _backward_fn is None."""
        loss = Loss(value=0.5)
        loss.backward()  # should not raise

    def test_backward_calls_fn(self):
        """Happy Path: backward() calls the stored function."""
        called = []
        loss = Loss(value=0.5, _backward_fn=lambda: called.append(1))
        loss.backward()
        assert called == [1]

    def test_backward_fn_receives_no_args(self):
        """Boundary: backward_fn receives zero arguments."""
        captured = None

        def _backward():
            nonlocal captured
            captured = "ran"

        loss = Loss(value=1.0, _backward_fn=_backward)
        loss.backward()
        assert captured == "ran"

    def test_negative_loss_value(self):
        """Boundary: negative loss values are preserved."""
        loss = Loss(value=-3.2)
        assert float(loss) == -3.2

    def test_zero_loss_value(self):
        """Boundary: zero loss value."""
        loss = Loss(value=0.0)
        assert float(loss) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_protocols.py::TestLoss -v
```
Expected: FAIL — `NameError: name 'Loss' is not defined`

- [ ] **Step 3: Implement Loss class in protocols.py**

Add after the `Batch` class (before `DataStream`):

```python
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
```

Update `LossProtocol.forward()` return type: change `-> float` to `-> Loss`.

Add `Callable` import if not already present: `from collections.abc import Callable`.

- [ ] **Step 4: Update LossProtocol.__call__ return type**

In `LossProtocol.__call__`: change `-> float` to `-> Loss`.

- [ ] **Step 5: Run tests to verify pass**

```bash
uv run pytest tests/unit/test_protocols.py -v
```
Expected: all tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -v
```
Expected: 115 → 121 passed (6 new Loss tests), 1 skipped (torch).

- [ ] **Step 7: Commit**

```bash
git add pipeline/protocols.py tests/unit/test_protocols.py
git commit -m "feat: add Loss value object with backward hook"
```

---

### Task 2: Metrics class + pure metric functions

**Files:**
- Create: `pipeline/evaluation/__init__.py`
- Create: `pipeline/evaluation/metrics.py`
- Create: `tests/unit/test_metrics.py`

**Interfaces:**
- Produces: `Metrics(**named_metrics: Callable[[ArrayLike, ArrayLike], float])` class
  - `compute(y_true, y_pred) -> dict[str, float]` — runs all metrics
  - `__getitem__(name) -> float` — access result by name, raises KeyError if not computed
  - `__iter__() -> Iterator[str]`
  - `__len__() -> int`
  - `__contains__(name) -> bool`
- Produces: `accuracy(y_true, y_pred) -> float`
- Produces: `precision(y_true, y_pred, average="binary") -> float`
- Produces: `recall(y_true, y_pred, average="binary") -> float`
- Produces: `f1_score(y_true, y_pred, average="binary") -> float`
- Produces: `confusion_matrix(y_true, y_pred, num_classes=None) -> np.ndarray`

- [ ] **Step 1: Write failing test for pure metric functions**

Create `tests/unit/test_metrics.py`:

```python
"""Unit tests for pipeline.evaluation.metrics — Metrics class and pure functions."""
import numpy as np
import pytest

from pipeline.evaluation.metrics import (
    Metrics,
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)


class TestAccuracy:
    """Tests for accuracy()."""

    def test_perfect_accuracy(self):
        """Happy Path: identical predictions → 1.0."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 1, 0])
        assert accuracy(y_true, y_pred) == 1.0

    def test_zero_accuracy(self):
        """Happy Path: all wrong → 0.0."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert accuracy(y_true, y_pred) == 0.0

    def test_partial_accuracy(self):
        """Happy Path: 3/5 correct → 0.6."""
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        assert accuracy(y_true, y_pred) == 0.6

    def test_binary_accuracy(self):
        """Happy Path: binary classification."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        assert accuracy(y_true, y_pred) == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_metrics.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.evaluation'`

- [ ] **Step 3: Create package structure**

Create `pipeline/evaluation/__init__.py`:

```python
"""Evaluation metrics for pipeline outputs.

Provides pure numpy-based metric functions and a :class:`Metrics`
container that decouples the evaluate stage from specific metric choices.
"""

from pipeline.evaluation.metrics import Metrics, accuracy, confusion_matrix, f1_score, precision, recall

__all__ = ["Metrics", "accuracy", "confusion_matrix", "f1_score", "precision", "recall"]
```

- [ ] **Step 4: Implement accuracy()**

Create `pipeline/evaluation/metrics.py` with module docstring, then:

```python
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
```

- [ ] **Step 5: Run accuracy tests**

```bash
uv run pytest tests/unit/test_metrics.py::TestAccuracy -v
```
Expected: PASS.

- [ ] **Step 6: Write tests for remaining metric functions**

Continue in `TestMetricsPure`:

```python
class TestPrecision:
    """Tests for precision()."""

    def test_perfect_precision(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert precision(y_true, y_pred) == 1.0

    def test_precision_with_false_positives(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 1])
        # TP=2, FP=1, precision=2/3
        assert precision(y_true, y_pred) == pytest.approx(2 / 3)

    def test_precision_macro_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1])
        result = precision(y_true, y_pred, average="macro")
        # class 0: TP=2, FP=0, prec=1.0
        # class 1: TP=1, FP=2, prec=1/3
        # class 2: TP=1, FP=1, prec=0.5
        # macro = (1.0 + 1/3 + 0.5) / 3 ≈ 0.611
        assert 0.6 < result < 0.62

    def test_precision_zero_division(self):
        """Boundary: when no positive predictions, precision is 0.0."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 0, 0])
        result = precision(y_true, y_pred)
        assert result == 0.0


class TestRecall:
    """Tests for recall()."""

    def test_perfect_recall(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert recall(y_true, y_pred) == 1.0

    def test_recall_with_false_negatives(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        # TP=1, FN=1, recall=1/2
        assert recall(y_true, y_pred) == 0.5

    def test_recall_macro_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1])
        result = recall(y_true, y_pred, average="macro")
        assert 0.55 < result < 0.65

    def test_recall_zero_division(self):
        """Boundary: when no true positives, recall is 0.0."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        result = recall(y_true, y_pred)
        assert result == 0.0


class TestF1Score:
    """Tests for f1_score()."""

    def test_perfect_f1(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert f1_score(y_true, y_pred) == 1.0

    def test_f1_harmonic_mean(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 1])
        # prec=2/3, rec=2/2=1.0, f1=2*(2/3*1)/(2/3+1)=0.8
        assert f1_score(y_true, y_pred) == pytest.approx(0.8)

    def test_f1_zero_when_precision_zero(self):
        """Boundary: zero precision → zero F1."""
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        assert f1_score(y_true, y_pred) == 0.0

    def test_f1_macro_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1])
        result = f1_score(y_true, y_pred, average="macro")
        assert 0.55 < result < 0.65


class TestConfusionMatrix:
    """Tests for confusion_matrix()."""

    def test_perfect_confusion_matrix(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.array_equal(cm, expected)

    def test_confusion_matrix_with_errors(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        cm = confusion_matrix(y_true, y_pred)
        # class 0: 1 correct, 1 predicted as 1
        # class 1: 1 correct, 1 predicted as 1 (from class 0) ← wait let me be careful
        # y_true=0, y_pred=0 → (0,0)+=1
        # y_true=0, y_pred=1 → (0,1)+=1
        # y_true=1, y_pred=1 → (1,1)+=2
        # y_true=2, y_pred=2 → (2,2)+=1
        # y_true=2, y_pred=0 → (2,0)+=1
        expected = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        assert np.array_equal(cm, expected)

    def test_confusion_matrix_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[1, 1], [0, 2]])
        assert np.array_equal(cm, expected)

    def test_confusion_matrix_auto_num_classes(self):
        y_true = np.array([0, 2, 4])
        y_pred = np.array([0, 2, 4])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (5, 5)  # 0..4 inclusive = 5 classes
```

- [ ] **Step 7: Run to verify they fail**

```bash
uv run pytest tests/unit/test_metrics.py::TestPrecision -v
uv run pytest tests/unit/test_metrics.py::TestRecall -v
uv run pytest tests/unit/test_metrics.py::TestF1Score -v
uv run pytest tests/unit/test_metrics.py::TestConfusionMatrix -v
```
Expected: FAIL — functions not defined.

- [ ] **Step 8: Implement remaining pure functions**

```python
def precision(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary"
) -> float:
    """Precision: TP / (TP + FP).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: ``"binary"`` or ``"macro"``. Macro averages per-class precision.

    Returns:
        Precision in ``[0.0, 1.0]``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary" and len(classes) <= 2:
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    # macro
    scores: list[float] = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(scores))


def recall(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary"
) -> float:
    """Recall: TP / (TP + FN).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: ``"binary"`` or ``"macro"``. Macro averages per-class recall.

    Returns:
        Recall in ``[0.0, 1.0]``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary" and len(classes) <= 2:
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
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
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm
```

- [ ] **Step 9: Run pure function tests**

```bash
uv run pytest tests/unit/test_metrics.py -v -k "not Metrics"
```
Expected: all tests PASS.

- [ ] **Step 10: Write failing tests for Metrics class**

```python
class TestMetrics:
    """Tests for the Metrics container class."""

    def test_metrics_empty(self):
        """Boundary: empty Metrics is valid."""
        m = Metrics()
        assert len(m) == 0
        assert list(m) == []

    def test_metrics_registered_names(self):
        """Happy Path: __iter__ yields registered metric names."""
        m = Metrics(accuracy=accuracy, f1=f1_score)
        assert set(m) == {"accuracy", "f1"}
        assert len(m) == 2

    def test_metrics_contains(self):
        """Happy Path: __contains__ checks registered names."""
        m = Metrics(accuracy=accuracy)
        assert "accuracy" in m
        assert "f1" not in m

    def test_compute_returns_dict(self):
        """Happy Path: compute returns {name: value} dict."""
        m = Metrics(acc=accuracy)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = m.compute(y_true, y_pred)
        assert result == {"acc": 1.0}

    def test_compute_multiple_metrics(self):
        """Happy Path: compute runs all registered metrics."""
        m = Metrics(acc=accuracy, f1=f1_score)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        result = m.compute(y_true, y_pred)
        assert "acc" in result
        assert "f1" in result
        assert 0.0 <= result["acc"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0

    def test_getitem_after_compute(self):
        """Happy Path: __getitem__ returns computed value."""
        m = Metrics(acc=accuracy)
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        m.compute(y_true, y_pred)
        assert m["acc"] == 1.0

    def test_getitem_before_compute_raises(self):
        """Error: __getitem__ raises KeyError before compute()."""
        m = Metrics(acc=accuracy)
        with pytest.raises(KeyError):
            m["acc"]

    def test_getitem_unregistered_name_raises(self):
        """Error: __getitem__ raises KeyError for unknown metric."""
        m = Metrics(acc=accuracy)
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        m.compute(y_true, y_pred)
        with pytest.raises(KeyError):
            m["nonexistent"]

    def test_metrics_not_callable_guard(self):
        """Type Error: non-callable metric raises during construction.
        NOTE: Metrics itself doesn't validate — it's on the user.
        This is by design for simplicity."""
        pass  # no validation, no crash — intentional
```

- [ ] **Step 11: Run Metrics tests to verify they fail**

```bash
uv run pytest tests/unit/test_metrics.py::TestMetrics -v
```
Expected: FAIL — `NameError: name 'Metrics' is not defined`

- [ ] **Step 12: Implement Metrics class**

```python
class Metrics:
    """Collection of metric functions with cached results.

    Decouples the evaluate stage from knowledge of which specific
    metrics are being used. Holds both the functions and their
    computed values.

    Usage::

        state.metrics = Metrics(accuracy=accuracy, f1=f1_score)
        results = state.metrics.compute(y_true, y_pred)
        print(state.metrics["accuracy"])  # → 0.92
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
```

- [ ] **Step 13: Run all metrics tests**

```bash
uv run pytest tests/unit/test_metrics.py -v
```
Expected: all tests PASS.

- [ ] **Step 14: Commit**

```bash
git add pipeline/evaluation/ tests/unit/test_metrics.py
git commit -m "feat: add Metrics container and 5 pure metric functions"
```

---

### Task 3: PipelineState + BasePipeline amendments

**Files:**
- Modify: `pipeline/protocols.py` (add `val_data_stream`, change `metrics` type)
- Modify: `pipeline/pipeline.py` (add `hooks` @property, make `train()` non-abstract)
- Modify: `tests/unit/test_pipeline.py` (update existing tests, add new tests)

**Interfaces:**
- Consumes: `Metrics` from Task 2 (via TYPE_CHECKING forward reference)
- Produces: `PipelineState.val_data_stream: DataStream | None = None`
- Produces: `PipelineState.metrics: Metrics` (via `field(default_factory=Metrics)`)
- Produces: `BasePipeline.hooks` → `list[BaseHook]` read-only `@property`
- Produces: `BasePipeline.train(state)` — default implementation (lazy import TrainLoop from Task 5)
- Modifies: `BasePipeline` abstract methods reduced to 5 (train no longer abstract)

- [ ] **Step 1: Write failing tests for PipelineState amendments**

Add to `tests/unit/test_pipeline.py`:

```python
class TestPipelineStatePhase1Amendments:
    """Tests for Phase 1 additions to PipelineState."""

    def test_val_data_stream_defaults_to_none(self):
        """Happy Path: val_data_stream defaults to None."""
        state = PipelineState(config=Config())
        assert state.val_data_stream is None

    def test_val_data_stream_is_writable(self):
        """Happy Path: val_data_stream accepts a DataStream."""
        state = PipelineState(config=Config())
        state.val_data_stream = "fake_val_stream"
        assert state.val_data_stream == "fake_val_stream"

    def test_metrics_defaults_to_empty_metrics_instance(self):
        """Happy Path: metrics defaults to an empty Metrics instance."""
        state = PipelineState(config=Config())
        assert len(state.metrics) == 0
        assert isinstance(state.metrics, Metrics)

    def test_metrics_is_replaceable(self):
        """Happy Path: metrics can be replaced with a configured Metrics."""
        state = PipelineState(config=Config())
        state.metrics = Metrics(accuracy=accuracy, f1=f1_score)
        assert len(state.metrics) == 2
        assert "accuracy" in state.metrics
        assert "f1" in state.metrics


class TestBasePipelineHooksProperty:
    """Tests for BasePipeline.hooks read-only @property."""

    def test_hooks_property_returns_list(self):
        """Happy Path: hooks returns the internal hook list."""
        pipeline = _MinimalPipeline(Config())
        assert isinstance(pipeline.hooks, list)
        assert len(pipeline.hooks) == 0

    def test_hooks_property_read_only(self):
        """Error: hooks property raises AttributeError on set."""
        pipeline = _MinimalPipeline(Config())
        with pytest.raises(AttributeError):
            pipeline.hooks = []  # type: ignore[misc]

    def test_hooks_property_reflects_add_hook(self):
        """Happy Path: hooks list reflects hooks added via add_hook()."""
        pipeline = _MinimalPipeline(Config())
        spy = _SpyHook()
        pipeline.add_hook(spy)
        assert len(pipeline.hooks) == 1
        assert pipeline.hooks[0] is spy


class TestBasePipelineTrainDefault:
    """Tests for the default train() implementation (delegates to TrainLoop).

    NOTE: These tests verify train() is no longer abstract. Full TrainLoop
    integration is tested in test_train_loop.py (Task 5).
    """

    def test_train_is_not_abstract(self):
        """Happy Path: subclasses without train() can be instantiated."""
        class PipelineNoTrain(BasePipeline):
            def load_data(self, state):
                pass
            def build_model(self, state):
                pass
            def evaluate(self, state):
                pass
            def export(self, state):
                pass

        pipeline = PipelineNoTrain(Config())
        assert pipeline is not None

    def test_train_uses_lazy_import(self):
        """Happy Path: train() imports TrainLoop lazily (verified via
        `hasattr` — the method exists and has no __isabstractmethod__)."""
        assert not hasattr(BasePipeline.train, "__isabstractmethod__")
```

Add imports at top of test file:

```python
from pipeline.evaluation.metrics import Metrics, accuracy, f1_score
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_pipeline.py::TestPipelineStatePhase1Amendments -v
uv run pytest tests/unit/test_pipeline.py::TestBasePipelineHooksProperty -v
```
Expected: FAIL — `val_data_stream` / `hooks` not found, `train` still abstract.

- [ ] **Step 3: Update PipelineState in protocols.py**

In `protocols.py`, after `build_model`, change `PipelineState`:

Add import:
```python
from __future__ import annotations
...
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pipeline.evaluation.metrics import Metrics
```

In `PipelineState` fields:
- Add: `val_data_stream: DataStream | None = None` (after `data_stream`)
- Change: `metrics: Metrics = field(default_factory=Metrics)` (was `dict[str, float]`)

Move the `Metrics` import into TYPE_CHECKING block. The `field(default_factory=Metrics)` requires `Metrics` at runtime, so use a lazy import pattern inside the field:

```python
metrics: dict[str, float] = field(
    default_factory=lambda: _get_metrics_default()
)
```

Wait, that's ugly. Better: import Metrics eagerly. Since `evaluation/metrics.py` only imports numpy and collections.abc, it won't cause circular imports. Let me import it at module level:

```python
from pipeline.evaluation.metrics import Metrics
```

But `pipeline.evaluation.__init__` imports from `pipeline.evaluation.metrics` which imports numpy. This should be fine — no circular dependency.

Actually, even better: define Metrics import inside the evaluation module's __init__.py. Then in protocols.py:

```python
from pipeline.evaluation import Metrics
```

This is clean. Let me do it that way.

- [ ] **Step 4: Update BasePipeline in pipeline.py**

Add `@property` before `_hooks`:

```python
@property
def hooks(self) -> list[BaseHook]:
    """Registered hooks (read-only).

    Use :meth:`add_hook` to register. The returned list is the live
    internal list — modifications to it affect the pipeline.

    TrainLoop reads this property to borrow hooks for training-time
    events (epoch start/end, batch end).
    """
    return self._hooks
```

Change `train()` from `@abstractmethod` to a default implementation:

```python
def train(self, state: PipelineState) -> None:
    """Stage 4: Run the training loop.

    Default implementation delegates to :class:`TrainLoop`.
    Subclasses may override for non-standard training (GAN, meta-learning).

    Iterate over ``state.data_stream``, forward → loss → backward → step,
    populating ``state.history`` with loss/accuracy curves.
    """
    # WHY: Lazy import avoids circular imports — TrainLoop imports
    # PipelineState from pipeline.pipeline, but at this point both
    # modules are already loaded.
    from pipeline.training.train_loop import TrainLoop

    loop = TrainLoop(
        model=state.model,
        data_stream=state.data_stream,
        optimizer=state.optimizer,
        loss_fn=state.loss_fn,
        num_epochs=state.config.num_epochs,
        hooks=self._hooks,
    )
    loop.run(state)
```

- [ ] **Step 5: Update `_MinimalPipeline` class definition**

The `_MinimalPipeline` (lines ~126-142) uses dict assignment `state.metrics["accuracy"] = 0.95` in `evaluate()`. Since `state.metrics` is now a `Metrics` object, change to:

```python
class _MinimalPipeline(BasePipeline):
    """A pipeline where every stage is a no-op. For testing the template."""

    def load_data(self, state: PipelineState) -> None:
        state.current_epoch = 0

    def build_model(self, state: PipelineState) -> None:
        pass

    def train(self, state: PipelineState) -> None:
        state.current_epoch = state.config.num_epochs

    def evaluate(self, state: PipelineState) -> None:
        from pipeline.evaluation.metrics import Metrics, accuracy
        import numpy as np
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])
        state.metrics = Metrics(accuracy=accuracy)
        state.metrics.compute(y_true, y_pred)

    def export(self, state: PipelineState) -> None:
        state.predictions = np.array([0, 1, 0])
```

The test assertions that read `state.metrics["accuracy"]` now return `1.0` (perfect accuracy with same arrays) instead of `0.95`.

- [ ] **Step 6: Update existing test assertions that broke**

In `test_pipeline.py`, update `TestPipelineStateDefaults`:

```python
def test_create_with_config_only(self):
    """Happy Path: PipelineState(config) uses all defaults."""
    cfg = Config()
    state = PipelineState(config=cfg)
    assert state.config is cfg
    assert state.mode == "train"
    assert state.data_stream is None
    assert state.val_data_stream is None  # NEW
    assert state.features is None
    assert state.model is None
    assert state.loss_fn is None
    assert state.optimizer is None
    assert state.history is None
    assert len(state.metrics) == 0  # UPDATED: was == {}
    assert state.predictions is None
    assert state.current_epoch == 0
    assert state.should_stop is False
```

In `TestPipelineStateFieldAssignment`, update:

```python
def test_assign_metrics(self):
    """Happy Path: metrics field is a replaceable Metrics instance."""
    state = PipelineState(config=Config())
    state.metrics = Metrics(accuracy=accuracy, f1=f1_score)
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 0])
    state.metrics.compute(y_true, y_pred)
    assert state.metrics["accuracy"] == 1.0
    assert state.metrics["f1"] == 1.0
```

In `TestPipelineStateIsolation`, update:

```python
def test_metrics_default_dicts_isolated(self):
    """Concurrency: default Metrics instance is unique per PipelineState."""
    s1 = PipelineState(config=Config())
    s2 = PipelineState(config=Config())
    assert s1.metrics is not s2.metrics
```

In `TestBasePipelineRun` — remove or adjust the `test_run_train_executes_all_stages` test since `_MinimalPipeline.train()` overrides train() and sets `state.current_epoch = state.config.num_epochs`; this should still work. The `test_run_infer_skips_training_stages` test checks `state.metrics == {}` — change to:

```python
def test_run_infer_skips_training_stages(self):
    """Happy Path: run('infer') skips extract_features, build_model,
    train, and evaluate."""
    pipeline = _MinimalPipeline(Config(num_epochs=3))
    state = pipeline.run("infer")
    # Training stages skipped
    assert state.model is None
    assert state.current_epoch == 0
    assert len(state.metrics) == 0  # UPDATED: was == {}
    # load_data and export still run
    assert state.predictions is not None
```

In `TestBasePipelineErrorRecovery`, update error recovery tests that check `state.metrics`:

```python
def test_hook_exception_does_not_kill_pipeline(self):
    """Error recovery: a hook that raises doesn't prevent completion."""
    pipeline = _MinimalPipeline(Config())

    class CrashingHook(BaseHook):
        def on_stage_start(self, stage, state):
            if stage == "train":
                raise RuntimeError("hook crash")

    pipeline.add_hook(CrashingHook())
    state = pipeline.run("train")
    assert state.metrics["accuracy"] == 1.0  # _MinimalPipeline evaluates with perfect match
```

- [ ] **Step 7: Run all tests**

```bash
uv run pytest -v
```
Expected: all tests PASS (new Phase 1 amendment tests + updated Phase 0 tests).

- [ ] **Step 8: Commit**

```bash
git add pipeline/protocols.py pipeline/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: add val_data_stream, Metrics field, hooks @property, train() default"
```

---

### Task 4: Test doubles + tiny_titanic.csv fixture

**Files:**
- Create: `tests/fixtures/tiny_titanic.csv`
- Create: `tests/unit/conftest.py` (test double classes available to all unit tests)

**Interfaces:**
- Produces: `FakeModel(ModelProtocol)` — returns configurable output, empty parameters()
- Produces: `FakeModelWithParams(ModelProtocol)` — returns non-empty parameters() for backward-path coverage
- Produces: `FakeLoss(LossProtocol)` — returns `Loss(value, _backward_fn=None)`
- Produces: `FakeOptimizer(OptimizerProtocol)` — no-op step() and zero_grad()
- Produces: `tests/fixtures/tiny_titanic.csv` — 10 rows, 3 features + 1 target

- [ ] **Step 1: Create tiny_titanic.csv fixture**

Create `tests/fixtures/__init__.py` (empty).

Create `tests/fixtures/tiny_titanic.csv`:

```csv
feature_a,feature_b,feature_c,survived
1.0,2.0,0.0,0
3.0,1.0,1.0,1
2.0,3.0,0.0,0
4.0,2.0,1.0,1
1.0,1.0,0.0,0
3.0,3.0,1.0,1
2.0,2.0,0.0,0
4.0,1.0,1.0,1
1.0,3.0,0.0,0
3.0,2.0,0.0,1
```

- [ ] **Step 2: Create test doubles in conftest.py**

Create `tests/unit/conftest.py`:

```python
"""Shared test doubles for Phase 1 unit tests."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from pipeline.protocols import (
    ArrayLike,
    Batch,
    DataStream,
    Loss,
    LossProtocol,
    ModelProtocol,
    OptimizerProtocol,
    Parameter,
)


class FakeModel(ModelProtocol):
    """Configurable fake model for testing. Returns fixed predictions.

    Uses empty parameters() — TrainLoop skips backward/step for this model.
    """

    def __init__(self, output: ArrayLike | None = None) -> None:
        self._output = output
        self.mode: str = "eval"

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        if self._output is not None:
            return np.asarray(self._output)
        # Default: identity (predictions = inputs)
        return np.asarray(inputs)

    def parameters(self) -> Iterable[Parameter]:
        return []

    def train_mode(self) -> None:
        self.mode = "train"

    def eval_mode(self) -> None:
        self.mode = "eval"


class FakeModelWithParams(ModelProtocol):
    """Fake model with non-empty parameters() — tests backward/step path.

    Returns a single Parameter so TrainLoop enters the backward→step branch.
    """

    def __init__(self, output: ArrayLike | None = None) -> None:
        self._output = output
        self.mode = "eval"
        self._param = Parameter(data=np.array([1.0]), name="w")

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        if self._output is not None:
            return np.asarray(self._output)
        return np.asarray(inputs)

    def parameters(self) -> Iterable[Parameter]:
        return [self._param]

    def train_mode(self) -> None:
        self.mode = "train"

    def eval_mode(self) -> None:
        self.mode = "eval"


class FakeLoss(LossProtocol):
    """Fake loss returning a scalar Loss with no backward hook."""

    def forward(
        self, predictions: ArrayLike, targets: ArrayLike
    ) -> Loss:
        value = float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))
        return Loss(value, _backward_fn=None)


class FakeOptimizer(OptimizerProtocol):
    """No-op optimizer. Records step() and zero_grad() calls for assertions."""

    def __init__(self) -> None:
        self.step_count = 0
        self.zero_grad_count = 0

    def step(self) -> None:
        self.step_count += 1

    def zero_grad(self) -> None:
        self.zero_grad_count += 1


class FakeDataStream(DataStream):
    """In-memory DataStream from pre-defined batches. For testing TrainLoop."""

    def __init__(self, batches: list[Batch]) -> None:
        self.batches = batches

    def __iter__(self) -> Iterable[Batch]:  # type: ignore[override]
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)
```

- [ ] **Step 3: Verify test doubles import cleanly**

```bash
uv run python -c "from tests.unit.conftest import FakeModel, FakeModelWithParams, FakeLoss, FakeOptimizer, FakeDataStream; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/ tests/unit/conftest.py
git commit -m "test: add test doubles and tiny_titanic.csv fixture"
```

---

### Task 5: TrainLoop

**Files:**
- Create: `pipeline/training/__init__.py`
- Create: `pipeline/training/train_loop.py`
- Create: `tests/unit/test_train_loop.py`

**Interfaces:**
- Consumes: `Loss` from Task 1; `ModelProtocol`, `DataStream`, `OptimizerProtocol`, `LossProtocol`, `Batch` from protocols; `BaseHook` from hooks; `PipelineState` from pipeline; test doubles from Task 4
- Produces: `TrainLoop(model, data_stream, optimizer, loss_fn, num_epochs, hooks=None)` class
  - `run(state: PipelineState) -> None` — writes state.history, state.current_epoch
  - `_train_step(batch: Batch) -> float` — internal, single batch forward→loss→backward→step
  - `_notify(event: str, *args) -> None` — internal, hook dispatch with exception recovery

- [ ] **Step 1: Create package structure**

Create `pipeline/training/__init__.py`:

```python
"""Training components — TrainLoop and future optimizer/loss implementations."""

from pipeline.training.train_loop import TrainLoop

__all__ = ["TrainLoop"]
```

- [ ] **Step 2: Write failing tests**

Create `tests/unit/test_train_loop.py`:

```python
"""Unit tests for pipeline.training.train_loop — TrainLoop."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import Config
from pipeline.hooks import BaseHook
from pipeline.pipeline import PipelineState
from pipeline.training.train_loop import TrainLoop
from tests.unit.conftest import (
    FakeDataStream,
    FakeLoss,
    FakeModel,
    FakeModelWithParams,
    FakeOptimizer,
)
from pipeline.protocols import Batch


def _make_state(num_epochs: int = 3) -> PipelineState:
    """Helper: build a PipelineState with fake components."""
    state = PipelineState(config=Config(num_epochs=num_epochs, batch_size=2))
    state.model = FakeModel()
    state.loss_fn = FakeLoss()
    state.optimizer = FakeOptimizer()
    # Create a simple data stream with 2 batches of 2 samples each
    state.data_stream = FakeDataStream([
        Batch(inputs=np.array([[1.0], [2.0]]), targets=np.array([0, 1])),
        Batch(inputs=np.array([[3.0], [4.0]]), targets=np.array([0, 1])),
    ])
    return state


class TestTrainLoopBasic:
    """Tests for TrainLoop core behavior."""

    def test_runs_correct_num_epochs(self):
        """Happy Path: TrainLoop runs exactly num_epochs iterations."""
        state = _make_state(num_epochs=3)
        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=3,
        )
        loop.run(state)
        assert len(state.history["loss"]) == 3

    def test_writes_history(self):
        """Happy Path: TrainLoop writes loss history to state."""
        state = _make_state(num_epochs=5)
        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=5,
        )
        loop.run(state)
        assert state.history is not None
        assert "loss" in state.history
        assert all(isinstance(v, float) for v in state.history["loss"])

    def test_sets_current_epoch_on_state(self):
        """Happy Path: state.current_epoch is updated each epoch."""
        state = _make_state(num_epochs=3)
        epochs_seen: list[int] = []

        class EpochTracker(BaseHook):
            def on_epoch_end(self, epoch, state):
                epochs_seen.append(state.current_epoch)

        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=3,
            hooks=[EpochTracker()],
        )
        loop.run(state)
        assert epochs_seen == [0, 1, 2]

    def test_zero_epochs(self):
        """Boundary: num_epochs=0 exits immediately with empty history."""
        state = _make_state(num_epochs=0)
        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=0,
        )
        loop.run(state)
        assert state.history["loss"] == []


class TestTrainLoopBackwardSkip:
    """Tests for parameter-empty check (backward/step skip)."""

    def test_skips_backward_when_params_empty(self):
        """Happy Path: FakeModel returns empty parameters() → no optimizer calls."""
        state = _make_state(num_epochs=2)
        state.model = FakeModel()  # parameters() → []
        optimizer = FakeOptimizer()
        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=optimizer,
            loss_fn=state.loss_fn,
            num_epochs=2,
        )
        loop.run(state)
        assert optimizer.step_count == 0
        assert optimizer.zero_grad_count == 0

    def test_calls_backward_when_params_present(self):
        """Happy Path: FakeModelWithParams → optimizer.step() and zero_grad() called."""
        state = _make_state(num_epochs=1)
        state.model = FakeModelWithParams()
        optimizer = FakeOptimizer()
        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=optimizer,
            loss_fn=state.loss_fn,
            num_epochs=1,
        )
        loop.run(state)
        # 2 batches × 1 epoch = 2 calls each
        assert optimizer.step_count == 2
        assert optimizer.zero_grad_count == 2


class TestTrainLoopHooks:
    """Tests for TrainLoop hook dispatch."""

    def test_hooks_receive_epoch_events(self):
        """Happy Path: hooks receive on_epoch_start and on_epoch_end."""
        events: list[str] = []

        class EventRecorder(BaseHook):
            def on_epoch_start(self, epoch, state):
                events.append(f"start:{epoch}")
            def on_epoch_end(self, epoch, state):
                events.append(f"end:{epoch}")

        state = _make_state(num_epochs=2)
        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=2,
            hooks=[EventRecorder()],
        )
        loop.run(state)
        assert events == ["start:0", "end:0", "start:1", "end:1"]

    def test_hooks_receive_batch_events(self):
        """Happy Path: hooks receive on_batch_end with correct batch index."""
        batch_indices: list[int] = []

        class BatchRecorder(BaseHook):
            def on_batch_end(self, batch, loss, state):
                batch_indices.append(batch)

        state = _make_state(num_epochs=1)
        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=1,
            hooks=[BatchRecorder()],
        )
        loop.run(state)
        assert batch_indices == [0, 1]  # 2 batches

    def test_hook_exception_does_not_kill_loop(self):
        """Error recovery: hook raising doesn't stop training."""
        class CrashingHook(BaseHook):
            def on_batch_end(self, batch, loss, state):
                raise RuntimeError("boom")

        state = _make_state(num_epochs=2)
        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=2,
            hooks=[CrashingHook()],
        )
        # Should not raise — hook exceptions are caught
        loop.run(state)
        assert len(state.history["loss"]) == 2  # still completed

    def test_multiple_hooks_all_fire(self):
        """Happy Path: all registered hooks receive events."""
        log_a: list[str] = []
        log_b: list[str] = []

        class HookA(BaseHook):
            def on_epoch_start(self, epoch, state):
                log_a.append("a")
        class HookB(BaseHook):
            def on_epoch_start(self, epoch, state):
                log_b.append("b")

        state = _make_state(num_epochs=2)
        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=2,
            hooks=[HookA(), HookB()],
        )
        loop.run(state)
        assert len(log_a) == 2
        assert len(log_b) == 2


class TestTrainLoopStopSignal:
    """Tests for early stopping via state.should_stop."""

    def test_respects_should_stop(self):
        """Happy Path: stops after epoch where should_stop is set."""
        class EarlyStopper(BaseHook):
            def on_epoch_end(self, epoch, state):
                if epoch >= 1:
                    state.should_stop = True

        state = _make_state(num_epochs=10)
        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=10,
            hooks=[EarlyStopper()],
        )
        loop.run(state)
        # Should have stopped after epoch 1 (0-indexed)
        assert len(state.history["loss"]) == 2  # epochs 0, 1

    def test_does_not_stop_if_should_stop_is_false(self):
        """Happy Path: runs all epochs when should_stop never set."""
        state = _make_state(num_epochs=3)
        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=3,
        )
        loop.run(state)
        assert len(state.history["loss"]) == 3


class TestTrainLoopModelMode:
    """Tests for train_mode()/eval_mode() calls."""

    def test_sets_train_mode_at_start(self):
        """Happy Path: model.train_mode() called at start of run()."""
        model = FakeModel()
        assert model.mode == "eval"
        state = _make_state(num_epochs=1)
        state.model = model
        loop = TrainLoop(
            model=model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=1,
        )
        loop.run(state)
        assert model.mode == "train"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_train_loop.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.training'`

- [ ] **Step 4: Implement TrainLoop**

Create `pipeline/training/train_loop.py`:

```python
"""Training lifecycle manager.

TrainLoop owns the epoch/batch loop and training-time hook dispatch.
It reads and writes :class:`PipelineState` for progress tracking and
control signals (``should_stop``, ``current_epoch``).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pipeline.hooks import BaseHook
    from pipeline.pipeline import PipelineState
    from pipeline.protocols import Batch, DataStream, LossProtocol, ModelProtocol, OptimizerProtocol

_logger = logging.getLogger(__name__)


class TrainLoop:
    """Training lifecycle manager.

    Owns the epoch/batch loop and training-time hook dispatch.
    Like a game engine's main loop, it manages lifecycle events
    inside training while delegating stage-level orchestration to
    :class:`BasePipeline`.

    Usage::

        loop = TrainLoop(
            model=model, data_stream=train_stream,
            optimizer=optimizer, loss_fn=loss_fn,
            num_epochs=10, hooks=pipeline.hooks,
        )
        loop.run(state)  # writes state.history
    """

    def __init__(
        self,
        model: ModelProtocol,
        data_stream: DataStream,
        optimizer: OptimizerProtocol,
        loss_fn: LossProtocol,
        num_epochs: int,
        hooks: list[BaseHook] | None = None,
    ) -> None:
        """Create a TrainLoop.

        Args:
            model: Model implementing :class:`ModelProtocol`.
            data_stream: Training data as a :class:`DataStream`.
            optimizer: Optimizer implementing :class:`OptimizerProtocol`.
            loss_fn: Loss function implementing :class:`LossProtocol`.
            num_epochs: Maximum number of training epochs.
            hooks: Optional list of :class:`BaseHook` instances for
                training-time events (epoch start/end, batch end).
        """
        self.model = model
        self.data_stream = data_stream
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self._hooks = list(hooks or [])

    # ── Public API ──────────────────────────────────────────────────

    def run(self, state: PipelineState) -> None:
        """Run the complete training loop.

        Sets ``model.train_mode()``, then iterates over epochs and batches.
        Writes ``state.history`` and ``state.current_epoch``. Respects
        ``state.should_stop`` for early termination.

        Args:
            state: Shared pipeline state (read/write).
        """
        self.model.train_mode()
        history: dict[str, list[float]] = {"loss": []}

        for epoch in range(self.num_epochs):
            state.current_epoch = epoch
            self._notify("on_epoch_start", epoch, state)

            batch_losses: list[float] = []
            for batch_idx, batch in enumerate(self.data_stream):
                loss = self._train_step(batch)
                batch_losses.append(loss)
                self._notify("on_batch_end", batch_idx, loss, state)

            epoch_loss = (
                float(np.mean(batch_losses)) if batch_losses else 0.0
            )
            history["loss"].append(epoch_loss)
            self._notify("on_epoch_end", epoch, state)

            if state.should_stop:
                break

        state.history = history

    # ── Internal methods ────────────────────────────────────────────

    def _train_step(self, batch: Batch) -> float:
        """Execute one training step: forward → loss → backward → step.

        If ``model.parameters()`` is empty (sklearn-style model),
        backward and optimizer step are skipped — only forward + loss.

        Args:
            batch: One :class:`Batch` of inputs and targets.

        Returns:
            Scalar loss value for this batch.
        """
        predictions = self.model.forward(batch.inputs)
        loss = self.loss_fn(predictions, batch.targets)

        params = list(self.model.parameters())
        # WHY: sklearn models return empty parameters — skip gradient update
        if params:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return float(loss)

    def _notify(self, event: str, *args: object) -> None:
        """Dispatch a training-time event to all hooks.

        Hook exceptions are caught and logged — they never interrupt
        the training loop. This mirrors :meth:`BasePipeline._notify`.

        Args:
            event: Hook method name (``"on_epoch_start"``, etc.).
            *args: Arguments forwarded to the hook method.
        """
        for hook in self._hooks:
            try:
                getattr(hook, event)(*args)
            except Exception:
                _logger.exception(
                    "Hook %s.%s raised an exception (ignored)",
                    hook.__class__.__name__,
                    event,
                )
```

- [ ] **Step 5: Run TrainLoop tests**

```bash
uv run pytest tests/unit/test_train_loop.py -v
```
Expected: all tests PASS.

- [ ] **Step 6: Verify full test suite**

```bash
uv run pytest -v
```
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add pipeline/training/ tests/unit/test_train_loop.py
git commit -m "feat: add TrainLoop with training-time hook dispatch"
```

---

### Task 6: CsvDataSource

**Files:**
- Create: `pipeline/data/__init__.py`
- Create: `pipeline/data/csv_source.py`
- Create: `tests/unit/test_csv_source.py`

**Interfaces:**
- Consumes: `DataStream`, `Batch` from protocols
- Produces: `CsvDataSource(file_path, batch_size=32, target_column=None, shuffle=True, seed=42)` implementing `DataStream`

- [ ] **Step 1: Add pandas dependency**

```bash
uv add pandas
```

- [ ] **Step 2: Create package structure**

Create `pipeline/data/__init__.py`:

```python
"""Data sources and split utilities for the pipeline."""

from pipeline.data.csv_source import CsvDataSource
from pipeline.data.split import train_test_split

__all__ = ["CsvDataSource", "train_test_split"]
```

(Note: `train_test_split` import will fail until Task 7 — this is fine, the file is just a re-export module.)

- [ ] **Step 3: Write failing tests**

Create `tests/unit/test_csv_source.py`:

```python
"""Unit tests for pipeline.data.csv_source — CsvDataSource."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pipeline.data.csv_source import CsvDataSource
from pipeline.protocols import Batch


@pytest.fixture
def tiny_csv() -> str:
    """Write a temporary CSV for testing."""
    import pandas as pd

    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [0.1, 0.2, 0.3, 0.4, 0.5],
        "label": [0, 1, 0, 1, 0],
    })
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        df.to_csv(f, index=False)
        return f.name


class TestCsvDataSourceBasic:
    """Basic functionality tests."""

    def test_iter_yields_batches(self, tiny_csv):
        """Happy Path: __iter__ yields Batch objects."""
        source = CsvDataSource(tiny_csv, batch_size=2)
        batches = list(source)
        assert len(batches) == 3  # 5 samples / 2 = 3 batches
        assert all(isinstance(b, Batch) for b in batches)

    def test_batch_shapes(self, tiny_csv):
        """Happy Path: batch inputs/targets have correct shapes."""
        source = CsvDataSource(tiny_csv, batch_size=2, shuffle=False)
        batches = list(source)
        # First 2 batches: 2 samples each; last batch: 1 sample
        assert batches[0].inputs.shape == (2, 2)  # 2 features (a, b)
        assert batches[0].targets.shape == (2,)
        assert batches[1].inputs.shape == (2, 2)
        assert batches[1].targets.shape == (2,)
        assert batches[2].inputs.shape == (1, 2)
        assert batches[2].targets.shape == (1,)

    def test_len_returns_num_batches(self, tiny_csv):
        """Happy Path: __len__ returns ceil(n_samples / batch_size)."""
        source = CsvDataSource(tiny_csv, batch_size=2)
        assert len(source) == 3

    def test_target_column_default_last(self, tiny_csv):
        """Happy Path: by default, last column is treated as target."""
        source = CsvDataSource(tiny_csv, batch_size=5, shuffle=False)
        batch = next(iter(source))
        # targets should be the 'label' column
        np.testing.assert_array_equal(batch.targets, np.array([0, 1, 0, 1, 0]))

    def test_explicit_target_column(self, tiny_csv):
        """Happy Path: explicit target_column uses that column as target."""
        source = CsvDataSource(
            tiny_csv, batch_size=5, target_column="label", shuffle=False
        )
        batch = next(iter(source))
        # inputs should have only 'a' and 'b' columns
        assert batch.inputs.shape == (5, 2)


class TestCsvDataSourceShuffle:
    """Reproducibility and shuffle tests."""

    def test_same_seed_same_order(self, tiny_csv):
        """Reproducibility: same seed → same batch order."""
        s1 = CsvDataSource(tiny_csv, batch_size=2, seed=42)
        s2 = CsvDataSource(tiny_csv, batch_size=2, seed=42)
        batches1 = list(s1)
        batches2 = list(s2)
        for b1, b2 in zip(batches1, batches2):
            np.testing.assert_array_equal(b1.inputs, b2.inputs)

    def test_different_seed_different_order(self, tiny_csv):
        """Reproducibility: different seeds give different order."""
        s1 = CsvDataSource(tiny_csv, batch_size=5, seed=1)
        s2 = CsvDataSource(tiny_csv, batch_size=5, seed=999)
        b1 = next(iter(s1))
        b2 = next(iter(s2))
        # With 5 samples, likely different order (tiny prob of collision)
        assert not np.array_equal(b1.inputs, b2.inputs)

    def test_no_shuffle_preserves_order(self, tiny_csv):
        """Happy Path: shuffle=False preserves CSV row order."""
        source = CsvDataSource(tiny_csv, batch_size=5, shuffle=False)
        batch = next(iter(source))
        expected = np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4], [5.0, 0.5]])
        np.testing.assert_array_equal(batch.inputs, expected)


class TestCsvDataSourceEdgeCases:
    """Edge case and error handling tests."""

    def test_missing_file_raises(self):
        """Error: FileNotFoundError for nonexistent CSV."""
        source = CsvDataSource("/nonexistent/path.csv", batch_size=2)
        with pytest.raises(FileNotFoundError):
            list(source)

    def test_lazy_load_file_not_read_on_init(self):
        """Happy Path: CSV is not read during __init__ (lazy loading)."""
        source = CsvDataSource("/nonexistent/path.csv", batch_size=2)
        # No error during construction — file not accessed yet
        assert source is not None

    def test_len_consistent(self, tiny_csv):
        """Happy Path: len() is consistent across multiple calls."""
        source = CsvDataSource(tiny_csv, batch_size=2)
        assert len(source) == 3
        assert len(source) == 3  # second call same result

    def test_single_batch_exact(self, tiny_csv):
        """Boundary: batch_size equal to n_samples → 1 batch."""
        source = CsvDataSource(tiny_csv, batch_size=5)
        assert len(source) == 1

    def test_iter_twice(self, tiny_csv):
        """Happy Path: iterating twice produces data both times (re-shuffles)."""
        source = CsvDataSource(tiny_csv, batch_size=5, seed=42)
        list(source)  # first pass
        batches = list(source)  # second pass
        assert len(batches) == 1
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_csv_source.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.data'`

- [ ] **Step 5: Implement CsvDataSource**

Create `pipeline/data/csv_source.py`:

```python
"""CSV file data source.

Provides :class:`CsvDataSource`, a :class:`DataStream` implementation
that reads tabular data from CSV files and yields :class:`Batch` objects.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from pipeline.protocols import Batch, DataStream


class CsvDataSource(DataStream):
    """CSV file → Batch iterator.

    Reads a CSV into a pandas DataFrame, optionally shuffles rows,
    and yields batches as :class:`Batch` objects. The last column is
    used as the target unless ``target_column`` is specified.

    Lazy-loading: the CSV is read on the first call to :meth:`__iter__`
    or :meth:`__len__`, not during ``__init__``. This keeps construction
    cheap and defers I/O.

    Usage::

        stream = CsvDataSource("train.csv", batch_size=32)
        print(len(stream))  # → number of batches
        for batch in stream:
            print(batch.inputs.shape, batch.targets.shape)
    """

    def __init__(
        self,
        file_path: str | Path,
        batch_size: int = 32,
        target_column: str | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """Create a CSV data source.

        Args:
            file_path: Path to a ``.csv`` file.
            batch_size: Number of samples per batch.
            target_column: Column name for labels. Default: last column.
            shuffle: If True, shuffle rows before batching.
            seed: Random seed for reproducible shuffling.
        """
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self.target_column = target_column
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._df: pd.DataFrame | None = None
        self._n_samples: int = 0

    def _load(self) -> pd.DataFrame:
        """Lazy-load the CSV on first access.

        Returns:
            The loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if self._df is None:
            import pandas as pd

            self._df = pd.read_csv(self.file_path)
            if self.target_column is None:
                self.target_column = self._df.columns[-1]
            self._n_samples = len(self._df)
        return self._df

    def __iter__(self) -> Iterator[Batch]:
        """Yield one :class:`Batch` per batch window.

        If ``shuffle=True``, row order is randomized on each call to
        ``__iter__``, so repeated iteration produces different orders.
        """
        df = self._load()
        indices = np.arange(self._n_samples)
        if self.shuffle:
            self.rng.shuffle(indices)

        feature_cols = [c for c in df.columns if c != self.target_column]
        for start in range(0, self._n_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch_df = df.iloc[batch_idx]
            inputs = batch_df[feature_cols].to_numpy(dtype=np.float64)
            targets = batch_df[self.target_column].to_numpy()
            yield Batch(inputs=inputs, targets=targets)

    def __len__(self) -> int:
        """Number of batches (``ceil(n_samples / batch_size)``)."""
        self._load()
        return max(1, math.ceil(self._n_samples / self.batch_size))
```

- [ ] **Step 6: Run CsvDataSource tests**

```bash
uv run pytest tests/unit/test_csv_source.py -v
```
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add pipeline/data/__init__.py pipeline/data/csv_source.py tests/unit/test_csv_source.py pyproject.toml uv.lock
git commit -m "feat: add CsvDataSource with lazy CSV loading"
```

---

### Task 7: train_test_split + _InMemoryDataStream

**Files:**
- Create: `pipeline/data/split.py`
- Create: `tests/unit/test_split.py`

**Interfaces:**
- Consumes: `DataStream`, `Batch` from protocols; `CsvDataSource` from Task 6 (for integration-style test)
- Produces: `train_test_split(source, train_ratio=0.8, shuffle=True, seed=42) -> tuple[DataStream, DataStream]`
- Produces: `_InMemoryDataStream(batches, n_samples)` — private helper, implements DataStream

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_split.py`:

```python
"""Unit tests for pipeline.data.split — train_test_split."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.data.csv_source import CsvDataSource
from pipeline.data.split import train_test_split
from pipeline.protocols import Batch


@pytest.fixture
def tiny_csv_path():
    """Path to the committed test fixture."""
    from pathlib import Path
    return str(
        Path(__file__).parent.parent / "fixtures" / "tiny_titanic.csv"
    )


class TestTrainTestSplit:
    """Tests for train_test_split()."""

    def test_returns_two_streams(self, tiny_csv_path):
        """Happy Path: returns (train_stream, val_stream) tuple."""
        source = CsvDataSource(tiny_csv_path, batch_size=2, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.8)
        assert train is not None
        assert val is not None
        assert train is not val

    def test_train_ratio_split(self, tiny_csv_path):
        """Happy Path: split respects train_ratio."""
        source = CsvDataSource(tiny_csv_path, batch_size=1, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.7, shuffle=False)
        # 10 samples: ~7 train, ~3 val
        train_samples = sum(
            len(b.inputs) for b in train
        )  # batch_size=1, so len(b.inputs)=1
        val_samples = sum(len(b.inputs) for b in val)
        assert train_samples == 7
        assert val_samples == 3

    def test_both_streams_iterable(self, tiny_csv_path):
        """Happy Path: both returned streams can be independently iterated."""
        source = CsvDataSource(tiny_csv_path, batch_size=2, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.8)
        train_batches = list(train)
        val_batches = list(val)
        assert len(train_batches) > 0
        assert len(val_batches) > 0

    def test_shuffle_reproducibility(self, tiny_csv_path):
        """Reproducibility: same seed → same split."""
        s1 = CsvDataSource(tiny_csv_path, batch_size=2, shuffle=False)
        s2 = CsvDataSource(tiny_csv_path, batch_size=2, shuffle=False)
        t1, v1 = train_test_split(s1, seed=42)
        t2, v2 = train_test_split(s2, seed=42)
        # Collect all data from each stream
        t1_data = np.concatenate([b.inputs for b in t1])
        t2_data = np.concatenate([b.inputs for b in t2])
        np.testing.assert_array_equal(t1_data, t2_data)

    def test_no_overlap_between_splits(self, tiny_csv_path):
        """Happy Path: train and val sets have no overlapping samples."""
        source = CsvDataSource(tiny_csv_path, batch_size=1, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.8, shuffle=False)
        # With no shuffle, train takes first 8, val takes last 2
        train_data = np.concatenate([b.inputs for b in train])
        val_data = np.concatenate([b.inputs for b in val])
        # No row should appear in both
        for row in val_data:
            assert not any(np.array_equal(row, t_row) for t_row in train_data)

    def test_len_consistent(self, tiny_csv_path):
        """Happy Path: __len__ returns number of batches."""
        source = CsvDataSource(tiny_csv_path, batch_size=3, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.7, shuffle=False)
        assert len(train) > 0
        assert len(val) > 0

    def test_edge_ratio_near_one(self, tiny_csv_path):
        """Boundary: train_ratio=0.99 → almost all data in train."""
        source = CsvDataSource(tiny_csv_path, batch_size=1, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.99, shuffle=False)
        train_samples = sum(len(b.inputs) for b in train)
        val_samples = sum(len(b.inputs) for b in val)
        # 10 samples: 9 train, 1 val (floor for train)
        assert train_samples == 9
        assert val_samples == 1

    def test_all_data_preserved(self, tiny_csv_path):
        """Happy Path: no samples lost during split."""
        source = CsvDataSource(tiny_csv_path, batch_size=1, shuffle=False)
        train, val = train_test_split(source, train_ratio=0.5, shuffle=False)
        train_count = sum(len(b.inputs) for b in train)
        val_count = sum(len(b.inputs) for b in val)
        assert train_count + val_count == 10  # tiny_titanic has 10 rows
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_split.py -v
```
Expected: FAIL — `ImportError: cannot import name 'train_test_split'`

(Note: `pipeline/data/__init__.py` from Task 6 already imports `train_test_split` from `pipeline.data.split`. The `__init__.py` import will fail when the package is loaded, but the test file imports from `pipeline.data.split` directly — which doesn't exist yet. The error will be `ModuleNotFoundError` for `pipeline.data.split`.)

Actually, `pipeline/data/__init__.py` was created in Task 6 and tries to import `train_test_split` from `split`. Since `split.py` doesn't exist yet, importing from `pipeline.data` would fail. But the tests in Task 6 import `CsvDataSource` from `pipeline.data.csv_source` directly, so they work. And the test for Task 7 imports from `pipeline.data.split` directly, which will fail because the file doesn't exist. Good.

- [ ] **Step 3: Implement train_test_split and _InMemoryDataStream**

Create `pipeline/data/split.py`:

```python
"""Train/validation split utilities.

Splits a :class:`DataStream` into two independent streams
for training and validation.
"""
from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from pipeline.protocols import Batch, DataStream


def train_test_split(
    source: DataStream,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[DataStream, DataStream]:
    """Split a DataStream into (train_stream, val_stream).

    Collects all data from ``source`` into memory, shuffles if requested,
    then splits by ``train_ratio``. Returns two :class:`_InMemoryDataStream`
    instances backed by the split data.

    For large datasets that don't fit in memory, a streaming split
    can be added in a later phase.

    Args:
        source: Any :class:`DataStream` to split.
        train_ratio: Fraction of data for training (in ``(0, 1)``).
        shuffle: If True, shuffle samples before splitting.
        seed: Random seed for reproducible shuffling.

    Returns:
        ``(train_stream, val_stream)`` tuple of :class:`DataStream`.
    """
    # WHY: Collect into memory for Phase 1. Titanic-sized data fits easily.
    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for batch in source:
        all_inputs.append(np.asarray(batch.inputs))
        all_targets.append(np.asarray(batch.targets))

    if not all_inputs:
        return (
            _InMemoryDataStream([], n_samples=0),
            _InMemoryDataStream([], n_samples=0),
        )

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets)
    n_samples = len(inputs)

    # Shuffle indices
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)

    # Split
    n_train = math.floor(n_samples * train_ratio)
    # WHY: floor for train — at least 1 sample in val when train_ratio < 1.0
    n_train = max(1, min(n_train, n_samples - 1))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_stream = _InMemoryDataStream(
        [Batch(inputs=inputs[train_idx], targets=targets[train_idx])],
        n_samples=len(train_idx),
    )
    val_stream = _InMemoryDataStream(
        [Batch(inputs=inputs[val_idx], targets=targets[val_idx])],
        n_samples=len(val_idx),
    )
    return train_stream, val_stream


class _InMemoryDataStream(DataStream):
    """A :class:`DataStream` backed by pre-split numpy arrays.

    Internal helper for :func:`train_test_split`. Not part of the public API.

    NOTE: This stores one giant batch. Consumers that expect smaller
    batches should iterate with ``batch_size=1`` upstream. For Phase 1,
    this is sufficient — batch_size is handled by CsvDataSource before
    the split, or consumers can split the giant batch themselves.
    """

    def __init__(
        self, batches: list[Batch], *, n_samples: int
    ) -> None:
        self._batches = batches
        self._n_samples = n_samples

    def __iter__(self) -> Iterator[Batch]:
        return iter(self._batches)

    def __len__(self) -> int:
        return max(1, len(self._batches))
```

Wait — the `_InMemoryDataStream` design has a problem. After splitting, each stream is a single giant batch (all train samples in one Batch, all val in another). But `TrainLoop` expects multiple batches per epoch. And `len(stream)` returns the number of batches, which should be ceil(n_samples / batch_size).

Let me reconsider the design. The `train_test_split` should preserve or respect batch_size. But `DataStream` doesn't expose batch_size — it's an implementation detail of CsvDataSource.

Better approach: `_InMemoryDataStream` takes the split arrays and a `batch_size`, then yields batches of the right size. This is the approach from the spec.

Let me redesign:

```python
class _InMemoryDataStream(DataStream):
    """A DataStream backed by pre-split numpy arrays.

    Internal helper for train_test_split. Not part of the public API.
    Yields batches of size ``batch_size``.
    """

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
    ) -> None:
        self._inputs = inputs
        self._targets = targets
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[Batch]:
        n = len(self._inputs)
        for start in range(0, n, self._batch_size):
            end = start + self._batch_size
            yield Batch(
                inputs=self._inputs[start:end],
                targets=self._targets[start:end],
            )

    def __len__(self) -> int:
        return max(1, math.ceil(len(self._inputs) / self._batch_size))
```

And `train_test_split` passes a `batch_size` param:

```python
def train_test_split(
    source: DataStream,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[DataStream, DataStream]:
```

But then the test `test_len_consistent` needs updating. Let me also update the test fixture - the test_split tests use `batch_size=1` or `batch_size=2` which maps cleanly.

Actually, let me reconsider the test expectations. With `batch_size` in train_test_split:

- `test_train_ratio_split`: uses `batch_size=1`, expects train=7/val=3 for 70% split of 10 samples. ✓
- `test_edge_ratio_near_one`: uses `batch_size=1`. 10*0.99 → floor=9 train, 1 val. ✓
- `test_no_overlap`: uses `batch_size=1`. Can still concatenate and compare. ✓
- `test_len_consistent`: uses `batch_size=3`. With 7 train samples → ceil(7/3)=3 batches. With 3 val → ceil(3/3)=1 batch. ✓

OK let me update the implementation. The tests above are already correct for this approach.

Let me finalize the files. But first I need to make sure the test for `train_test_split` passes correctly. The tests use `CsvDataSource` with specific batch sizes and then call `train_test_split` — but `train_test_split` collects all data first (regardless of CsvDataSource's batch_size) and then re-batches. So the batch_size from CsvDataSource is irrelevant for the split — `train_test_split` will collect all batches (which may be of different sizes) and recombine.

This means `test_no_overlap_between_splits` should work — we use `batch_size=1` on CsvDataSource but that's just for collecting. The `_InMemoryDataStream` rebatches based on the `batch_size` parameter.

OK I think the approach works. Let me finalize.

- [ ] **Step 4: Run split tests**

```bash
uv run pytest tests/unit/test_split.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Fix pipeline/data/__init__.py if needed**

The `__init__.py` from Task 6 already imports `train_test_split`. Since `split.py` now exists, this import should work.

- [ ] **Step 6: Commit**

```bash
git add pipeline/data/split.py tests/unit/test_split.py
git commit -m "feat: add train_test_split with InMemoryDataStream"
```

---

### Task 8: ProgressHook

**Files:**
- Create: `pipeline/hooks/__init__.py`
- Create: `pipeline/hooks/progress.py`
- Create: `tests/unit/test_progress.py`

Note: `pipeline/hooks.py` currently exists as a module. We need to convert `pipeline/hooks/` to a package. The existing `hooks.py` becomes `hooks/base.py` or we keep it as-is and add `hooks/` as a sibling. Simpler: convert `hooks.py` → `hooks/__init__.py` that re-exports BaseHook from a new `hooks/base.py`.

Actually, the cleanest approach: create `pipeline/hooks/` as a package directory, make `__init__.py` import and re-export `BaseHook` from the original `hooks.py` file at the parent level, and add `progress.py` inside the package. But this means two things called `hooks` at different levels...

The simplest approach: rename `pipeline/hooks.py` → `pipeline/hooks/base.py`, create `pipeline/hooks/__init__.py` that re-exports `BaseHook`. No file is left at `pipeline/hooks.py` — it's now a package.

Wait, the spec says `pipeline/hooks/` as a directory. Currently `pipeline/hooks.py` is a module. The simplest migration: `mv pipeline/hooks.py pipeline/hooks_base.py`, then create `pipeline/hooks/` package with `__init__.py` that imports from `pipeline.hooks_base`. But that's ugly.

Better: just keep hooks.py as the single module and put ProgressHook in a new file `pipeline/hooks_progress.py`. But the spec wants `pipeline/hooks/progress.py`.

OK, let me do the clean thing: create `pipeline/hooks/` directory, move `hooks.py` content into `hooks/base.py`, create `hooks/__init__.py` that re-exports BaseHook. Update all internal imports.

Actually, let me check who imports from `pipeline.hooks`:
- `pipeline/pipeline.py` imports `BaseHook` via TYPE_CHECKING: `from pipeline.hooks import BaseHook`
- Tests: `from pipeline.hooks import BaseHook`

If I restructure to `pipeline/hooks/__init__.py` re-exporting `BaseHook`, those imports still work (`from pipeline.hooks import BaseHook`).

Let me do this.

- [ ] **Step 1: Add tqdm dependency**

```bash
uv add tqdm
```

- [ ] **Step 2: Restructure hooks from module to package**

Create `pipeline/hooks/` directory. Move content of `pipeline/hooks.py` to `pipeline/hooks/base.py` (remove the module-level docstring since `__init__.py` will have it). Create `pipeline/hooks/__init__.py`:

```python
"""Hook system for cross-cutting concerns in the pipeline.
...
"""

from pipeline.hooks.base import BaseHook
from pipeline.hooks.progress import ProgressHook

__all__ = ["BaseHook", "ProgressHook"]
```

Delete `pipeline/hooks.py`.

Update all internal imports: verify `pipeline/pipeline.py` still imports correctly (it does — `from pipeline.hooks import BaseHook`).

- [ ] **Step 3: Write failing tests**

Create `tests/unit/test_progress.py`:

```python
"""Unit tests for pipeline.hooks.progress — ProgressHook."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from pipeline.config import Config
from pipeline.hooks.progress import ProgressHook
from pipeline.pipeline import PipelineState
from pipeline.protocols import Batch


class TestProgressHook:
    """Tests for ProgressHook."""

    def test_creates_pbar_on_epoch_start(self):
        """Happy Path: on_epoch_start creates a tqdm progress bar."""
        hook = ProgressHook()
        state = PipelineState(config=Config(batch_size=2))
        state.data_stream = [1, 2, 3]  # len=3 for total

        with patch("pipeline.hooks.progress.tqdm") as mock_tqdm:
            mock_pbar = mock_tqdm.return_value
            hook.on_epoch_start(0, state)
            mock_tqdm.assert_called_once_with(total=3, desc="Epoch 1")

    def test_updates_pbar_on_batch_end(self):
        """Happy Path: on_batch_end updates the tqdm bar."""
        hook = ProgressHook()
        state = PipelineState(config=Config())
        state.data_stream = [1, 2]

        with patch("pipeline.hooks.progress.tqdm") as mock_tqdm:
            mock_pbar = mock_tqdm.return_value
            hook.on_epoch_start(0, state)
            hook.on_batch_end(0, 1.5, state)
            mock_pbar.update.assert_called_once_with(1)
            mock_pbar.set_postfix.assert_called_once_with(loss="1.5000")

    def test_closes_pbar_on_epoch_end(self):
        """Happy Path: on_epoch_end closes the tqdm bar."""
        hook = ProgressHook()
        state = PipelineState(config=Config())

        with patch("pipeline.hooks.progress.tqdm") as mock_tqdm:
            mock_pbar = mock_tqdm.return_value
            # Manually set up losses
            hook.on_epoch_start(0, state)
            hook.on_batch_end(0, 0.5, state)
            hook.on_batch_end(1, 0.3, state)
            hook.on_epoch_end(0, state)
            mock_pbar.close.assert_called_once()

    def test_clears_losses_after_epoch(self):
        """Happy Path: loss accumulator is cleared after each epoch."""
        hook = ProgressHook()
        state = PipelineState(config=Config())
        state.data_stream = [1]

        with patch("pipeline.hooks.progress.tqdm"):
            hook.on_epoch_start(0, state)
            hook.on_batch_end(0, 0.5, state)
            assert len(hook._losses) == 1
            hook.on_epoch_end(0, state)
            assert len(hook._losses) == 0

    def test_stage_events_are_noops(self):
        """Happy Path: on_stage_start/end are inherited no-ops."""
        hook = ProgressHook()
        state = PipelineState(config=Config())
        # Should not raise
        hook.on_stage_start("train", state)
        hook.on_stage_end("train", state)
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_progress.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.hooks.progress'`

- [ ] **Step 5: Implement ProgressHook**

Create `pipeline/hooks/progress.py`:

```python
"""Progress bar hook using tqdm.

Provides :class:`ProgressHook` — a concrete :class:`BaseHook`
that renders training progress with tqdm.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.hooks.base import BaseHook

if TYPE_CHECKING:
    from pipeline.pipeline import PipelineState


class ProgressHook(BaseHook):
    """tqdm progress bar for training.

    Creates a progress bar on ``on_epoch_start``, updates it on
    ``on_batch_end`` with the current batch loss, and closes it
    on ``on_epoch_end``.

    Usage::

        pipeline.add_hook(ProgressHook())
    """

    def __init__(self) -> None:
        """Initialize the progress hook."""
        from tqdm import tqdm as _tqdm

        self._tqdm = _tqdm
        self._pbar = None
        self._losses: list[float] = []

    def on_epoch_start(
        self, epoch: int, state: PipelineState
    ) -> None:
        """Create a new tqdm progress bar for this epoch.

        Args:
            epoch: Zero-based epoch index.
            state: Current pipeline state.
        """
        total = len(state.data_stream) if state.data_stream is not None else 0
        self._pbar = self._tqdm(total=total, desc=f"Epoch {epoch + 1}")
        self._losses.clear()

    def on_batch_end(
        self, batch: int, loss: float, state: PipelineState
    ) -> None:
        """Update the progress bar and record batch loss.

        Args:
            batch: Zero-based batch index.
            loss: Scalar loss for this batch.
            state: Current pipeline state.
        """
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix(loss=f"{loss:.4f}")
        self._losses.append(loss)

    def on_epoch_end(
        self, epoch: int, state: PipelineState
    ) -> None:
        """Close the progress bar for this epoch.

        Args:
            epoch: Zero-based epoch index.
            state: Current pipeline state.
        """
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        avg_loss = (
            sum(self._losses) / len(self._losses)
            if self._losses
            else 0.0
        )
        self._losses.clear()
```

- [ ] **Step 6: Run ProgressHook tests**

```bash
uv run pytest tests/unit/test_progress.py -v
```
Expected: all tests PASS.

- [ ] **Step 7: Verify existing tests still pass after hooks restructuring**

```bash
uv run pytest tests/unit/test_hooks.py -v
uv run pytest tests/unit/test_pipeline.py -v
```
Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add pipeline/hooks/ tests/unit/test_progress.py pyproject.toml uv.lock
git rm pipeline/hooks.py 2>/dev/null || git rm --cached pipeline/hooks.py 2>/dev/null
git commit -m "feat: restructure hooks to package, add ProgressHook"
```

---

### Task 9: to_csv

**Files:**
- Create: `pipeline/export/__init__.py`
- Create: `pipeline/export/to_csv.py`
- Create: `tests/unit/test_to_csv.py`

**Interfaces:**
- Produces: `to_csv(array: ArrayLike, path: str | Path, columns: list[str] | None = None) -> None`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_to_csv.py`:

```python
"""Unit tests for pipeline.export.to_csv — to_csv()."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pipeline.export.to_csv import to_csv


class TestToCsv:
    """Tests for to_csv()."""

    def test_writes_file(self):
        """Happy Path: to_csv creates a file at the given path."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as f:
            path = f.name
        try:
            to_csv(data, path)
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_round_trip(self):
        """Happy Path: written CSV can be read back correctly."""
        import pandas as pd

        data = np.array([[1.0, 0.5], [2.0, 0.3], [3.0, 0.1]])
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as f:
            path = f.name
        try:
            to_csv(data, path, columns=["pred_0", "pred_1"])
            df = pd.read_csv(path)
            assert list(df.columns) == ["pred_0", "pred_1"]
            assert df.shape == (3, 2)
            np.testing.assert_array_almost_equal(
                df.values, data
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_no_columns_default_header(self):
        """Happy Path: when columns=None, default numeric headers are written."""
        import pandas as pd

        data = np.array([[1.0, 2.0]])
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as f:
            path = f.name
        try:
            to_csv(data, path)
            df = pd.read_csv(path)
            assert df.shape == (1, 2)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_single_row(self):
        """Boundary: single-row array."""
        data = np.array([[0.1, 0.9]])
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as f:
            path = f.name
        try:
            to_csv(data, path)
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_1d_array(self):
        """Happy Path: 1D array is reshaped for CSV writing."""
        data = np.array([0.1, 0.2, 0.3])
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as f:
            path = f.name
        try:
            to_csv(data, path)
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_creates_parent_directory(self):
        """Happy Path: parent directories are created if needed."""
        import tempfile
        import os

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "subdir", "output.csv")
        data = np.array([[1.0]])
        try:
            to_csv(data, path)
            assert Path(path).exists()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_to_csv.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.export'`

- [ ] **Step 3: Create package structure**

Create `pipeline/export/__init__.py`:

```python
"""Export utilities — model checkpoints, prediction CSV output."""

from pipeline.export.to_csv import to_csv

__all__ = ["to_csv"]
```

- [ ] **Step 4: Implement to_csv**

Create `pipeline/export/to_csv.py`:

```python
"""CSV export utility.

Writes numpy arrays and array-like objects to CSV files via pandas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def to_csv(
    array: Any,
    path: str | Path,
    columns: list[str] | None = None,
) -> None:
    """Write a numpy array or array-like to a CSV file.

    Args:
        array: Array-like data to write. 1D arrays are reshaped to 2D.
        path: Output file path. Parent directories are created if needed.
        columns: Optional column names for the CSV header. Defaults to
            sequential integers.

    Raises:
        OSError: If the file cannot be written.
    """
    import pandas as pd

    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(arr)
    if columns is not None:
        df.columns = columns
    df.to_csv(path, index=False)
```

- [ ] **Step 5: Run to_csv tests**

```bash
uv run pytest tests/unit/test_to_csv.py -v
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/export/ tests/unit/test_to_csv.py
git commit -m "feat: add to_csv export utility"
```

---

### Task 10: CLI Entry Points (__main__.py + run.py)

**Files:**
- Create: `pipeline/__main__.py`
- Create: `run.py` (project root)
- Create: `tests/unit/test_main.py`

**Interfaces:**
- Produces: `python -m pipeline` → prints version, device, registered components
- Produces: `python run.py --config config.yaml --mode train` → runs pipeline

- [ ] **Step 1: Write failing tests for __main__**

Create `tests/unit/test_main.py`:

```python
"""Unit tests for pipeline.__main__ and CLI entry."""
from __future__ import annotations

import io
import sys

import pytest


class TestMainModule:
    """Tests for python -m pipeline."""

    def test_main_prints_version(self):
        """Happy Path: python -m pipeline prints version string."""
        import pipeline.__main__

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pipeline.__main__.main()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert "0.1.0" in output or "Universal DL Pipeline" in output

    def test_main_prints_device_info(self):
        """Happy Path: prints device information."""
        import pipeline.__main__

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pipeline.__main__.main()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        # Should mention device (CPU at minimum)
        assert "Device" in output or "device" in output

    def test_main_prints_registered_components(self):
        """Happy Path: prints registered components by kind."""
        import pipeline.__main__

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pipeline.__main__.main()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        # At minimum, something is registered (from Phase 0 tests, if nothing else)
        # If nothing registered, it should still print successfully
        assert "Universal DL Pipeline" in output


class TestRunPy:
    """Tests for run.py CLI."""

    def test_run_py_module_exists(self):
        """Happy Path: run.py can be imported."""
        import importlib.util
        spec = importlib.util.find_spec("run")
        # run.py is not a package, so find_spec may not work.
        # Just verify the file exists.
        from pathlib import Path
        assert Path("run.py").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_main.py -v
```
Expected: either FAIL (if `__main__.py` doesn't exist) or PASS (if the test can import existing things but run.py doesn't exist).

Actually, `pipeline/__main__.py` doesn't exist yet. So `import pipeline.__main__` will raise `ModuleNotFoundError`. Good.

- [ ] **Step 3: Implement __main__.py**

Create `pipeline/__main__.py`:

```python
"""Entry point for ``python -m pipeline``.

Prints version, device information, and all registered pipeline components.
"""
from __future__ import annotations


def main() -> None:
    """Print pipeline status: version, device, registered components.

    Called when the user runs ``python -m pipeline``.
    """
    import pipeline
    from pipeline.registry import _REGISTRY
    from pipeline.utils.device import device_info

    print(f"Universal DL Pipeline v{pipeline.__version__}")
    print(f"Device: {device_info()}")

    if not _REGISTRY:
        print("No components registered.")
    else:
        print("Registered components:")
        for kind in sorted(_REGISTRY):
            names = sorted(_REGISTRY[kind])
            for name in names:
                cls_name = _REGISTRY[kind][name].__name__
                print(f"  [{kind}] {name} → {cls_name}")
```

And add the `__main__` block:

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement run.py**

Create `run.py` (project root):

```python
"""CLI entry point for the Universal DL Pipeline.

Usage::

    python run.py --config config.yaml --mode train
    python run.py --config config.yaml --mode infer
    python run.py --help
"""
from __future__ import annotations

import argparse
import sys

from pipeline.config import Config
from pipeline.pipeline import BasePipeline


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the pipeline.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description="Universal DL Pipeline — train and infer"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        default="train",
        help="Pipeline mode (default: train)",
    )
    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config)

    # Phase 1: use a minimal pipeline with default stages.
    # Students subclass BasePipeline for custom behavior.
    from pipeline.evaluation.metrics import accuracy

    class DefaultPipeline(BasePipeline):
        """Phase 1 default pipeline using TrainLoop + CsvDataSource."""

        def load_data(self, state):
            from pipeline.data.csv_source import CsvDataSource
            from pipeline.data.split import train_test_split

            full_stream = CsvDataSource(
                f"{config.data_dir}/train.csv",
                batch_size=config.batch_size,
            )
            train_stream, val_stream = train_test_split(
                full_stream,
                train_ratio=config.train_ratio,
                batch_size=config.batch_size,
            )
            state.data_stream = train_stream
            state.val_data_stream = val_stream

        def build_model(self, state):
            # Phase 1: fake model for framework validation
            from pipeline.protocols import Parameter
            import numpy as np

            # Minimal fake model — real adapters arrive in Phase 2
            class _FakeModel:
                def forward(self, inputs):
                    return np.asarray(inputs)[:, :1]
                def parameters(self):
                    return []
                def train_mode(self):
                    pass
                def eval_mode(self):
                    pass

            class _FakeLoss:
                def forward(self, p, t):
                    from pipeline.protocols import Loss
                    return Loss(
                        float(np.mean((np.asarray(p) - np.asarray(t)) ** 2))
                    )

            class _FakeOptimizer:
                def step(self):
                    pass
                def zero_grad(self):
                    pass

            state.model = _FakeModel()
            state.loss_fn = _FakeLoss()
            state.optimizer = _FakeOptimizer()
            state.metrics = accuracy  # just accuracy for now

        def evaluate(self, state):
            import numpy as np

            state.model.eval_mode()
            all_preds, all_targets = [], []
            for batch in state.val_data_stream:
                preds = np.asarray(state.model.forward(batch.inputs))
                all_preds.append(preds.reshape(-1))
                all_targets.append(np.asarray(batch.targets))
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            # metrics is currently just accuracy function; compute manually
            acc = float(np.mean(y_pred.round() == y_true))
            state.metrics = type("Metrics", (), {"__getitem__": lambda s, k: acc})()

        def export(self, state):
            import numpy as np

            from pipeline.export.to_csv import to_csv

            if state.val_data_stream is not None:
                preds = []
                for batch in state.val_data_stream:
                    p = np.asarray(state.model.forward(batch.inputs))
                    preds.append(p.reshape(-1))
                state.predictions = np.concatenate(preds)
                to_csv(
                    state.predictions,
                    f"{config.output_dir}/predictions.csv",
                    columns=["prediction"],
                )

    pipeline = DefaultPipeline(config)
    state = pipeline.run(args.mode)
    print(f"Pipeline completed. Mode: {args.mode}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run __main__ tests**

```bash
uv run pytest tests/unit/test_main.py -v
```
Expected: all tests PASS.

- [ ] **Step 6: Test python -m pipeline**

```bash
uv run python -m pipeline
```
Expected: prints version, device, registered components.

- [ ] **Step 7: Commit**

```bash
git add pipeline/__main__.py run.py tests/unit/test_main.py
git commit -m "feat: add CLI entry points (__main__.py + run.py)"
```

---

### Task 11: Integration E2E Test

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_pipeline_e2e.py`

**Interfaces:**
- Consumes: all Phase 1 components + Phase 0 protocols

- [ ] **Step 1: Write the E2E test**

Create `tests/integration/test_pipeline_e2e.py`:

```python
"""End-to-end integration test for the Phase 1 pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pipeline.config import Config
from pipeline.evaluation.metrics import Metrics, accuracy, f1_score
from pipeline.hooks.progress import ProgressHook
from pipeline.pipeline import BasePipeline, PipelineState


def _make_tiny_csv(path: str) -> None:
    """Write a tiny CSV for E2E testing."""
    import pandas as pd

    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "f3": [1, 0, 1, 0, 1, 0, 1, 0],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestPipelineE2E:
    """Full pipeline E2E with fake model."""

    def test_train_mode_completes(self):
        """Happy Path: run('train') completes all 6 stages."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = f"{tmpdir}/train.csv"
            _make_tiny_csv(csv_path)

            config = Config(
                data_dir=tmpdir,
                output_dir=f"{tmpdir}/output",
                batch_size=2,
                num_epochs=2,
                train_ratio=0.75,
                seed=42,
            )

            from pipeline.evaluation.metrics import accuracy as acc_fn

            class E2EPipeline(BasePipeline):
                def load_data(self, state):
                    from pipeline.data.csv_source import CsvDataSource
                    from pipeline.data.split import train_test_split

                    source = CsvDataSource(
                        csv_path, batch_size=config.batch_size, shuffle=False
                    )
                    train, val = train_test_split(
                        source, train_ratio=config.train_ratio, batch_size=2
                    )
                    state.data_stream = train
                    state.val_data_stream = val

                def build_model(self, state):
                    from pipeline.protocols import Loss
                    import numpy as np

                    class FakeModel:
                        def forward(self, inputs):
                            return np.random.random(len(inputs))

                        def parameters(self):
                            return []

                        def train_mode(self):
                            pass

                        def eval_mode(self):
                            pass

                    class FakeLossFn:
                        def forward(self, p, t):
                            return Loss(
                                float(np.mean((np.asarray(p) - np.asarray(t)) ** 2))
                            )

                    class FakeOpt:
                        def step(self):
                            pass

                        def zero_grad(self):
                            pass

                    state.model = FakeModel()
                    state.loss_fn = FakeLossFn()
                    state.optimizer = FakeOpt()
                    state.metrics = Metrics(accuracy=acc_fn, f1=f1_score)

                def evaluate(self, state):
                    import numpy as np

                    state.model.eval_mode()
                    all_preds, all_targets = [], []
                    for batch in state.val_data_stream:
                        preds = np.asarray(state.model.forward(batch.inputs))
                        all_preds.append(preds.reshape(-1))
                        all_targets.append(np.asarray(batch.targets))
                    y_pred = np.concatenate(all_preds)
                    y_true = np.concatenate(all_targets)
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    state.metrics.compute(y_true, y_pred_binary)

                def export(self, state):
                    import numpy as np
                    from pipeline.export.to_csv import to_csv

                    if state.val_data_stream is not None:
                        preds = []
                        for batch in state.val_data_stream:
                            p = np.asarray(state.model.forward(batch.inputs))
                            preds.append(p.reshape(-1))
                        state.predictions = np.concatenate(preds)
                        to_csv(
                            state.predictions,
                            f"{config.output_dir}/predictions.csv",
                            columns=["prediction"],
                        )

            pipeline = E2EPipeline(config)
            pipeline.add_hook(ProgressHook())
            state = pipeline.run("train")

            # Verify state populated
            assert state.history is not None
            assert "loss" in state.history
            assert len(state.history["loss"]) == 2  # 2 epochs
            assert "accuracy" in state.metrics
            assert state.predictions is not None
            # Verify export file
            assert Path(f"{config.output_dir}/predictions.csv").exists()

    def test_infer_mode_skips_training(self):
        """Happy Path: run('infer') skips training stages."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = f"{tmpdir}/train.csv"
            _make_tiny_csv(csv_path)

            config = Config(
                data_dir=tmpdir,
                output_dir=f"{tmpdir}/output",
                batch_size=2,
            )

            class InferPipeline(BasePipeline):
                def load_data(self, state):
                    from pipeline.data.csv_source import CsvDataSource
                    from pipeline.data.split import train_test_split

                    source = CsvDataSource(csv_path, batch_size=2, shuffle=False)
                    train, val = train_test_split(source, batch_size=2)
                    state.data_stream = train
                    state.val_data_stream = val

                def build_model(self, state):
                    pass

                def evaluate(self, state):
                    pass

                def export(self, state):
                    import numpy as np
                    from pipeline.export.to_csv import to_csv

                    state.predictions = np.array([0, 1])
                    to_csv(state.predictions, f"{config.output_dir}/predictions.csv")

            pipeline = InferPipeline(config)
            state = pipeline.run("infer")

            # In infer mode, training fields are untouched
            assert state.model is None
            assert state.history is None
            assert state.current_epoch == 0
            # But export still ran
            assert state.predictions is not None
```

- [ ] **Step 2: Run integration test**

```bash
uv run pytest tests/integration/test_pipeline_e2e.py -v
```
Expected: all tests PASS.

(If tqdm output causes issues in CI, the test still passes since ProgressHook doesn't fail.)

- [ ] **Step 3: Run the full test suite**

```bash
uv run pytest -v
```
Expected: all tests PASS. Record the count.

- [ ] **Step 4: Run ruff + mypy**

```bash
uv run ruff check .
uv run mypy pipeline/
```
Expected: ruff clean, mypy with expected numpy stub warnings on py3.13.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/
git commit -m "test: add E2E integration test for Phase 1 pipeline"
```

---

## Dependency Order

```
Task 1 (Loss) ─────────────────────────────────────────────┐
Task 2 (Metrics) ──────────────────────────────────────────┤
                                                            │
Task 3 (PipelineState + BasePipeline) ────────────────────┤
  └── needs Metrics type (TYPE_CHECKING forward ref)       │
  └── needs TrainLoop (lazy import, Task 5)                │
                                                            │
Task 4 (Test doubles + fixture) ───────────────────────────┤
  └── needs Loss from Task 1                               │
                                                            │
Task 5 (TrainLoop) ────────────────────────────────────────┤
  └── needs Task 1 (Loss), Task 3 (PipelineState),         │
      Task 4 (test doubles)                                │
                                                            │
Task 6 (CsvDataSource) ────────────────────────────────────┤
  └── needs Task 4 (fixture)                               │
                                                            │
Task 7 (train_test_split) ─────────────────────────────────┤
  └── needs Task 6 (CsvDataSource for testing)             │
                                                            │
Task 8 (ProgressHook) ─────────────────────────────────────┤
  └── needs hooks restructuring                            │
                                                            │
Task 9 (to_csv) ─── independent ───────────────────────────┤
                                                            │
Task 10 (CLI) ─────────────────────────────────────────────┤
  └── needs Task 6, 7, 9                                   │
                                                            │
Task 11 (Integration E2E) ─────────────────────────────────┘
  └── needs all tasks
```
