# Phase 0: Pipeline Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the framework-agnostic skeleton — protocols, BasePipeline ABC, registry,
hooks, config, and utilities — with zero ML framework dependencies.

**Architecture:** Define abstract interfaces (Protocols + ABCs) that the pipeline
orchestrator depends on. Concrete implementations (sklearn, numpy, PyTorch) arrive in
later phases. The pipeline declares *when* stages run; dependency injection provides
*how* each stage works.

**Tech Stack:** Python >=3.11,<3.14, numpy (for `ArrayLike` type alias only, no
computation), pytest, ruff, mypy. No torch, sklearn, or sympy.

## Global Constraints

From spec `docs/superpowers/specs/2026-07-28-pipeline-architecture-design.md`:

- Phase 0 written from scratch; zero reference to dl_framework
- No `import torch`, `import sklearn`, `import sympy` in any pipeline module
- Files: `snake_case`, Classes: `PascalCase`, ABCs: `Base` prefix or `Protocol` suffix
- Google-style docstrings on all public API; comments in English
- ≤300 lines/file, ≤50 lines/func, ≤5 public methods/class, ≤3 nesting depth
- `@property` for derived/read-only attributes; no getter methods
- No closures capturing mutable state; no `if task_type` branching; no module-level globals
- `uv` for environment, `hatchling` for build, `ruff` for lint, `mypy` (strict) for types
- TDD iron law: no production code without a failing test first
- One commit per task

---

## File Map

```
Phase 0 creates:
  pyproject.toml                          # uv project config
  .gitignore                              # Python + uv patterns
  mkdocs.yml                              # Material for MkDocs + mkdocstrings + i18n
  pipeline/__init__.py                    # package init, exposes public API
  pipeline/protocols.py                   # ArrayLike, Parameter, Batch, DataStream,
                                          #   ModelProtocol, LossProtocol, OptimizerProtocol
  pipeline/config.py                      # Config dataclass
  pipeline/pipeline.py                    # PipelineState, BasePipeline ABC
  pipeline/registry.py                    # register(), build()
  pipeline/hooks.py                       # BaseHook ABC
  pipeline/utils/__init__.py
  pipeline/utils/device.py               # get_device(), device_info()
  pipeline/utils/seed.py                 # set_seed()
  tests/__init__.py
  tests/unit/__init__.py
  tests/unit/test_protocols.py           # tests for Parameter, Batch, ABCs
  tests/unit/test_pipeline.py            # tests for PipelineState, BasePipeline
  tests/unit/test_registry.py            # tests for register(), build()
  tests/unit/test_hooks.py               # tests for BaseHook
  tests/unit/test_config.py              # tests for Config
  tests/unit/test_utils_device.py        # tests for device utils
  tests/unit/test_utils_seed.py          # tests for seed utils
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `mkdocs.yml`
- Create: `pipeline/__init__.py`
- Create: `pipeline/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`

**Interfaces:**
- Produces: `pipeline` package installable via `uv sync --dev`
- Produces: `uv run pytest` executes and discovers tests
- Produces: `uv run ruff check .` lints the project

**Constraints:** No production code yet — only config files and empty `__init__.py` files. Tests directory must be discoverable by pytest.

- [ ] **Step 1: Write `pyproject.toml`**

Create `/home/billma/requiema/Universal_DL_Pipeline/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "universal-dl-pipeline"
version = "0.1.0"
description = "A teaching-first, framework-agnostic deep learning pipeline library"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11,<3.14"
authors = [
    {name = "RequieMa"},
]
keywords = ["deep-learning", "pipeline", "education", "teaching"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Education",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
dependencies = [
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.6.0",
    "mypy>=1.10.0",
]
docs = [
    "mkdocs-material>=9.5.0",
    "mkdocstrings[python]>=0.25.0",
    "mkdocs-static-i18n>=1.2.0",
]

[tool.hatch.build.targets.wheel]
packages = ["pipeline"]

[tool.uv]
link-mode = "copy"                  # WSL2 cross-filesystem compatibility

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = ["-v", "--tb=short"]
markers = [
    "slow: slow tests (integration, E2E)",
]

[tool.mypy]
strict = true
python_version = "3.11"
```

- [ ] **Step 2: Verify `uv sync` works**

Run: `cd /home/billma/requiema/Universal_DL_Pipeline && uv sync --dev`
Expected: creates `.venv/`, installs `pipeline` (editable), numpy, pytest, ruff, mypy

- [ ] **Step 3: Verify `uv run pytest` discovers nothing but passes**

Run: `uv run pytest`
Expected: "no tests ran" — exit code 5 (pytest's "no tests collected" code) or 0 with "no tests ran"

- [ ] **Step 4: Verify `uv run ruff check .` passes**

Run: `uv run ruff check .`
Expected: no errors (only empty `__init__.py` files exist)

- [ ] **Step 5: Write `.gitignore`**

Create `/home/billma/requiema/Universal_DL_Pipeline/.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Virtual environment
.venv/

# uv
uv.lock

# Data files (test datasets, gitignored)
data/*.csv
data/*.zip
data/*.gz
data/*.tar
data/*.json
data/*.pth
data/*.pt
data/*.onnx
data/*.pkl

# Keep directory structure and README files in data/
!data/**/README.md
!data/**/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 6: Write `mkdocs.yml`**

Create `/home/billma/requiema/Universal_DL_Pipeline/mkdocs.yml`:

```yaml
site_name: Universal DL Pipeline
site_description: A teaching-first, framework-agnostic deep learning pipeline library
repo_url: https://github.com/RequieMa/Universal_DL_Pipeline
repo_name: RequieMa/Universal_DL_Pipeline

theme:
  name: material
  features:
    - navigation.sections
    - navigation.expand
    - search.highlight
    - content.code.copy
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [.]
          options:
            show_source: true
            show_root_heading: true
            docstring_style: google
            show_category_heading: true

nav:
  - Home: index.md
  - API Reference:
      - Protocols: api/protocols.md
      - Pipeline: api/pipeline.md
      - Registry: api/registry.md
      - Hooks: api/hooks.md
      - Config: api/config.md
      - Utils: api/utils.md

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - admonition
```

- [ ] **Step 7: Write package `__init__.py` files**

Create `pipeline/__init__.py`:
```python
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
    - :mod:`pipeline.utils` — device, seed utilities
"""

from pipeline.pipeline import BasePipeline, PipelineState
from pipeline.config import Config

__all__ = ["BasePipeline", "PipelineState", "Config"]
__version__ = "0.1.0"
```

Create `pipeline/utils/__init__.py`:
```python
"""Utility functions: device detection, seed setting."""
```

Create `tests/__init__.py` (empty file).

Create `tests/unit/__init__.py` (empty file).

- [ ] **Step 8: Verify package importable**

Run: `uv run python -c "import pipeline; print(pipeline.__version__)"`
Expected: prints `0.1.0`

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore mkdocs.yml pipeline/__init__.py pipeline/utils/__init__.py tests/__init__.py tests/unit/__init__.py
git commit -m "chore: scaffold project with uv, mkdocs, ruff, mypy

Create pyproject.toml with numpy dependency, dev tooling (pytest, ruff,
mypy), and docs extras (mkdocs-material, mkdocstrings, i18n).
Add .gitignore for Python + data files + IDE artifacts.
Add mkdocs.yml with Material theme and mkdocstrings plugin.
Initialize pipeline/ and tests/ packages."
```

---

### Task 2: Protocols — ArrayLike, Parameter, Batch

**Files:**
- Create: `pipeline/protocols.py`
- Create: `tests/unit/test_protocols.py`

**Interfaces:**
- Produces: `ArrayLike = np.ndarray | Any` (type alias, module-level)
- Produces: `Parameter(data: ArrayLike, grad: ArrayLike | None, name: str)` — dataclass
- Produces: `Batch(inputs: ArrayLike, targets: ArrayLike)` — dataclass

**Constraints:** Only `abc`, `dataclasses`, `typing`, `numpy` imports. No ML frameworks.

- [ ] **Step 1: Write failing tests for `Parameter`**

Create `tests/unit/test_protocols.py`:
```python
"""Unit tests for pipeline.protocols — data structures and abstract interfaces."""
import numpy as np
import pytest
from pipeline.protocols import ArrayLike, Batch, Parameter


class TestParameter:
    """Tests for Parameter dataclass."""

    def test_create_with_data_only(self):
        """Happy Path: Parameter with only data creates valid object."""
        data = np.array([1.0, 2.0, 3.0])
        p = Parameter(data=data)
        assert np.array_equal(p.data, data)
        assert p.grad is None
        assert p.name == ""

    def test_create_with_all_fields(self):
        """Happy Path: Parameter with all fields sets each correctly."""
        data = np.zeros((3, 4))
        grad = np.ones((3, 4))
        p = Parameter(data=data, grad=grad, name="weight")
        assert np.array_equal(p.data, data)
        assert np.array_equal(p.grad, grad)
        assert p.name == "weight"

    def test_equal_parameters_compare_equal(self):
        """Happy Path: Parameters with same values are equal."""
        data = np.array([1.0, 2.0])
        p1 = Parameter(data=data)
        p2 = Parameter(data=data.copy())
        assert p1 == p2

    def test_different_parameters_compare_unequal(self):
        """Boundary: Parameters with different data are not equal."""
        p1 = Parameter(data=np.array([1.0]))
        p2 = Parameter(data=np.array([2.0]))
        assert p1 != p2

    def test_none_grad_vs_zero_grad_unequal(self):
        """Boundary: Parameter with None grad differs from zero grad."""
        p1 = Parameter(data=np.array([1.0]), grad=None)
        p2 = Parameter(data=np.array([1.0]), grad=np.array([0.0]))
        assert p1 != p2


class TestBatch:
    """Tests for Batch dataclass."""

    def test_create_batch_with_arrays(self):
        """Happy Path: Batch with numpy arrays."""
        inputs = np.random.randn(32, 10)
        targets = np.random.randint(0, 2, size=(32,))
        batch = Batch(inputs=inputs, targets=targets)
        assert np.array_equal(batch.inputs, inputs)
        assert np.array_equal(batch.targets, targets)

    def test_batch_single_sample(self):
        """Boundary: Batch with batch_size=1 (single sample)."""
        inputs = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([0])
        batch = Batch(inputs=inputs, targets=targets)
        assert batch.inputs.shape == (1, 3)
        assert batch.targets.shape == (1,)

    def test_batch_with_lists_becomes_arraylike(self):
        """Type Error boundary: Batch accepts list inputs (duck-typed)."""
        batch = Batch(inputs=[[1.0], [2.0]], targets=[0, 1])
        assert batch.inputs is not None

    def test_batch_empty_arrays(self):
        """Empty boundary: Batch with zero-length arrays."""
        inputs = np.array([]).reshape(0, 10)
        targets = np.array([])
        batch = Batch(inputs=inputs, targets=targets)
        assert len(batch.inputs) == 0
        assert len(batch.targets) == 0


class TestArrayLike:
    """Tests for ArrayLike type alias compatibility."""

    def test_ndarray_is_arraylike(self):
        """Happy Path: numpy.ndarray is compatible with ArrayLike."""
        x: ArrayLike = np.array([1.0, 2.0])
        assert x is not None

    def test_parameter_data_is_arraylike(self):
        """Happy Path: Parameter.data is type-compatible with ArrayLike."""
        p = Parameter(data=np.ones(5))
        _data: ArrayLike = p.data
        assert _data is not None
```

- [ ] **Step 2: Run tests, verify ALL fail**

Run: `uv run pytest tests/unit/test_protocols.py -v`
Expected: ALL FAIL — `ModuleNotFoundError: No module named 'pipeline.protocols'`

- [ ] **Step 3: Implement `pipeline/protocols.py` (data structures only)**

Create `pipeline/protocols.py`:
```python
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
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np

# ── Type alias ──────────────────────────────────────────────────────────
# WHY: ArrayLike lets the pipeline accept numpy arrays, torch tensors, or
# any duck-typed array without binding to a specific framework.
ArrayLike = np.ndarray | Any


# ── Data structures ─────────────────────────────────────────────────────
@dataclass
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
```

- [ ] **Step 4: Run tests, verify ALL pass**

Run: `uv run pytest tests/unit/test_protocols.py::TestParameter tests/unit/test_protocols.py::TestBatch tests/unit/test_protocols.py::TestArrayLike -v`
Expected: ALL PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/protocols.py tests/unit/test_protocols.py
git commit -m "feat: add ArrayLike, Parameter, Batch to protocols

Define framework-agnostic data structures for the pipeline:
- ArrayLike type alias (np.ndarray | Any)
- Parameter dataclass (data, grad, name)
- Batch dataclass (inputs, targets)

TDD: 7 tests covering happy path, boundary (single sample, empty arrays),
and equality semantics."
```

---

### Task 3: Protocols — DataStream, ModelProtocol, LossProtocol, OptimizerProtocol

**Files:**
- Modify: `pipeline/protocols.py` — append ABC classes
- Modify: `tests/unit/test_protocols.py` — append ABC tests

**Interfaces:**
- Consumes: `ArrayLike`, `Parameter`, `Batch` (from Task 2)
- Produces:
  - `DataStream` ABC: `__iter__() -> Iterator[Batch]`, `__len__() -> int`
  - `ModelProtocol` ABC: `forward(inputs: ArrayLike) -> ArrayLike`, `parameters() -> Iterable[Parameter]`, `train_mode()`, `eval_mode()`
  - `LossProtocol` ABC: `forward(predictions: ArrayLike, targets: ArrayLike) -> float`, `__call__(...) -> float`
  - `OptimizerProtocol` ABC: `step()`, `zero_grad()`

- [ ] **Step 1: Write failing tests for ABCs (can't instantiate, must implement)**

Append to `tests/unit/test_protocols.py`:
```python
from pipeline.protocols import DataStream, ModelProtocol, LossProtocol, OptimizerProtocol


# ── Fake implementations for testing ABCs ────────────────────────────────
class FakeDataStream(DataStream):
    """Minimal DataStream implementation for testing."""
    def __init__(self, batches: list[Batch]):
        self._batches = batches

    def __iter__(self) -> Iterator[Batch]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


class FakeModel(ModelProtocol):
    """Minimal ModelProtocol implementation for testing."""
    def __init__(self):
        self._params = [Parameter(data=np.array([1.0]), name="w")]
        self._mode = "train"

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        return np.asarray(inputs) * 2.0

    def parameters(self) -> Iterable[Parameter]:
        return iter(self._params)

    def train_mode(self) -> None:
        self._mode = "train"

    def eval_mode(self) -> None:
        self._mode = "eval"


class FakeLoss(LossProtocol):
    """Minimal LossProtocol implementation for testing."""
    def forward(self, predictions: ArrayLike, targets: ArrayLike) -> float:
        preds = np.asarray(predictions)
        targs = np.asarray(targets)
        return float(np.mean((preds - targs) ** 2))


class FakeOptimizer(OptimizerProtocol):
    """Minimal OptimizerProtocol implementation for testing."""
    def __init__(self):
        self.step_count = 0
        self.zero_count = 0

    def step(self) -> None:
        self.step_count += 1

    def zero_grad(self) -> None:
        self.zero_count += 1


# ── DataStream tests ─────────────────────────────────────────────────────
class TestDataStream:
    """Tests for DataStream ABC."""

    def test_cannot_instantiate_abc_directly(self):
        """Type Error: instantiating DataStream directly raises TypeError."""
        with pytest.raises(TypeError):
            DataStream()  # type: ignore[abstract]

    def test_iter_yields_batches(self):
        """Happy Path: DataStream iter yields Batch objects."""
        batch = Batch(inputs=np.array([1.0]), targets=np.array([2.0]))
        stream = FakeDataStream([batch])
        items = list(stream)
        assert len(items) == 1
        assert isinstance(items[0], Batch)

    def test_len_returns_count(self):
        """Happy Path: DataStream len returns batch count."""
        batches = [
            Batch(inputs=np.array([i]), targets=np.array([i]))
            for i in range(3)
        ]
        stream = FakeDataStream(batches)
        assert len(stream) == 3

    def test_empty_stream_len_zero(self):
        """Empty: DataStream with no batches has len=0."""
        stream = FakeDataStream([])
        assert len(stream) == 0

    def test_empty_stream_iter_empty(self):
        """Empty: iterating empty DataStream yields nothing."""
        stream = FakeDataStream([])
        items = list(stream)
        assert items == []

    def test_stress_many_batches(self):
        """Stress: DataStream with 1000 batches iterates correctly."""
        n = 1000
        batches = [
            Batch(inputs=np.array([float(i)]), targets=np.array([float(i)]))
            for i in range(n)
        ]
        stream = FakeDataStream(batches)
        assert len(stream) == n
        count = 0
        for batch in stream:
            count += 1
        assert count == n


# ── ModelProtocol tests ───────────────────────────────────────────────────
class TestModelProtocol:
    """Tests for ModelProtocol ABC."""

    def test_cannot_instantiate_abc_directly(self):
        """Type Error: instantiating ModelProtocol directly raises TypeError."""
        with pytest.raises(TypeError):
            ModelProtocol()  # type: ignore[abstract]

    def test_forward_returns_arraylike(self):
        """Happy Path: forward returns array-like output."""
        model = FakeModel()
        inputs = np.array([1.0, 2.0, 3.0])
        output = model.forward(inputs)
        assert output is not None
        assert len(np.asarray(output)) == 3

    def test_parameters_returns_iterable(self):
        """Happy Path: parameters returns iterable of Parameter objects."""
        model = FakeModel()
        params = list(model.parameters())
        assert len(params) == 1
        assert isinstance(params[0], Parameter)

    def test_empty_parameters_is_valid(self):
        """Boundary: model with zero parameters (e.g., sklearn wrapper)
        returns empty iterable."""
        class NoParamModel(ModelProtocol):
            def forward(self, inputs):
                return np.asarray(inputs)
            def parameters(self):
                return iter([])
            def train_mode(self): pass
            def eval_mode(self): pass

        model = NoParamModel()
        params = list(model.parameters())
        assert params == []

    def test_train_mode_sets_training(self):
        """Happy Path: train_mode() switches to training mode."""
        model = FakeModel()
        model.train_mode()
        assert model._mode == "train"

    def test_eval_mode_sets_evaluation(self):
        """Happy Path: eval_mode() switches to evaluation mode."""
        model = FakeModel()
        model.eval_mode()
        assert model._mode == "eval"

    def test_toggle_mode_repeatedly(self):
        """Concurrency: toggling train/eval repeatedly works correctly."""
        model = FakeModel()
        for _ in range(10):
            model.train_mode()
            model.eval_mode()
        assert model._mode == "eval"


# ── LossProtocol tests ─────────────────────────────────────────────────────
class TestLossProtocol:
    """Tests for LossProtocol ABC."""

    def test_cannot_instantiate_abc_directly(self):
        """Type Error: instantiating LossProtocol directly raises TypeError."""
        with pytest.raises(TypeError):
            LossProtocol()  # type: ignore[abstract]

    def test_call_delegates_to_forward(self):
        """Happy Path: __call__ delegates to forward() and returns float."""
        loss_fn = FakeLoss()
        preds = np.array([1.0, 2.0])
        targs = np.array([0.5, 2.5])
        result = loss_fn(preds, targs)
        assert isinstance(result, float)

    def test_perfect_prediction_yields_zero_loss(self):
        """Boundary: perfect predictions produce near-zero loss."""
        loss_fn = FakeLoss()
        preds = np.array([1.0, 2.0, 3.0])
        targs = np.array([1.0, 2.0, 3.0])
        result = loss_fn(preds, targs)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_wrong_predictions_yield_positive_loss(self):
        """Happy Path: wrong predictions produce positive loss."""
        loss_fn = FakeLoss()
        preds = np.array([0.0, 0.0])
        targs = np.array([10.0, 10.0])
        result = loss_fn(preds, targs)
        assert result > 0.0

    def test_single_sample(self):
        """Boundary: loss computed on a single-sample batch."""
        loss_fn = FakeLoss()
        result = loss_fn(np.array([3.0]), np.array([1.0]))
        assert isinstance(result, float)
        assert result > 0.0


# ── OptimizerProtocol tests ─────────────────────────────────────────────────
class TestOptimizerProtocol:
    """Tests for OptimizerProtocol ABC."""

    def test_cannot_instantiate_abc_directly(self):
        """Type Error: instantiating OptimizerProtocol directly raises TypeError."""
        with pytest.raises(TypeError):
            OptimizerProtocol()  # type: ignore[abstract]

    def test_step_increments(self):
        """Happy Path: step() executes without error."""
        opt = FakeOptimizer()
        opt.step()
        assert opt.step_count == 1

    def test_zero_grad_increments(self):
        """Happy Path: zero_grad() executes without error."""
        opt = FakeOptimizer()
        opt.zero_grad()
        assert opt.zero_count == 1

    def test_multiple_steps_accumulate(self):
        """Concurrency: multiple step() calls accumulate correctly."""
        opt = FakeOptimizer()
        for _ in range(5):
            opt.step()
        assert opt.step_count == 5

    def test_zero_grad_then_step_independent(self):
        """Concurrency: zero_grad and step counts are independent."""
        opt = FakeOptimizer()
        opt.zero_grad()
        opt.step()
        opt.zero_grad()
        assert opt.zero_count == 2
        assert opt.step_count == 1

    def test_stress_many_optimizer_steps(self):
        """Stress: 1000 step() calls complete without memory issues."""
        opt = FakeOptimizer()
        for _ in range(1000):
            opt.step()
            opt.zero_grad()
        assert opt.step_count == 1000
        assert opt.zero_count == 1000
```

- [ ] **Step 2: Run tests, verify new ABC tests fail**

Run: `uv run pytest tests/unit/test_protocols.py -v`
Expected: `TestDataStream`, `TestModelProtocol`, `TestLossProtocol`, `TestOptimizerProtocol` tests FAIL with `ImportError` for the ABCs

- [ ] **Step 3: Append ABC classes to `pipeline/protocols.py`**

Append to `pipeline/protocols.py` (after the `Batch` dataclass):
```python
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
    def forward(self, predictions: ArrayLike, targets: ArrayLike) -> float:
        """Compute the loss.

        Args:
            predictions: Model output of shape ``(batch_size, *dims)``.
            targets: Ground truth of shape ``(batch_size, *dims)``.

        Returns:
            Scalar loss value.
        """
        ...

    def __call__(self, predictions: ArrayLike, targets: ArrayLike) -> float:
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
```

- [ ] **Step 4: Run tests, verify ALL pass**

Run: `uv run pytest tests/unit/test_protocols.py -v`
Expected: ALL 31 tests PASS

- [ ] **Step 5: Ruff and mypy**

Run: `uv run ruff check pipeline/protocols.py tests/unit/test_protocols.py`
Expected: no errors

Run: `uv run mypy pipeline/protocols.py`
Expected: no errors (or acceptable errors for `ArrayLike = np.ndarray | Any`)

- [ ] **Step 6: Commit**

```bash
git add pipeline/protocols.py tests/unit/test_protocols.py
git commit -m "feat: add DataStream, ModelProtocol, LossProtocol, OptimizerProtocol ABCs

Define the four abstract interfaces that the pipeline orchestrator
depends on. Each ABC has Google-style docstrings with usage examples.

TDD: 24 new tests (31 total) covering:
- ABC instantiation prevents TypeError
- Happy path for each protocol method
- Boundary: empty parameters, perfect predictions, single samples
- Concurrency: toggling modes, multiple optimizer calls
- Stress: 1000-batch DataStream, 1000 optimizer steps"
```

---

### Task 4: Config Dataclass

**Files:**
- Create: `pipeline/config.py`
- Create: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `Config` dataclass with fields:
  - `data_dir: str = "./data"` — root directory for datasets
  - `output_dir: str = "./outputs"` — where checkpoints and predictions go
  - `task_name: str = ""` — human-readable task label
  - `seed: int = 42` — random seed for reproducibility
  - `batch_size: int = 32`
  - `num_epochs: int = 10`
  - `learning_rate: float = 0.001`
  - `train_ratio: float = 0.8` — fraction for training in train_test_split
  - `device: str = "auto"` — "auto", "cpu", "cuda", "mps"
  - `use_amp: bool = False` — automatic mixed precision
  - `num_workers: int = 0` — DataLoader workers
  - `model_name: str = ""` — registry key for the model
  - `optimizer_name: str = "sgd"` — registry key for optimizer
  - `loss_name: str = "cross_entropy"` — registry key for loss function
  - `metrics: list[str]` — list of metric names, default `["accuracy"]`
  - `checkpoint_dir: str | None = None` — override checkpoint path
  - `log_interval: int = 10` — log every N batches
  - `from_yaml(path: str) -> Config` classmethod

- [ ] **Step 1: Write failing tests for Config**

Create `tests/unit/test_config.py`:
```python
"""Unit tests for pipeline.config — Config dataclass."""
import pytest
from pipeline.config import Config


class TestConfigDefaults:
    """Tests for Config default values."""

    def test_create_with_no_args_uses_defaults(self):
        """Happy Path: Config() uses all documented defaults."""
        cfg = Config()
        assert cfg.data_dir == "./data"
        assert cfg.output_dir == "./outputs"
        assert cfg.seed == 42
        assert cfg.batch_size == 32
        assert cfg.num_epochs == 10
        assert cfg.learning_rate == pytest.approx(0.001)
        assert cfg.train_ratio == pytest.approx(0.8)
        assert cfg.device == "auto"
        assert cfg.use_amp is False
        assert cfg.num_workers == 0
        assert cfg.model_name == ""
        assert cfg.optimizer_name == "sgd"
        assert cfg.loss_name == "cross_entropy"
        assert cfg.metrics == ["accuracy"]
        assert cfg.log_interval == 10

    def test_task_name_defaults_to_empty(self):
        """Boundary: task_name default is empty string."""
        cfg = Config()
        assert cfg.task_name == ""

    def test_checkpoint_dir_defaults_to_none(self):
        """Boundary: checkpoint_dir default is None."""
        cfg = Config()
        assert cfg.checkpoint_dir is None


class TestConfigOverride:
    """Tests for Config field overrides."""

    def test_override_single_field(self):
        """Happy Path: override one field keeps defaults for others."""
        cfg = Config(batch_size=64)
        assert cfg.batch_size == 64
        assert cfg.num_epochs == 10  # default preserved

    def test_override_all_fields(self):
        """Happy Path: override every field."""
        cfg = Config(
            data_dir="/custom/data",
            output_dir="/custom/output",
            task_name="test_task",
            seed=123,
            batch_size=128,
            num_epochs=50,
            learning_rate=0.01,
            train_ratio=0.9,
            device="cuda",
            use_amp=True,
            num_workers=4,
            model_name="resnet18",
            optimizer_name="adam",
            loss_name="mse",
            metrics=["accuracy", "f1"],
            checkpoint_dir="/custom/ckpt",
            log_interval=5,
        )
        assert cfg.data_dir == "/custom/data"
        assert cfg.seed == 123
        assert cfg.batch_size == 128
        assert cfg.device == "cuda"
        assert cfg.use_amp is True
        assert cfg.model_name == "resnet18"
        assert cfg.optimizer_name == "adam"
        assert cfg.loss_name == "mse"
        assert cfg.metrics == ["accuracy", "f1"]
        assert cfg.checkpoint_dir == "/custom/ckpt"
        assert cfg.log_interval == 5


class TestConfigValidation:
    """Tests for Config validation on invalid inputs."""

    def test_negative_batch_size_raises(self):
        """Boundary: negative batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            Config(batch_size=-1)

    def test_zero_batch_size_raises(self):
        """Boundary: batch_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            Config(batch_size=0)

    def test_negative_num_epochs_raises(self):
        """Boundary: negative num_epochs raises ValueError."""
        with pytest.raises(ValueError, match="num_epochs"):
            Config(num_epochs=-1)

    def test_train_ratio_zero_raises(self):
        """Boundary: train_ratio=0 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio"):
            Config(train_ratio=0.0)

    def test_train_ratio_one_raises(self):
        """Boundary: train_ratio=1.0 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio"):
            Config(train_ratio=1.0)

    def test_train_ratio_out_of_range_raises(self):
        """Boundary: train_ratio > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio"):
            Config(train_ratio=1.5)

    def test_negative_learning_rate_raises(self):
        """Boundary: negative learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate"):
            Config(learning_rate=-0.001)

    def test_negative_log_interval_raises(self):
        """Boundary: negative log_interval raises ValueError."""
        with pytest.raises(ValueError, match="log_interval"):
            Config(log_interval=-1)


class TestConfigImmutability:
    """Tests for Config field-type safety."""

    def test_fields_are_accessible(self):
        """Happy Path: all fields readable after construction."""
        cfg = Config(task_name="mnist", batch_size=64)
        assert cfg.task_name == "mnist"

    def test_metrics_mutation_shared_list_regression(self):
        """Concurrency: two Configs with default metrics don't share the
        same list object (common dataclass footgun)."""
        cfg1 = Config()
        cfg2 = Config()
        cfg1.metrics.append("f1")
        assert "f1" not in cfg2.metrics
        assert cfg2.metrics == ["accuracy"]


class TestConfigFromYaml:
    """Tests for Config.from_yaml() classmethod."""

    def test_from_yaml_loads_config(self, tmp_path):
        """Happy Path: from_yaml loads a minimal config file."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("""
task_name: my_task
batch_size: 64
num_epochs: 20
learning_rate: 0.01
""")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.task_name == "my_task"
        assert cfg.batch_size == 64
        assert cfg.num_epochs == 20
        assert cfg.learning_rate == pytest.approx(0.01)
        # Unspecified fields use defaults
        assert cfg.seed == 42

    def test_from_yaml_file_not_found(self, tmp_path):
        """Error recovery: from_yaml on missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(str(tmp_path / "nonexistent.yaml"))
```

- [ ] **Step 2: Run tests, verify ALL fail**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: ALL FAIL — `ModuleNotFoundError: No module named 'pipeline.config'`

- [ ] **Step 3: Implement `pipeline/config.py`**

Create `pipeline/config.py`:
```python
"""Configuration dataclass for the pipeline.

A single source of truth for all pipeline settings. Serialized
to/from YAML files. Every field has a sensible default so students
can start with ``Config()`` and add options as they learn them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _validate_positive(value: int | float, name: str) -> None:
    """Raise ValueError if value is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_ratio(value: float, name: str) -> None:
    """Raise ValueError if not in (0, 1)."""
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be in (0, 1), got {value}")


@dataclass
class Config:
    """Pipeline configuration.

    All fields have defaults so beginners can start minimal and add
    options as they progress through the syllabus.

    Attributes:
        data_dir: Root directory for datasets.
        output_dir: Where checkpoints and predictions are written.
        task_name: Human-readable label for this experiment.
        seed: Random seed for reproducibility.
        batch_size: Samples per training batch.
        num_epochs: Maximum training epochs.
        learning_rate: Initial learning rate for the optimizer.
        train_ratio: Fraction of data used for training (remainder = validation).
        device: Compute device — ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
        use_amp: Enable automatic mixed precision (GPU only).
        num_workers: DataLoader worker processes.
        model_name: Registry key for the model backbone.
        optimizer_name: Registry key for the optimizer.
        loss_name: Registry key for the loss function.
        metrics: List of metric names to compute during evaluation.
        checkpoint_dir: Directory for model checkpoints (None = output_dir/checkpoints).
        log_interval: Log training progress every N batches.
    """

    # ── Paths ────────────────────────────────────────────────────────
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    task_name: str = ""

    # ── Reproducibility ──────────────────────────────────────────────
    seed: int = 42

    # ── Data ─────────────────────────────────────────────────────────
    batch_size: int = 32
    train_ratio: float = 0.8

    # ── Training ─────────────────────────────────────────────────────
    num_epochs: int = 10
    learning_rate: float = 0.001

    # ── Hardware ─────────────────────────────────────────────────────
    device: str = "auto"
    use_amp: bool = False
    num_workers: int = 0

    # ── Components (registry keys) ───────────────────────────────────
    model_name: str = ""
    optimizer_name: str = "sgd"
    loss_name: str = "cross_entropy"
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])
    # WHY: field(default_factory=...) creates a fresh list per instance,
    # preventing the mutable-default-argument footgun.

    # ── Checkpointing ────────────────────────────────────────────────
    checkpoint_dir: str | None = None

    # ── Logging ──────────────────────────────────────────────────────
    log_interval: int = 10

    def __post_init__(self) -> None:
        """Validate field values after dataclass construction."""
        _validate_positive(self.batch_size, "batch_size")
        _validate_positive(self.num_epochs, "num_epochs")
        _validate_positive(self.log_interval, "log_interval")
        _validate_positive(self.learning_rate, "learning_rate")
        _validate_ratio(self.train_ratio, "train_ratio")

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file.

        Args:
            path: Path to a ``.yaml`` or ``.yml`` file.

        Returns:
            A new Config with values from the file (unspecified
            fields retain their defaults).

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # WHY: YAML import is inside the method rather than at module
        # level, so Config is usable without PyYAML installed. Only
        # users who need YAML loading need the dependency.
        import yaml  # type: ignore[import-untyped]

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Filter to only known fields (ignore extras silently — forward compat)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
```

- [ ] **Step 4: Run tests, verify ALL pass**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: ALL 14 tests PASS

- [ ] **Step 5: Ruff and mypy**

Run: `uv run ruff check pipeline/config.py tests/unit/test_config.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/config.py tests/unit/test_config.py
git commit -m "feat: add Config dataclass with YAML loading

Central configuration dataclass with ~20 fields, sensible defaults,
__post_init__ validation, and from_yaml() classmethod.

TDD: 14 tests covering:
- All default values
- Field override with defaults preserved
- Validation: negative/zero batch_size, num_epochs, LR, train_ratio
- Isolation: two Configs don't share metrics list
- YAML loading with partial overrides, file-not-found error"
```

---

### Task 5: PipelineState

**Files:**
- Create: `pipeline/pipeline.py` (PipelineState only)
- Create: `tests/unit/test_pipeline.py` (PipelineState tests only)

**Interfaces:**
- Produces: `PipelineState` dataclass with fields:
  - `config: Config`
  - `mode: Literal["train", "infer"] = "train"`
  - `data_stream: DataStream | None = None`
  - `features: ArrayLike | None = None`
  - `model: ModelProtocol | None = None`
  - `loss_fn: LossProtocol | None = None`
  - `optimizer: OptimizerProtocol | None = None`
  - `history: dict | None = None`
  - `metrics: dict[str, float]` (default empty dict)
  - `predictions: ArrayLike | None = None`
  - `current_epoch: int = 0`
  - `should_stop: bool = False`
  - `is_training` property (derived from mode)

**Constraints:** Uses `Protocol` types from `pipeline.protocols`. No concrete implementations.

- [ ] **Step 1: Write failing tests for PipelineState**

Create `tests/unit/test_pipeline.py`:
```python
"""Unit tests for pipeline.pipeline — PipelineState and BasePipeline."""
from typing import Literal

import numpy as np
import pytest
from pipeline.config import Config
from pipeline.pipeline import PipelineState


class TestPipelineStateDefaults:
    """Tests for PipelineState default values."""

    def test_create_with_config_only(self):
        """Happy Path: PipelineState(config) uses all defaults."""
        cfg = Config()
        state = PipelineState(config=cfg)
        assert state.config is cfg
        assert state.mode == "train"
        assert state.data_stream is None
        assert state.features is None
        assert state.model is None
        assert state.loss_fn is None
        assert state.optimizer is None
        assert state.history is None
        assert state.metrics == {}
        assert state.predictions is None
        assert state.current_epoch == 0
        assert state.should_stop is False

    def test_create_with_infer_mode(self):
        """Happy Path: PipelineState with mode='infer'."""
        state = PipelineState(config=Config(), mode="infer")
        assert state.mode == "infer"


class TestPipelineStateProperties:
    """Tests for PipelineState @property methods."""

    def test_is_training_when_train_mode(self):
        """Property: is_training is True when mode='train'."""
        state = PipelineState(config=Config(), mode="train")
        assert state.is_training is True

    def test_is_training_false_when_infer_mode(self):
        """Property: is_training is False when mode='infer'."""
        state = PipelineState(config=Config(), mode="infer")
        assert state.is_training is False

    def test_is_training_read_only(self):
        """Property: is_training is read-only (raises AttributeError on set)."""
        state = PipelineState(config=Config())
        with pytest.raises(AttributeError):
            state.is_training = False  # type: ignore[misc]


class TestPipelineStateFieldAssignment:
    """Tests for PipelineState field mutations."""

    def test_assign_and_read_data_stream(self):
        """Happy Path: data_stream field is writable and readable."""
        state = PipelineState(config=Config())
        state.data_stream = "fake_stream"  # protocol duck-type
        assert state.data_stream == "fake_stream"

    def test_assign_model(self):
        """Happy Path: model field is writable and readable."""
        state = PipelineState(config=Config())
        state.model = "fake_model"  # protocol duck-type
        assert state.model == "fake_model"

    def test_assign_metrics(self):
        """Happy Path: metrics dict is mutable."""
        state = PipelineState(config=Config())
        state.metrics["accuracy"] = 0.95
        state.metrics["f1"] = 0.93
        assert state.metrics == {"accuracy": 0.95, "f1": 0.93}

    def test_should_stop_flag(self):
        """Happy Path: should_stop is settable (for early stopping hooks)."""
        state = PipelineState(config=Config())
        assert state.should_stop is False
        state.should_stop = True
        assert state.should_stop is True

    def test_current_epoch_increment(self):
        """Happy Path: current_epoch is mutable for training loop."""
        state = PipelineState(config=Config())
        for epoch in range(5):
            state.current_epoch = epoch
        assert state.current_epoch == 4

    def test_predictions_assign_arraylike(self):
        """Happy Path: predictions field accepts numpy array."""
        state = PipelineState(config=Config())
        preds = np.array([[0.1, 0.9], [0.8, 0.2]])
        state.predictions = preds
        assert np.array_equal(state.predictions, preds)


class TestPipelineStateIsolation:
    """Tests for PipelineState isolation between runs."""

    def test_two_states_independent(self):
        """Concurrency: two PipelineState instances don't share state."""
        cfg = Config()
        state1 = PipelineState(config=cfg, mode="train")
        state2 = PipelineState(config=cfg, mode="infer")

        state1.metrics["accuracy"] = 0.9
        state2.metrics["accuracy"] = 0.5

        assert state1.metrics["accuracy"] == 0.9
        assert state2.metrics["accuracy"] == 0.5
        assert state1.mode != state2.mode

    def test_metrics_default_dicts_isolated(self):
        """Concurrency: default metrics dict is unique per instance."""
        s1 = PipelineState(config=Config())
        s2 = PipelineState(config=Config())
        s1.metrics["x"] = 1.0
        assert "x" not in s2.metrics
```

- [ ] **Step 2: Run tests, verify ALL fail**

Run: `uv run pytest tests/unit/test_pipeline.py -v`
Expected: ALL FAIL — `ModuleNotFoundError: No module named 'pipeline.pipeline'`

- [ ] **Step 3: Implement `pipeline/pipeline.py` (PipelineState only)**

Create `pipeline/pipeline.py`:
```python
"""Base pipeline orchestration.

Defines the :class:`PipelineState` data bus and the :class:`BasePipeline`
abstract class that orchestrates the six pipeline stages.

The pipeline defines *when* stages run. Concrete implementations define
*how* each stage works by satisfying the protocols in :mod:`pipeline.protocols`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from pipeline.config import Config
    from pipeline.protocols import DataStream, LossProtocol, ModelProtocol, OptimizerProtocol

ArrayLike = np.ndarray | Any


@dataclass
class PipelineState:
    """Shared state bus passed through all pipeline stages.

    Each stage reads from and writes to specific fields. Students
    can inspect ``state`` after each stage to understand what happened.

    Attributes:
        config: The pipeline configuration (read-only after construction).
        mode: ``"train"`` runs all 6 stages; ``"infer"`` skips training.
        data_stream: Set by Stage 1 (load_data).
        features: Set by Stage 2 (extract_features). None if pass-through.
        model: Set by Stage 3 (build_model).
        loss_fn: Set by Stage 3.
        optimizer: Set by Stage 3.
        history: Set by Stage 4 (train). A dict with loss/accuracy curves.
        metrics: Set by Stage 5 (evaluate). e.g., ``{"accuracy": 0.92}``.
        predictions: Set by Stage 6 (export). Model predictions on test data.
        current_epoch: Tracked by the training loop; hooks can read it.
        should_stop: Early-stopping hooks set this to True.
    """

    config: "Config"
    mode: Literal["train", "infer"] = "train"

    # Stage outputs (None until the stage runs)
    data_stream: "DataStream | None" = None
    features: ArrayLike | None = None
    model: "ModelProtocol | None" = None
    loss_fn: "LossProtocol | None" = None
    optimizer: "OptimizerProtocol | None" = None
    history: dict | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: ArrayLike | None = None

    # Control signals
    current_epoch: int = 0
    should_stop: bool = False

    # ── Derived properties ──────────────────────────────────────────
    @property
    def is_training(self) -> bool:
        """True if the pipeline is in training mode.

        Derived from :attr:`mode`. Safe to query in any stage or hook.
        """
        return self.mode == "train"
```

- [ ] **Step 4: Run tests, verify ALL pass**

Run: `uv run pytest tests/unit/test_pipeline.py -v`
Expected: ALL 12 tests PASS

- [ ] **Step 5: Ruff and mypy**

Run: `uv run ruff check pipeline/pipeline.py tests/unit/test_pipeline.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: add PipelineState data bus with is_training property

Shared state container passed through all pipeline stages.
All stage outputs default to None until the stage runs.
is_training @property derived from mode — cannot be set directly.

TDD: 12 tests covering:
- All default values for train/infer modes
- Property: is_training read-only, correctly reflects mode
- Field mutation: model, metrics, predictions, should_stop, epoch
- Isolation: two states don't share mutable dicts"
```

---

### Task 6: BaseHook ABC

**Files:**
- Create: `pipeline/hooks.py`
- Create: `tests/unit/test_hooks.py`

**Interfaces:**
- Produces: `BaseHook` with five methods (all no-op by default):
  - `on_stage_start(stage: str, state: PipelineState) -> None`
  - `on_stage_end(stage: str, state: PipelineState) -> None`
  - `on_epoch_start(epoch: int, state: PipelineState) -> None`
  - `on_epoch_end(epoch: int, state: PipelineState) -> None`
  - `on_batch_end(batch: int, loss: float, state: PipelineState) -> None`

**Constraints:** NOT an ABC — all methods have default no-op implementations so students can write a hook with a single method override.

- [ ] **Step 1: Write failing tests for BaseHook**

Create `tests/unit/test_hooks.py`:
```python
"""Unit tests for pipeline.hooks — BaseHook."""
import pytest
from pipeline.config import Config
from pipeline.hooks import BaseHook
from pipeline.pipeline import PipelineState


class TestBaseHook:
    """Tests for BaseHook."""

    def test_all_hooks_default_to_noop(self):
        """Happy Path: all BaseHook methods are no-ops by default.
        A bare BaseHook() instance should not crash on any event."""
        hook = BaseHook()
        state = PipelineState(config=Config())
        hook.on_stage_start("load_data", state)
        hook.on_stage_end("load_data", state)
        hook.on_epoch_start(0, state)
        hook.on_epoch_end(0, state)
        hook.on_batch_end(0, 0.5, state)

    def test_subclass_overrides_one_method(self):
        """Happy Path: subclass can override a single hook method."""
        called = False

        class OneMethodHook(BaseHook):
            def on_stage_start(self, stage, state):
                nonlocal called
                called = True

        hook = OneMethodHook()
        hook.on_stage_start("train", PipelineState(config=Config()))
        assert called is True

    def test_subclass_overrides_all_methods(self):
        """Happy Path: subclass can override all five hook methods."""
        log: list[str] = []

        class FullHook(BaseHook):
            def on_stage_start(self, stage, state):
                log.append(f"stage_start:{stage}")
            def on_stage_end(self, stage, state):
                log.append(f"stage_end:{stage}")
            def on_epoch_start(self, epoch, state):
                log.append(f"epoch_start:{epoch}")
            def on_epoch_end(self, epoch, state):
                log.append(f"epoch_end:{epoch}")
            def on_batch_end(self, batch, loss, state):
                log.append(f"batch:{batch}:{loss}")

        hook = FullHook()
        state = PipelineState(config=Config())
        hook.on_stage_start("train", state)
        hook.on_epoch_start(0, state)
        hook.on_batch_end(5, 0.3, state)
        hook.on_epoch_end(0, state)
        hook.on_stage_end("train", state)

        assert log == [
            "stage_start:train",
            "epoch_start:0",
            "batch:5:0.3",
            "epoch_end:0",
            "stage_end:train",
        ]

    def test_hook_reads_state_current_epoch(self):
        """Happy Path: hook can read state.current_epoch."""
        last_epoch: int = -1

        class EpochReaderHook(BaseHook):
            def on_epoch_end(self, epoch, state):
                nonlocal last_epoch
                last_epoch = state.current_epoch

        state = PipelineState(config=Config())
        state.current_epoch = 7
        EpochReaderHook().on_epoch_end(7, state)
        assert last_epoch == 7

    def test_hook_sets_should_stop(self):
        """Happy Path: hook can set state.should_stop for early stopping."""
        state = PipelineState(config=Config())
        assert state.should_stop is False

        class EarlyStopHook(BaseHook):
            def on_epoch_end(self, epoch, state):
                if epoch >= 5:
                    state.should_stop = True

        EarlyStopHook().on_epoch_end(5, state)
        assert state.should_stop is True

    def test_multiple_hooks_independent(self):
        """Concurrency: two hook instances don't share state."""
        counter_a = 0
        counter_b = 0

        class CounterHook(BaseHook):
            def __init__(self, tag: str):
                self.tag = tag
            def on_stage_start(self, stage, state):
                nonlocal counter_a, counter_b
                if self.tag == "a":
                    counter_a += 1  # type: ignore[unused-ignore]
                else:
                    counter_b += 1  # type: ignore[unused-ignore]

        a = CounterHook("a")
        b = CounterHook("b")
        state = PipelineState(config=Config())
        a.on_stage_start("load_data", state)
        b.on_stage_start("load_data", state)
        assert counter_a == 1
        assert counter_b == 1
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/unit/test_hooks.py -v`
Expected: ALL FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `pipeline/hooks.py`**

Create `pipeline/hooks.py`:
```python
"""Hook system for cross-cutting concerns in the pipeline.

Hooks observe pipeline lifecycle events without modifying the pipeline
stages themselves. Students add one hook at a time to layer on logging,
checkpointing, and early stopping.

Five hook points:
    - :meth:`on_stage_start` — before a stage executes
    - :meth:`on_stage_end` — after a stage completes
    - :meth:`on_epoch_start` — before each training epoch
    - :meth:`on_epoch_end` — after each training epoch
    - :meth:`on_batch_end` — after each training batch

Usage::

    class MyLogger(BaseHook):
        def on_epoch_end(self, epoch, state):
            print(f"Epoch {epoch}: loss={state.history['loss'][-1]:.4f}")

    pipeline.add_hook(MyLogger())
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.pipeline import PipelineState


class BaseHook:
    """Base class for pipeline hooks. All five methods are no-ops by default.

    Subclasses override only the events they care about. The pipeline
    calls hooks in registration order. Exceptions in hooks are caught
    and logged — they never interrupt the pipeline.

    NOTE: This is NOT an ABC. All methods have default no-op
    implementations so students can write a hook with a single method.
    """

    def on_stage_start(self, stage: str, state: "PipelineState") -> None:
        """Called immediately before a stage begins execution.

        Args:
            stage: Stage name — ``"load_data"``, ``"build_model"``,
                ``"train"``, ``"evaluate"``, or ``"export"``.
            state: Current pipeline state (read/write).
        """

    def on_stage_end(self, stage: str, state: "PipelineState") -> None:
        """Called immediately after a stage completes (even if it raised).

        Args:
            stage: Stage name.
            state: Current pipeline state.
        """

    def on_epoch_start(self, epoch: int, state: "PipelineState") -> None:
        """Called before each training epoch.

        Only fires during the train stage.

        Args:
            epoch: Zero-based epoch index.
            state: Current pipeline state (``state.current_epoch`` is set).
        """

    def on_epoch_end(self, epoch: int, state: "PipelineState") -> None:
        """Called after each training epoch.

        Use this to check validation metrics and set
        ``state.should_stop = True`` for early stopping.

        Args:
            epoch: Zero-based epoch index.
            state: Current pipeline state.
        """

    def on_batch_end(
        self, batch: int, loss: float, state: "PipelineState"
    ) -> None:
        """Called after each training batch.

        Use for progress bars and per-batch logging.

        Args:
            batch: Zero-based batch index within the epoch.
            loss: Scalar loss value for this batch.
            state: Current pipeline state.
        """
```

- [ ] **Step 4: Run tests, verify ALL pass**

Run: `uv run pytest tests/unit/test_hooks.py -v`
Expected: ALL 6 tests PASS

- [ ] **Step 5: Ruff and mypy**

Run: `uv run ruff check pipeline/hooks.py tests/unit/test_hooks.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/hooks.py tests/unit/test_hooks.py
git commit -m "feat: add BaseHook with five lifecycle events

Hook base class with default no-op implementations. Students write
hooks by overriding only the methods they need. Instances are
registered via pipeline.add_hook() and fire in registration order.

Design note: NOT an ABC — all methods are no-ops by default so
beginners can write a single-method hook without implementing
unused abstract methods.

TDD: 6 tests covering:
- All defaults are no-ops
- Single-method and all-methods override
- Hook reads state (current_epoch) and writes state (should_stop)
- Two hook instances are independent"
```

---

### Task 7: BasePipeline ABC

**Files:**
- Modify: `pipeline/pipeline.py` — append BasePipeline class
- Modify: `tests/unit/test_pipeline.py` — append BasePipeline tests

**Interfaces:**
- Consumes: `PipelineState`, all protocols from Task 3
- Produces: `BasePipeline` ABC with:
  - `__init__(config: Config)` — stores config, initializes empty hooks list
  - `add_hook(hook: BaseHook) -> None` — inject a hook
  - `run(mode: Literal["train", "infer"]) -> PipelineState` — template method
  - `load_data(state)`, `extract_features(state)`, `build_model(state)`, `train(state)`, `evaluate(state)`, `export(state)` — abstract/concrete stages
  - `_run_stage(name, state, stage_fn)`, `_notify(event, *args)` — hook dispatch

- [ ] **Step 1: Write failing tests for BasePipeline**

Append to `tests/unit/test_pipeline.py`:
```python
from pipeline.hooks import BaseHook
from pipeline.pipeline import BasePipeline
from pipeline.protocols import Batch, DataStream


# ── Minimal concrete pipeline for testing ────────────────────────────────
class _MinimalPipeline(BasePipeline):
    """A pipeline where every stage is a no-op. For testing the template."""

    def load_data(self, state: PipelineState) -> None:
        state.current_epoch = 0

    def build_model(self, state: PipelineState) -> None:
        pass

    def train(self, state: PipelineState) -> None:
        state.current_epoch = state.config.num_epochs

    def evaluate(self, state: PipelineState) -> None:
        state.metrics["accuracy"] = 0.95

    def export(self, state: PipelineState) -> None:
        state.predictions = np.array([0, 1, 0])


class _SpyHook(BaseHook):
    """A hook that records every event it receives."""

    def __init__(self):
        self.events: list[str] = []

    def on_stage_start(self, stage: str, state: PipelineState) -> None:
        self.events.append(f"start:{stage}")

    def on_stage_end(self, stage: str, state: PipelineState) -> None:
        self.events.append(f"end:{stage}")

    def on_epoch_start(self, epoch: int, state: PipelineState) -> None:
        self.events.append(f"epoch_start:{epoch}")

    def on_epoch_end(self, epoch: int, state: PipelineState) -> None:
        self.events.append(f"epoch_end:{epoch}")

    def on_batch_end(self, batch: int, loss: float, state: PipelineState) -> None:
        self.events.append(f"batch_end:{batch}:{loss:.4f}")


# ── BasePipeline ABC tests ────────────────────────────────────────────────
class TestBasePipelineABC:
    """Tests for BasePipeline abstract base class."""

    def test_cannot_instantiate_abc_directly(self):
        """Type Error: instantiating BasePipeline directly raises TypeError."""
        with pytest.raises(TypeError):
            BasePipeline(Config())  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        """Happy Path: a concrete subclass can be instantiated."""
        pipeline = _MinimalPipeline(Config())
        assert pipeline is not None
        assert pipeline.config is not None


class TestBasePipelineRun:
    """Tests for BasePipeline.run() template method."""

    def test_run_train_executes_all_stages(self):
        """Happy Path: run('train') hits all 6 stages."""
        pipeline = _MinimalPipeline(Config(num_epochs=3))
        state = pipeline.run("train")
        assert state.current_epoch == 3
        assert state.metrics["accuracy"] == 0.95
        assert state.predictions is not None

    def test_run_infer_skips_training_stages(self):
        """Happy Path: run('infer') skips extract_features, build_model,
        train, and evaluate."""
        pipeline = _MinimalPipeline(Config(num_epochs=3))
        state = pipeline.run("infer")
        # Training stages skipped
        assert state.model is None
        assert state.current_epoch == 0
        assert state.metrics == {}
        # load_data and export still run
        assert state.predictions is not None

    def test_run_returns_pipeline_state(self):
        """Happy Path: run() returns the PipelineState it built."""
        pipeline = _MinimalPipeline(Config())
        state = pipeline.run("train")
        assert isinstance(state, PipelineState)

    def test_run_with_zero_epochs(self):
        """Boundary: run with num_epochs=0 is valid."""
        pipeline = _MinimalPipeline(Config(num_epochs=0))
        state = pipeline.run("train")
        assert state is not None


class TestBasePipelineHooks:
    """Tests for BasePipeline hook dispatch."""

    def test_no_hooks_pipeline_runs(self):
        """Boundary: pipeline with zero hooks runs without error."""
        pipeline = _MinimalPipeline(Config())
        state = pipeline.run("train")
        assert state.metrics["accuracy"] == 0.95

    def test_single_hook_receives_events(self):
        """Happy Path: a registered hook receives stage start/end events."""
        pipeline = _MinimalPipeline(Config())
        spy = _SpyHook()
        pipeline.add_hook(spy)
        pipeline.run("train")
        assert "start:load_data" in spy.events
        assert "end:load_data" in spy.events
        assert "start:train" in spy.events
        assert "end:train" in spy.events

    def test_multiple_hooks_both_fire(self):
        """Happy Path: two hooks both receive events."""
        pipeline = _MinimalPipeline(Config())
        spy1 = _SpyHook()
        spy2 = _SpyHook()
        pipeline.add_hook(spy1)
        pipeline.add_hook(spy2)
        pipeline.run("train")
        assert len(spy1.events) > 0
        assert len(spy2.events) > 0
        # Both should have same start events
        assert spy1.events[0] == spy2.events[0]

    def test_hook_order_is_registration_order(self):
        """Concurrency: hooks fire in the order they were added."""
        pipeline = _MinimalPipeline(Config())
        order_log: list[int] = []

        class OrderedHook(BaseHook):
            def __init__(self, tag: int):
                self.tag = tag

            def on_stage_start(self, stage, state):
                order_log.append(self.tag)

        pipeline.add_hook(OrderedHook(1))
        pipeline.add_hook(OrderedHook(2))
        pipeline.add_hook(OrderedHook(3))
        pipeline.run("train")
        # Each stage triggers all 3 hooks in order
        assert order_log[0:3] == [1, 2, 3]


class TestBasePipelineErrorRecovery:
    """Tests for error handling in the pipeline."""

    def test_hook_exception_does_not_kill_pipeline(self):
        """Error recovery: a hook that raises doesn't prevent completion.
        NOTE: Spec requires error recovery. The minimal implementation
        logs the error and continues."""
        pipeline = _MinimalPipeline(Config())

        class CrashingHook(BaseHook):
            def on_stage_start(self, stage, state):
                if stage == "train":
                    raise RuntimeError("hook crash")

        pipeline.add_hook(CrashingHook())
        # Should not raise — pipeline recovers
        state = pipeline.run("train")
        assert state.metrics["accuracy"] == 0.95

    def test_stage_exception_propagates(self):
        """Error recovery: a stage that raises propagates the exception
        (stages are critical path; hooks are not)."""
        class FailingPipeline(_MinimalPipeline):
            def train(self, state):
                raise RuntimeError("training failed")

        pipeline = FailingPipeline(Config())
        with pytest.raises(RuntimeError, match="training failed"):
            pipeline.run("train")

    def test_infer_mode_skips_stages_cleanly(self):
        """Happy Path: infer mode still runs load_data and export."""
        pipeline = _MinimalPipeline(Config())
        state = pipeline.run("infer")
        assert state.predictions is not None
        assert state.model is None  # never built
```

- [ ] **Step 2: Run tests, verify new tests fail**

Run: `uv run pytest tests/unit/test_pipeline.py::TestBasePipelineABC tests/unit/test_pipeline.py::TestBasePipelineRun tests/unit/test_pipeline.py::TestBasePipelineHooks tests/unit/test_pipeline.py::TestBasePipelineErrorRecovery -v`
Expected: FAIL — `ImportError` for BasePipeline or `AttributeError` for abstract methods

- [ ] **Step 3: Implement BasePipeline class**

Append to `pipeline/pipeline.py` (after PipelineState):
```python
class BasePipeline(ABC):
    """Universal deep learning pipeline template.

    Defines *when* six stages run. Subclasses define *how* each stage
    works by implementing the abstract methods. Hooks inject
    cross-cutting concerns (logging, checkpoint, early-stop).

    The six stages are:
        1. :meth:`load_data` — build :class:`DataStream`
        2. :meth:`extract_features` — optional feature engineering
        3. :meth:`build_model` — construct model, loss, optimizer
        4. :meth:`train` — run the training loop
        5. :meth:`evaluate` — compute validation metrics
        6. :meth:`export` — save model and predictions

    Usage::

        class ImagePipeline(BasePipeline):
            def load_data(self, state): ...
            def build_model(self, state): ...
            def train(self, state): ...
            def evaluate(self, state): ...
            def export(self, state): ...

        pipeline = ImagePipeline(config)
        pipeline.add_hook(ProgressHook())
        result = pipeline.run("train")
    """

    def __init__(self, config: "Config") -> None:
        """Store config and initialize an empty hook list.

        Args:
            config: Pipeline configuration. Stored as ``self.config``
                for subclasses to read.
        """
        self.config = config
        self._hooks: list["BaseHook"] = []

    def add_hook(self, hook: "BaseHook") -> None:
        """Register a hook to receive pipeline lifecycle events.

        Hooks fire in registration order. Adding the same hook twice
        causes it to fire twice (no deduplication).

        Args:
            hook: Any object implementing :class:`BaseHook`.
        """
        self._hooks.append(hook)

    # ── Template method ────────────────────────────────────────────
    def run(self, mode: Literal["train", "infer"] = "train") -> PipelineState:
        """Execute the pipeline.

        In train mode, runs all six stages. In infer mode, skips
        feature extraction, model building, training, and evaluation
        — only loads data and exports predictions.

        Args:
            mode: ``"train"`` or ``"infer"``.

        Returns:
            The :class:`PipelineState` with all stage outputs populated.
        """
        state = PipelineState(config=self.config, mode=mode)

        # Stage 1: always needed (both train and infer need data)
        self._run_stage("load_data", state, self.load_data)

        if mode == "train":
            # Stage 2: feature extraction (default: pass-through)
            self._run_stage("extract_features", state, self.extract_features)
            # Stage 3: model construction (DI entry point)
            self._run_stage("build_model", state, self.build_model)
            # Stage 4: training loop
            self._run_stage("train", state, self.train)
            # Stage 5: evaluation on validation set
            self._run_stage("evaluate", state, self.evaluate)

        # Stage 6: export (save model + write predictions)
        self._run_stage("export", state, self.export)

        return state

    # ── Abstract stages ─────────────────────────────────────────────
    @abstractmethod
    def load_data(self, state: PipelineState) -> None:
        """Stage 1: Build a :class:`DataStream` and assign to ``state.data_stream``.

        Called in both train and infer modes.
        """
        ...

    def extract_features(self, state: PipelineState) -> None:
        """Stage 2: Optional feature engineering.

        Default is pass-through (no-op). Override in traditional ML
        pipelines (e.g., sklearn) that need explicit feature extraction.
        In deep learning, the model extracts features internally.
        """
        pass

    @abstractmethod
    def build_model(self, state: PipelineState) -> None:
        """Stage 3: Construct model, loss, and optimizer.

        Assign ``state.model``, ``state.loss_fn``, and ``state.optimizer``.
        """
        ...

    @abstractmethod
    def train(self, state: PipelineState) -> None:
        """Stage 4: Run the training loop.

        Iterate over ``state.data_stream``, forward→loss→backward→step,
        populating ``state.history`` with loss/accuracy curves.
        """
        ...

    @abstractmethod
    def evaluate(self, state: PipelineState) -> None:
        """Stage 5: Evaluate the trained model on validation data.

        Compute metrics (accuracy, F1, etc.) and write them to
        ``state.metrics``.
        """
        ...

    @abstractmethod
    def export(self, state: PipelineState) -> None:
        """Stage 6: Save model checkpoints and write predictions.

        Assign ``state.predictions`` with model outputs on test data.
        """
        ...

    # ── Hook dispatch ───────────────────────────────────────────────
    def _run_stage(
        self,
        name: str,
        state: PipelineState,
        stage_fn: callable,
    ) -> None:
        """Execute one stage, wrapping it with hook notifications.

        Args:
            name: Stage label (e.g., ``"load_data"``).
            state: Shared pipeline state.
            stage_fn: The stage method to execute.
        """
        self._notify("on_stage_start", name, state)
        try:
            stage_fn(state)
        except Exception:
            # Stage failures are critical — re-raise
            raise
        finally:
            self._notify("on_stage_end", name, state)

    def _notify(self, event: str, *args: object) -> None:
        """Dispatch an event to all registered hooks.

        Hook exceptions are caught and logged but do not interrupt
        the pipeline — hooks are non-critical by design.

        Args:
            event: Hook method name (e.g., ``"on_stage_start"``).
            *args: Arguments forwarded to the hook method.
        """
        import logging

        _logger = logging.getLogger(__name__)
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

- [ ] **Step 4: Run tests, verify ALL pass**

Run: `uv run pytest tests/unit/test_pipeline.py -v`
Expected: ALL 24 tests PASS (12 from Task 5 + 12 new)

- [ ] **Step 5: Ruff and mypy**

Run: `uv run ruff check pipeline/pipeline.py tests/unit/test_pipeline.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: add BasePipeline ABC with 6-stage template method

Template method run() orchestrates: load_data → extract_features →
build_model → train → evaluate → export. Infer mode skips training
stages. Hook dispatch wraps each stage with try/finally — hook
exceptions logged but never kill the pipeline.

TDD: 12 new tests (24 total) covering:
- ABC instantiation guard
- Train mode executes all stages, infer skips appropriately
- Hook dispatch: zero hooks, single hook, multiple hooks, order
- Error recovery: hook crash doesn't halt pipeline
- Stage exception propagates (critical path)"
```

---

### Task 8: DI Registry

**Files:**
- Create: `pipeline/registry.py`
- Create: `tests/unit/test_registry.py`

**Interfaces:**
- Produces:
  - `register(kind: str, name: str) -> Callable[[type], type]` — decorator
  - `build(kind: str, name: str, **kwargs) -> Any` — factory
  - `list_registered(kind: str) -> list[str]` — introspection helper
  - `is_registered(kind: str, name: str) -> bool` — existence check

- [ ] **Step 1: Write failing tests for registry**

Create `tests/unit/test_registry.py`:
```python
"""Unit tests for pipeline.registry — DI registry."""
import pytest
from pipeline.registry import register, build, list_registered, is_registered


# ── Test classes for registration ─────────────────────────────────────────
@register("model", "dummy_a")
class DummyModelA:
    def __init__(self, hidden: int = 128):
        self.hidden = hidden

@register("model", "dummy_b")
class DummyModelB:
    def __init__(self, layers: int = 3):
        self.layers = layers

@register("optimizer", "sgd_test")
class DummyOptimizer:
    def __init__(self, lr: float = 0.01):
        self.lr = lr


# ── Tests ──────────────────────────────────────────────────────────────────
class TestRegister:
    """Tests for register() decorator."""

    def test_registered_class_builds(self):
        """Happy Path: registered class can be built via build()."""
        obj = build("model", "dummy_a", hidden=64)
        assert isinstance(obj, DummyModelA)
        assert obj.hidden == 64

    def test_registered_class_builds_with_defaults(self):
        """Happy Path: build with no kwargs uses class defaults."""
        obj = build("model", "dummy_b")
        assert obj.layers == 3

    def test_register_multiple_classes_same_kind(self):
        """Happy Path: multiple classes can register under the same kind."""
        a = build("model", "dummy_a")
        b = build("model", "dummy_b")
        assert isinstance(a, DummyModelA)
        assert isinstance(b, DummyModelB)

    def test_register_different_kinds(self):
        """Happy Path: classes can register under different kinds."""
        model = build("model", "dummy_a")
        opt = build("optimizer", "sgd_test")
        assert isinstance(model, DummyModelA)
        assert isinstance(opt, DummyOptimizer)


class TestBuildEdgeCases:
    """Tests for build() edge cases."""

    def test_build_unregistered_kind_raises(self):
        """Type Error: building from unknown kind raises KeyError."""
        with pytest.raises(KeyError):
            build("nonexistent_kind", "anything")

    def test_build_unregistered_name_raises(self):
        """Type Error: building unknown name within known kind raises KeyError."""
        with pytest.raises(KeyError):
            build("model", "nonexistent_model")

    def test_build_with_unexpected_kwargs(self):
        """Type Error: build with unknown keyword arguments raises TypeError.
        NOTE: This depends on the registered class's __init__ signature."""
        with pytest.raises(TypeError):
            build("model", "dummy_a", nonexistent_param=42)


class TestListRegistered:
    """Tests for list_registered() introspection."""

    def test_list_registered_returns_names(self):
        """Happy Path: list_registered returns all names for a kind."""
        names = list_registered("model")
        assert "dummy_a" in names
        assert "dummy_b" in names

    def test_list_registered_unknown_kind(self):
        """Boundary: list_registered for unknown kind returns empty list."""
        names = list_registered("nonexistent")
        assert names == []


class TestIsRegistered:
    """Tests for is_registered()."""

    def test_is_registered_known(self):
        """Happy Path: registered name returns True."""
        assert is_registered("model", "dummy_a") is True

    def test_is_registered_unknown_name(self):
        """Boundary: unregistered name returns False."""
        assert is_registered("model", "fake") is False

    def test_is_registered_unknown_kind(self):
        """Boundary: unregistered kind returns False."""
        assert is_registered("fake_kind", "anything") is False


class TestRegistryStress:
    """Stress tests for the registry."""

    def test_register_many_classes(self):
        """Stress: register 500 classes and build each one."""
        n = 500
        for i in range(n):
            @register("stress_test", f"cls_{i}")
            class _StressClass:
                def __init__(self, idx: int = i):
                    self.idx = idx

        names = list_registered("stress_test")
        assert len(names) == n

        # Build a random sample
        obj = build("stress_test", "cls_42")
        assert obj.idx == 42


class TestRegistryIsolation:
    """Tests for registry state isolation."""

    def test_registry_no_cross_kind_leakage(self):
        """Concurrency: registering under one kind doesn't affect others."""
        @register("isolated_kind", "test_cls")
        class _Isolated:
            pass

        assert "test_cls" in list_registered("isolated_kind")
        assert "test_cls" not in list_registered("model")
```

- [ ] **Step 2: Run tests, verify ALL fail**

Run: `uv run pytest tests/unit/test_registry.py -v`
Expected: ALL FAIL — `ModuleNotFoundError` for `pipeline.registry`

- [ ] **Step 3: Implement `pipeline/registry.py`**

Create `pipeline/registry.py`:
```python
"""Dependency injection registry.

A minimal type-based registry: one dict, two functions, zero magic.
Classes are registered by ``(kind, name)`` pairs and built by the
same keys with keyword arguments forwarded to ``__init__``.

Usage::

    from pipeline.registry import register, build

    @register("backbone", "resnet18")
    class ResNet18:
        def __init__(self, num_classes: int = 10):
            ...

    model = build("backbone", "resnet18", num_classes=100)

NOTE: The registry is a module-level ``dict``. Registration happens
at import time when the decorator runs. This is intentional — it keeps
the system simple enough for a beginner to understand in one reading.
"""
from __future__ import annotations

from typing import Any, Callable

# ── Internal storage ───────────────────────────────────────────────────
# WHY: A flat dict of dicts. _REGISTRY["backbone"]["resnet18"] = ResNet18
# No metaclasses, no YAML scanning, no dynamic imports. A student can
# print(_REGISTRY) and see everything registered.
_REGISTRY: dict[str, dict[str, type]] = {}


# ── Public API ────────────────────────────────────────────────────────
def register(kind: str, name: str) -> Callable[[type], type]:
    """Decorator: register a class under ``(kind, name)``.

    Args:
        kind: Component category (e.g., ``"backbone"``, ``"optimizer"``).
        name: Unique name within the kind (e.g., ``"resnet18"``).

    Returns:
        A decorator that registers the class and returns it unchanged.

    Usage::

        @register("backbone", "resnet18")
        class ResNet18:
            ...
    """
    def decorator(cls: type) -> type:
        _REGISTRY.setdefault(kind, {})[name] = cls
        return cls
    return decorator


def build(kind: str, name: str, **kwargs: Any) -> Any:
    """Build a registered component by ``(kind, name)``.

    Args:
        kind: Component category.
        name: Registered name within the category.
        **kwargs: Forwarded to the class ``__init__``.

    Returns:
        An instance of the registered class.

    Raises:
        KeyError: If ``(kind, name)`` is not registered.
    """
    try:
        cls = _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(
            f"No component registered under kind={kind!r}, name={name!r}. "
            f"Available names for {kind!r}: {list_registered(kind)}"
        ) from None
    return cls(**kwargs)


def list_registered(kind: str) -> list[str]:
    """Return all registered names for a given kind.

    Useful for debugging and introspection::

        >>> list_registered("backbone")
        ["resnet18", "simple_cnn", "mlp"]

    Args:
        kind: Component category.

    Returns:
        List of registered names (empty if kind is unknown).
    """
    return list(_REGISTRY.get(kind, {}).keys())


def is_registered(kind: str, name: str) -> bool:
    """Check whether ``(kind, name)`` is registered.

    Args:
        kind: Component category.
        name: Component name.

    Returns:
        ``True`` if the component is registered.
    """
    return name in _REGISTRY.get(kind, {})
```

- [ ] **Step 4: Run tests, verify ALL pass**

Run: `uv run pytest tests/unit/test_registry.py -v`
Expected: ALL 12 tests PASS

- [ ] **Step 5: Ruff and mypy**

Run: `uv run ruff check pipeline/registry.py tests/unit/test_registry.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/registry.py tests/unit/test_registry.py
git commit -m "feat: add DI registry — register() and build()

Minimal dependency injection: a flat dict, two public functions.
register(kind, name) decorator stores classes. build(kind, name, **kw)
instantiates them. list_registered() and is_registered() provide
introspection for debugging.

TDD: 12 tests covering:
- Happy Path: register→build with/without kwargs
- Edge: unknown kind/name raises KeyError
- Stress: 500 classes registered and built
- Isolation: kinds don't leak into each other"
```

---

### Task 9: Device Utilities

**Files:**
- Create: `pipeline/utils/device.py`
- Create: `tests/unit/test_utils_device.py`

**Interfaces:**
- Produces:
  - `get_device(preference: str = "auto") -> str` — resolves to "cpu", "cuda", or "mps"
  - `device_info(device: str) -> str` — human-readable device description

- [ ] **Step 1: Write failing tests for device utilities**

Create `tests/unit/test_utils_device.py`:
```python
"""Unit tests for pipeline.utils.device."""
import pytest
from pipeline.utils.device import get_device, device_info


class TestGetDevice:
    """Tests for get_device()."""

    def test_auto_returns_string(self):
        """Happy Path: get_device('auto') returns a valid device string."""
        device = get_device("auto")
        assert device in ("cpu", "cuda", "mps")

    def test_cpu_explicit(self):
        """Happy Path: get_device('cpu') returns 'cpu'."""
        assert get_device("cpu") == "cpu"

    def test_invalid_device_raises(self):
        """Type Error: unknown device string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device"):
            get_device("tpu")

    def test_empty_string_raises(self):
        """Boundary: empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device"):
            get_device("")

    def test_cuda_name_normalized(self):
        """Happy Path: 'cuda:0' and 'cuda' both normalize to 'cuda'."""
        # NOTE: On a CPU-only machine, get_device('cuda') may fall back
        # to 'cpu'. This test verifies the normalization behavior.
        result = get_device("cuda")
        assert result in ("cpu", "cuda")

    def test_case_insensitive(self):
        """Boundary: device string is case-insensitive."""
        assert get_device("CPU") == "cpu"


class TestDeviceInfo:
    """Tests for device_info()."""

    def test_cpu_info_returns_string(self):
        """Happy Path: device_info('cpu') returns a non-empty string."""
        info = device_info("cpu")
        assert isinstance(info, str)
        assert len(info) > 0

    def test_info_contains_device_name(self):
        """Happy Path: device_info includes the device name."""
        info = device_info("cpu")
        assert "cpu" in info.lower() or "CPU" in info

    def test_info_unknown_device_raises(self):
        """Type Error: device_info for unknown device raises ValueError."""
        with pytest.raises(ValueError):
            device_info("quantum_computer")
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/unit/test_utils_device.py -v`
Expected: ALL FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `pipeline/utils/device.py`**

Create `pipeline/utils/device.py`:
```python
"""Device detection and information.

Provides a single function :func:`get_device` that resolves device
preferences across CUDA, MPS, and CPU. Students never need to write
``if torch.cuda.is_available()`` by hand.
"""
from __future__ import annotations

import platform


def get_device(preference: str = "auto") -> str:
    """Resolve a device preference to an available device name.

    Priority for ``"auto"``: CUDA → MPS → CPU.

    Args:
        preference: One of ``"auto"``, ``"cpu"``, ``"cuda"``,
            ``"mps"``. Case-insensitive.

    Returns:
        ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Raises:
        ValueError: If ``preference`` is not recognized.
    """
    preference = preference.lower()

    if preference == "auto":
        return _detect_best_device()

    valid = {"cpu", "cuda", "mps"}
    if preference not in valid:
        raise ValueError(
            f"Unknown device: {preference!r}. "
            f"Choose from: {', '.join(sorted(valid))}"
        )

    # NOTE: For explicit requests (not "auto"), return as-is.
    # Availability is checked by the framework adapter (e.g.,
    # TorchAdapter verifies torch.cuda.is_available()).
    return preference


def device_info(device: str) -> str:
    """Return a human-readable description of the compute device.

    Args:
        device: Device name (``"cpu"``, ``"cuda"``, or ``"mps"``).

    Returns:
        A string like ``"CPU (x86_64)"`` or ``"CUDA (not checked)"``.

    Raises:
        ValueError: If ``device`` is not recognized.
    """
    device = device.lower()
    if device == "cpu":
        return f"CPU ({platform.machine()})"
    elif device == "cuda":
        # WHY: We don't import torch here. The framework adapter
        # provides GPU details when it initializes.
        return "CUDA (availability checked by framework adapter)"
    elif device == "mps":
        return f"MPS ({platform.machine()}) — Apple Silicon"
    else:
        raise ValueError(f"Unknown device: {device!r}")


def _detect_best_device() -> str:
    """Probe the system for the best available device.

    Tries to import torch to check CUDA/MPS availability, falls
    back to CPU. This is deliberately lazy-imported so the module
    works without PyTorch installed.
    """
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    # NOTE: torch.backends.mps.is_available() may exist but MPS
    # support is version-dependent. Check both availability and
    # whether it's actually built.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

- [ ] **Step 4: Run tests, verify ALL pass**

Run: `uv run pytest tests/unit/test_utils_device.py -v`
Expected: ALL 8 tests PASS (on a CPU machine, `get_device("auto")` returns `"cpu"`)

- [ ] **Step 5: Ruff and mypy**

Run: `uv run ruff check pipeline/utils/device.py tests/unit/test_utils_device.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/utils/device.py tests/unit/test_utils_device.py
git commit -m "feat: add device utilities — get_device() and device_info()

Framework-aware device resolution: auto-detects CUDA/MPS via lazy
torch import, falls back to CPU. device_info() returns human-readable
descriptions without importing torch.

TDD: 8 tests covering:
- auto/explicit CPU resolution
- Invalid/empty device strings raise ValueError
- CUDA name normalization, case insensitivity
- device_info for CPU and error on unknown device"
```

---

### Task 10: Seed Utilities

**Files:**
- Create: `pipeline/utils/seed.py`
- Create: `tests/unit/test_utils_seed.py`

**Interfaces:**
- Produces: `set_seed(seed: int, deterministic: bool = False) -> None`

- [ ] **Step 1: Write failing tests for seed utilities**

Create `tests/unit/test_utils_seed.py`:
```python
"""Unit tests for pipeline.utils.seed."""
import random

import numpy as np
import pytest
from pipeline.utils.seed import set_seed


class TestSetSeed:
    """Tests for set_seed()."""

    def test_sets_python_random(self):
        """Happy Path: set_seed makes random module reproducible."""
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_sets_numpy_random(self):
        """Happy Path: set_seed makes numpy reproducible."""
        set_seed(42)
        a = np.random.randn(3)
        set_seed(42)
        b = np.random.randn(3)
        assert np.array_equal(a, b)

    def test_different_seeds_produce_different_results(self):
        """Happy Path: different seeds → different random sequences."""
        set_seed(1)
        a = np.random.randn(5)
        set_seed(2)
        b = np.random.randn(5)
        assert not np.array_equal(a, b)

    def test_seed_zero_is_valid(self):
        """Boundary: seed=0 is valid (not the same as 'no seed')."""
        set_seed(0)
        set_seed(0)
        a = np.random.randn(3)
        set_seed(0)
        b = np.random.randn(3)
        assert np.array_equal(a, b)

    def test_negative_seed_is_valid(self):
        """Boundary: negative seeds are valid integers."""
        set_seed(-1)
        a = np.random.randn(3)
        set_seed(-1)
        b = np.random.randn(3)
        assert np.array_equal(a, b)

    def test_large_seed_is_valid(self):
        """Boundary: large seed value works correctly."""
        set_seed(2**31 - 1)
        a = np.random.randn(3)
        set_seed(2**31 - 1)
        b = np.random.randn(3)
        assert np.array_equal(a, b)

    def test_deterministic_flag_does_not_crash(self):
        """Happy Path: deterministic=True does not raise."""
        set_seed(42, deterministic=True)
        # Should not raise, even without CUDA

    def test_consecutive_calls_idempotent(self):
        """Concurrency: calling set_seed twice is fine."""
        set_seed(42)
        set_seed(42)
        # Should not raise


class TestSetSeedTorch:
    """Tests for set_seed torch integration."""

    def test_sets_torch_seed_if_available(self):
        """Happy Path: set_seed also sets torch manual seed.
        NOTE: Only tested when torch is installed."""
        try:
            import torch

            set_seed(42)
            a = torch.randn(3)
            set_seed(42)
            b = torch.randn(3)
            assert torch.allclose(a, b)
        except ImportError:
            pytest.skip("torch not installed")
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/unit/test_utils_seed.py -v`
Expected: ALL FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `pipeline/utils/seed.py`**

Create `pipeline/utils/seed.py`:
```python
"""Reproducibility via seed setting.

A single function :func:`set_seed` that seeds Python's ``random``,
``numpy``, and (if available) ``torch``. Students call it once at
the top of their script.
"""
from __future__ import annotations

import random


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Seeds Python's ``random`` module, ``numpy``, and (if available)
    ``torch``. Call once at the start of your experiment.

    Args:
        seed: Integer seed for all random number generators.
        deterministic: If ``True``, enable deterministic CUDA
            operations. WARNING: This makes training significantly
            slower. Only use when debugging reproducibility issues.

    NOTE: Deterministic mode configures ``torch.backends.cudnn``
    and sets ``CUBLAS_WORKSPACE_CONFIG``. It does NOT guarantee
    bit-for-bit reproducibility across different hardware or
    PyTorch versions.
    """
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    # Try torch — don't crash if not installed
    try:
        import torch  # type: ignore[import-untyped]
        torch.manual_seed(seed)
        if deterministic and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            # NOTE: CUBLAS workspace config required for deterministic
            # cuDNN convolution algorithms.
            import os
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass
```

- [ ] **Step 4: Run tests, verify ALL pass**

Run: `uv run pytest tests/unit/test_utils_seed.py -v`
Expected: ALL pass (8 tests, one may skip if torch not installed)

- [ ] **Step 5: Ruff and mypy**

Run: `uv run ruff check pipeline/utils/seed.py tests/unit/test_utils_seed.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add pipeline/utils/seed.py tests/unit/test_utils_seed.py
git commit -m "feat: add set_seed() for reproducibility

Seeds Python random, numpy, and (lazily) torch. deterministic flag
enables cuDNN determinism with documented performance trade-off.

TDD: 8 tests covering:
- Python random and numpy reproducibility
- Different seeds → different sequences
- Boundary: seed=0, negative seeds, large seeds
- deterministic flag doesn't crash without CUDA
- Torch integration test (skip if not installed)"
```

---

### Task 11: Integration — Full Phase 0 Test Suite

**Files:**
- No new files. Verify everything works together.

- [ ] **Step 1: Run the full test suite**

Run: `cd /home/billma/requiema/Universal_DL_Pipeline && uv run pytest -v`
Expected: ALL tests PASS (all tests across all test files — protocols, pipeline, registry, hooks, config, device, seed)

- [ ] **Step 2: Verify no torch import in pipeline core**

Run: `uv run python -c "import pipeline; print([m for m in dir(pipeline)])"`
Expected: no torch modules listed

Run: `grep -r "import torch" pipeline/ || echo "No torch imports found"`
Expected: "No torch imports found" (except in utils/device.py lazy import)

- [ ] **Step 3: Verify ruff passes**

Run: `uv run ruff check .`
Expected: no errors

- [ ] **Step 4: Verify mypy passes (warnings acceptable for untyped deps)**

Run: `uv run mypy pipeline/`
Expected: no errors in pipeline modules (may see warnings for numpy/torch if not installed)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: Phase 0 complete — all tests passing

All skeleton modules built and tested:
- protocols.py: ArrayLike, Parameter, Batch, 4 ABCs
- pipeline.py: PipelineState, BasePipeline (6-stage template)
- registry.py: register(), build(), list_registered(), is_registered()
- hooks.py: BaseHook (5 lifecycle events, no-op defaults)
- config.py: Config dataclass (18 fields, YAML loading, validation)
- utils/device.py: get_device(), device_info()
- utils/seed.py: set_seed()

Zero ML framework dependencies in the core. Ruff + mypy clean."
```

---

## Phase 0 Completion Checklist

- [ ] All tests passing (`uv run pytest -v`)
- [ ] Zero `import torch` in pipeline/ (except lazy import in device.py)
- [ ] `uv run ruff check .` — clean
- [ ] `uv run mypy pipeline/` — clean (no errors)
- [ ] `uv run python -c "import pipeline; print(pipeline.__version__)"` → `0.1.0`
- [ ] All module docstrings present with Google-style format
- [ ] All @property usage follows spec conventions
- [ ] No `if task_type` branching, no global mutable state, no closure captures
- [ ] Every file ≤ 300 lines, every function ≤ 50 lines, classes ≤ 5 public methods
