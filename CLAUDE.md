# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A **teaching-first, framework-agnostic deep learning pipeline library**. The pipeline defines *when* things happen (six stages: load → extract → build → train → evaluate → export). Concrete implementations — numpy, sklearn, PyTorch, or your own — define *how*. Protocols are the contract, not framework classes.

**Phase 2 (complete).** The core operates purely on protocols — no torch in `pipeline/`. Sklearn, numpy, and sympy adapters are built and tested. Contract tests verify both `SklearnModel` and `NumpyModel` satisfy `ModelProtocol`. Phase 3 (image/torch) is next. See `STATUS.md` for the full roadmap.

## Core architecture

### The six-stage pipeline (Template Method)

`BasePipeline` (`pipeline/pipeline.py`) is an ABC with six stages. `run("train")` executes all six; `run("infer")` runs only load_data + export. Subclasses implement the abstract methods:

1. `load_data(state)` — build a `DataStream`, assign to `state.data_stream`
2. `extract_features(state)` — optional feature engineering (default no-op)
3. `build_model(state)` — construct `state.model`, `state.loss_fn`, `state.optimizer`
4. `train(state)` — default delegates to `TrainLoop`; override for non-standard training
5. `evaluate(state)` — compute metrics on `state.val_data_stream`
6. `export(state)` — write predictions/checkpoints

### PipelineState — the shared data bus

`PipelineState` is a `@dataclass` passed through every stage. Each stage reads/writes specific fields. Control signals: `current_epoch` (tracked by TrainLoop), `should_stop` (set by early-stop hooks). Hooks observe but don't own the pipeline.

### Protocols — the framework-agnostic contract

`pipeline/protocols.py` defines the abstract interfaces every component must satisfy:

- **`DataStream`** — `__iter__` yields `Batch` objects, `__len__` returns batch count
- **`ModelProtocol`** — `forward()`, `parameters()`, `train_mode()`, `eval_mode()`
- **`LossProtocol`** — `forward(predictions, targets)` returns a `Loss` object
- **`OptimizerProtocol`** — `step()`, `zero_grad()`
- **`Parameter`** — wraps `data` (ArrayLike) + `grad` + `name`; optimizer mutates these in-place
- **`Batch`** — `inputs` + `targets` (ArrayLike each)
- **`Loss`** — wraps scalar `value` + optional `_backward_fn` callable (no-op for non-gradient models)
- **`ArrayLike`** — type alias (`np.ndarray | Any`) accepting any duck-typed array

### Registry — flat-dict DI

`pipeline/registry.py`: `_REGISTRY` is a dict-of-dicts (`_REGISTRY["kind"]["name"] = cls`). `@register("kind", "name")` decorator, `build("kind", "name", **kwargs)` factory. Zero metaclasses. Registration happens at import time. Students can `print(_REGISTRY)`.

### Hooks — cross-cutting concerns

Five hook points, all no-ops by default (`BaseHook` is not an ABC): `on_stage_start`, `on_stage_end`, `on_epoch_start`, `on_epoch_end`, `on_batch_end`. Called in registration order. Hook exceptions are **caught and logged** — they never interrupt the pipeline or training loop.

TrainLoop borrows the pipeline's hooks via `pipeline.hooks` property. `ProgressHook` wraps tqdm.

### Adapters — framework bridges

`pipeline/adapters/` provides `ModelProtocol` wrappers for sklearn and numpy:

- **`SklearnModel`** — wraps any sklearn estimator. `forward()` delegates to `predict_proba()` (classifiers) or `predict()` (regressors). `parameters()` returns `[]` — no gradient parameters. Use `StubLoss` + `StubOptimizer` to satisfy the pipeline contract since sklearn handles training internally via `.fit()`.
- **`NumpyModel`** — pure-numpy model with explicit `Parameter` objects. `forward()` caches intermediate activations; `backward(dL_doutput)` propagates gradients through all layers via the chain rule.
- **`NumpyOptimizer`** — bridges `OptimizerProtocol` with `SGD`/`Adam` update rules. Iterates over Parameter refs, delegates each to `rule.update(param)`.

### Training components

`pipeline/training/` includes `TrainLoop` plus pure-numpy optimizer rules (`SGD`, `Adam`) and loss functions (`MSELoss`, `CrossEntropyLoss`). Loss functions optionally attach a `NumpyModel` — when attached, `loss.backward()` propagates gradients to every parameter.

### Lazy imports

YAML (in `Config.from_yaml`), pandas (in `CsvDataSource._load` and `to_csv`), torch (in `get_device` and `set_seed`) are all imported inside functions — never at module level. The core works without any of them installed.

## Commands

```bash
# Install with dev deps (uses uv)
uv sync --dev

# Run all tests (225 unit + integration, ~0.3s)
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_metrics.py

# Run a single test by name
uv run pytest tests/unit/test_metrics.py::test_accuracy_perfect -v

# Run only unit tests (skip slow-marked integration)
uv run pytest -m "not slow"

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check (strict mode)
uv run mypy

# Build wheel (only pipeline/ goes in)
uv build --wheel

# Docs
uv run mkdocs build          # build site/
uv run mkdocs serve           # live preview at localhost:8000

# Module entry point
python -m pipeline            # version + device + registered components

# CLI entry point
python run.py --config config.yaml --mode train
```

## Build & package

- **Build backend:** hatchling, wheel only (`tool.hatch.build.targets.wheel` packages = `["pipeline"]`)
- **Python:** `>=3.11,<3.14` (uses `from __future__ import annotations` everywhere)
- **Core deps:** numpy, pandas, pyyaml, tqdm (no frameworks)
- **Dev deps:** pytest, pytest-cov, ruff, mypy, pandas-stubs, types-tqdm, scikit-learn, ipykernel, sympy, matplotlib
- **Docs deps:** mkdocs-material, mkdocstrings[python], mkdocs-static-i18n
- **Console script:** `dl-pipeline` → `pipeline.__main__:main`
- **Version:** 0.2.0

## Tooling conventions

- **Docs:** MkDocs Material + mkdocstrings, Google-style docstrings. Config in `mkdocs.yml`.
- **Linter:** ruff, line-length 100, target py311. Rules: E, F, I, N, W, UP, B, C4, SIM. One file-level ignore: `pipeline/pipeline.py` B027 (intentional no-op in `extract_features`).
- **Type checker:** mypy strict mode, python_version=3.12. Stub warnings expected on py3.13.
- **Formatter:** ruff format, double quotes, space indent.
- **WSL2:** `uv` link-mode is `copy` (cross-filesystem compatibility in `pyproject.toml`).

## Testing

- **283 tests** in `tests/unit/` (per-module), `tests/contract/` (adapter contract), and `tests/integration/` (E2E).
- **Test doubles** live in `tests/unit/conftest.py`: `FakeDataStream`, `FakeModel` (x→2*x), `FakeModelWithParams` (linear Wx+b with mutable Parameter refs), `FakeLoss` (MSE), `FakeOptimizer` (call-counting).
- **Fixture:** `tiny_titanic.csv` (10 rows, 4 cols, committed) at `tests/fixtures/`.
- Integration E2E test validates the full pipeline: CSV → TrainTestSplit → TrainLoop → Metrics → to_csv, plus infer-mode skip behavior.
- pytest config: `-v --tb=short`, marker `slow` for integration/E2E tests.
- TDD: every new module was written with tests first — this is a teaching project so test coverage is the mechanism for proving things work without a real framework.
- **No mocking frameworks** — only fakes and stubs. This is intentional: the codebase should be readable by students who don't know mock/patch.

## Phase roadmap (from STATUS.md)

Phase 0–2 complete. Phase 3 (NEXT) — image + TorchAdapter → M4. Phase 4 — production inference/checkpointing. Phase 5 — HPO + ensemble. Phase 6 — sequences/NLP.

## Key design rules

1. **No framework in core.** The `pipeline/` package never imports torch, sklearn, or sympy. Framework adapters live in separate subpackages and satisfy the protocols.
2. **Protocols are the boundary.** Any component that touches model/data/optimizer goes through the protocol interfaces. Never couple to a specific framework class.
3. **Fake models are valid.** Phase 1 uses fake models exclusively. This is by design — the pipeline structure is validated before real models are added.
4. **Lazy imports for optional deps.** YAML, pandas, torch are imported inside functions. The core module must import without them.
5. **Hooks don't interrupt.** Hook exceptions are logged and swallowed. A broken progress bar should never crash training.
6. **Google-style docstrings** on all public API. mkdocstrings generates API docs from them.
