# Phase 1: Pipeline Framework Design

Date: 2026-07-29
Status: pending review
Parent spec: [2026-07-28-pipeline-architecture-design.md](./2026-07-28-pipeline-architecture-design.md)

## 1. Overview

Phase 1 builds the pipeline **framework layer** — concrete implementations that operate on
Phase 0's abstract protocols. It introduces `numpy` and `pandas` for computation but still
**no torch, sklearn, or sympy**.

Phase 1 also makes targeted amendments to Phase 0 where the TrainLoop integration reveals
gaps in the original protocol design.

### Design goals

- **TrainLoop owns training-time hook dispatch.** Like a game engine's main loop, TrainLoop
  manages the lifecycle events inside training (epoch start/end, batch end). BasePipeline
  retains stage-level events (on_stage_start/end).
- **PipelineState is a state machine.** It holds the shared data bus and provides control
  signals (`should_stop`, `current_epoch`) that TrainLoop reads/writes.
- **SOLID compliance.** Each component has one reason to change. Metrics container decouples
  evaluate() from specific metric functions.

---

## 2. Phase 0 Amendments

### 2.1 Loss class (new in `protocols.py`)

**Problem:** `LossProtocol.forward()` returns `float`, but `float` has no `.backward()` method.
TrainLoop needs to call `.backward()` on the loss result to compute gradients.

**Solution:** Introduce a `Loss` value object that wraps the scalar and an optional backward hook.

```python
@dataclass
class Loss:
    """Loss computation result.

    ``float(loss)`` extracts the scalar; ``loss.backward()`` computes
    gradients into :attr:`Parameter.grad`. The backward pass is a no-op
    for non-gradient models (sklearn, Phase 1 fake models).
    """

    value: float
    _backward_fn: Callable[[], None] | None = None

    def backward(self) -> None:
        if self._backward_fn is not None:
            self._backward_fn()

    def __float__(self) -> float:
        return self.value
```

`LossProtocol.forward()` return type changes from `-> float` to `-> Loss`.

### 2.2 PipelineState: `val_data_stream` field (new)

**Problem:** Phase 0's `PipelineState` has only `data_stream`. After TrainTestSplit,
evaluate() needs a separate validation stream.

**Solution:** Add `val_data_stream: DataStream | None = None`. Backward compatible — existing
tests don't touch this field.

### 2.3 PipelineState: `metrics` type change

**Problem:** `metrics: dict[str, float]` is just a result box. `evaluate()` must hardcode which
metrics to compute, violating OCP.

**Solution:** New `Metrics` class (see §3.3). `PipelineState.metrics` changes from
`dict[str, float]` to `Metrics`.

### 2.4 BasePipeline: `hooks` read-only property (new)

**Problem:** TrainLoop needs read access to the hook list for training-time events. But
`_hooks` is a private attribute of BasePipeline — external classes shouldn't touch it.

**Solution:** Public read-only `@property`:

```python
class BasePipeline:
    @property
    def hooks(self) -> list[BaseHook]:
        """Registered hooks. Read-only; use add_hook() to register."""
        return self._hooks
```

### 2.5 BasePipeline: `train()` default implementation

**Problem:** `train()` is abstract. Every concrete pipeline must write its own epoch/batch loop
boilerplate, or delegate to TrainLoop. Delegation should be the default.

**Solution:** `train()` gets a default implementation that delegates to `TrainLoop`.
Subclasses can still override for non-standard loops (GANs, meta-learning).

```python
def train(self, state: PipelineState) -> None:
    """Default: delegate to TrainLoop."""
    # WHY: Lazy import inside the method avoids circular imports — TrainLoop
    # imports PipelineState from pipeline.pipeline, but at this point both
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

---

## 3. Phase 1 New Components

### 3.1 CsvDataSource (`pipeline/data/csv_source.py`)

Implements `DataStream`. Reads a CSV file into a pandas DataFrame, splits into batches,
yields `Batch` objects.

```
Constructor:
    file_path: str
    batch_size: int = 32
    target_column: str | None = None  (default: last column)
    shuffle: bool = True
    seed: int = 42

__iter__ → Iterator[Batch]:
    Lazy-load CSV on first call → shuffle indices → slice into batch_size windows

__len__ → int:
    Number of batches (ceil(n_samples / batch_size))
```

Lazy-loading: CSV is read only when `__iter__` or `__len__` is first called, not during
`__init__`. This keeps construction cheap and defers I/O.

### 3.2 train_test_split (`pipeline/data/split.py`)

Pure function, not a class. Takes a `DataStream`, returns two.

```python
def train_test_split(
    source: DataStream,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[DataStream, DataStream]:
    """Split a DataStream into (train_stream, val_stream).

    Collects all data → splits indices → returns two InMemoryDataStream
    instances backed by numpy arrays.
    """
```

Implementation detail: for Phase 1, data is collected into memory (Titanic is small).
The two returned streams are private `_InMemoryDataStream` instances — a simple DataStream
implementation backed by pre-split numpy arrays. This class is internal to `split.py`; it
is not part of the public API. A streaming split can be added later for large datasets.

### 3.3 Metrics (`pipeline/evaluation/metrics.py`)

#### Metrics container class

Holds both the metric functions to compute and their results. Decouples `evaluate()` from
knowledge of which specific metrics are being used.

```python
class Metrics:
    """Collection of metric functions with cached results.

    Usage::

        state.metrics = Metrics(accuracy=accuracy, f1=f1_score)
        state.metrics.compute(y_true, y_pred)
        print(state.metrics["accuracy"])  # → 0.92
    """

    def __init__(self, **named_metrics: Callable[[ArrayLike, ArrayLike], float]): ...
    def compute(self, y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]: ...
    def __getitem__(self, name: str) -> float: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __contains__(self, name: str) -> bool: ...
```

`compute()` does NOT accumulate across batches — it expects full y_true/y_pred arrays and
computes all metrics in one pass. This is appropriate for Phase 1's in-memory evaluation.
Streaming accumulation can be added as a `MetricProtocol` in a later phase if needed.

`__getitem__` raises `KeyError` if the metric name is not registered. If `compute()` has
not been called yet, `_values` is empty and any access raises `KeyError`.

#### Pure metric functions

```python
def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float: ...
def precision(y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary") -> float: ...
def recall(y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary") -> float: ...
def f1_score(y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary") -> float: ...
def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, num_classes: int | None = None) -> np.ndarray: ...
```

Pure numpy, no imports beyond numpy. Each function is independently testable and importable.

### 3.4 TrainLoop (`pipeline/training/train_loop.py`)

The training lifecycle manager. Owns training-time hook dispatch.

```
Constructor:
    model: ModelProtocol
    data_stream: DataStream
    optimizer: OptimizerProtocol
    loss_fn: LossProtocol
    num_epochs: int
    hooks: list[BaseHook] | None = None

run(state: PipelineState) -> None:
    model.train_mode()
    for epoch in range(num_epochs):
        state.current_epoch = epoch
        _notify("on_epoch_start", epoch, state)
        for batch_idx, batch in enumerate(data_stream):
            loss = _train_step(batch)
            _notify("on_batch_end", batch_idx, loss, state)
        _notify("on_epoch_end", epoch, state)
        if state.should_stop: break
    state.history = {"loss": [...]}

_train_step(batch: Batch) -> float:
    predictions = model.forward(batch.inputs)
    loss = loss_fn(predictions, batch.targets)
    params = list(model.parameters())
    if params:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss)
```

**Key behaviors:**
- Parameters-empty check: if `model.parameters()` is empty (sklearn/fake model), skip
  backward/step entirely.
- Hook exceptions are caught and logged — never kill the training loop.
- `should_stop` is checked after each epoch, enabling early stopping hooks (Phase 3).

`_notify()` mirrors `BasePipeline._notify()` — catches hook exceptions, logs them, continues.

**Line budget:** ≤ 80 lines (including `_notify` and `_train_step`).

### 3.5 ProgressHook (`pipeline/hooks/progress.py`)

Materializes the abstract `BaseHook` methods for training-time events.

```
on_epoch_start: create tqdm progress bar (total = len(data_stream))
on_batch_end:   update progress bar, display current batch loss
on_epoch_end:   close bar, print epoch summary (avg loss)
```

on_stage_start/end are no-ops (inherited from BaseHook). tqdm is the only new dependency
introduced by this hook.

### 3.6 to_csv (`pipeline/export/to_csv.py`)

Single pure function:

```python
def to_csv(array: ArrayLike, path: str | Path, columns: list[str] | None = None) -> None:
    """Write numpy array or array-like to a CSV file via pandas."""
```

### 3.7 CLI Entry Points

#### `pipeline/__main__.py`

```bash
python -m pipeline
```

Prints: version number, device info, all registered components keyed by kind.

#### `run.py` (project root)

```bash
python run.py --config config.yaml --mode train
python run.py --config config.yaml --mode infer
python run.py --help
```

Parses CLI args, loads Config from YAML, instantiates the concrete pipeline, registers hooks,
calls `run()`, prints final metrics.

---

## 4. Data Flow (train mode)

```
run.py
  │  Config.from_yaml("config.yaml")
  ▼
TitanicPipeline.run("train")
  │
  ├─[1] load_data(state)
  │      CsvDataSource("data/titanic/train.csv") → full_stream
  │      train_test_split(full_stream, train_ratio=0.8) → (train, val)
  │      state.data_stream = train
  │      state.val_data_stream = val
  │
  ├─[2] extract_features(state)  → pass-through
  │
  ├─[3] build_model(state)
  │      state.model = FakeModel()      ← test double in Phase 1
  │      state.loss_fn = FakeMSELoss()
  │      state.optimizer = FakeOptimizer()
  │      state.metrics = Metrics(accuracy=accuracy, f1=f1_score)
  │
  ├─[4] train(state)
  │      TrainLoop(...).run(state)
  │        epoch loop → on_epoch_start/end, on_batch_end → hooks
  │        state.history = {"loss": [...]}
  │
  ├─[5] evaluate(state)
  │      model.eval_mode()
  │      for batch in state.val_data_stream:
  │        collect predictions + targets
  │      state.metrics.compute(y_true, y_pred)
  │
  └─[6] export(state)
         to_csv(state.predictions, "outputs/predictions.csv")
```

### Infer mode

```
run("infer"):
  load_data → export
  (skips extract_features, build_model, train, evaluate)
```

---

## 5. Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│  run.py (CLI)                                                │
│  python run.py --config config.yaml --mode train|infer       │
└──────────────────────────┬───────────────────────────────────┘
                           │ Config.from_yaml
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  BasePipeline                                                │
│                                                              │
│  run("train"):          @property hooks (read-only)          │
│    load_data            ┌──────────────────────┐             │
│    extract_features     │ on_stage_start       │             │
│    build_model          │ on_stage_end         │             │
│    train ───────────────┤   ↓                                │
│    evaluate             │ TrainLoop     ←───── hooks 借用    │
│    export               │   on_epoch_start                  │
│                         │   on_epoch_end                    │
│  reads/writes           │   on_batch_end                    │
│  ───────────────────►   └──────────────────────┘             │
│  PipelineState (state machine)                               │
│    data_stream, val_data_stream, model, loss_fn, optimizer,  │
│    history, metrics (Metrics), predictions,                  │
│    current_epoch, should_stop, mode, config                  │
└──────────────────────────────────────────────────────────────┘

Phase 1 new (inside BasePipeline or used by it):
  CsvDataSource ──► train_test_split() ──► DataStream ×2
  TrainLoop ──► reads PipelineState, dispatches to hooks
  Metrics ──► holds metric functions + results
  ProgressHook ──► tqdm + batch loss
  to_csv ──► writes predictions
```

---

## 6. Error Handling

| Scenario | Behavior |
|----------|----------|
| CSV file missing | `FileNotFoundError` from `pandas.read_csv`, propagates |
| Empty CSV | `__len__` returns 1 (empty batch); TrainLoop does 0 iterations |
| Hook raises during train | Caught by `TrainLoop._notify()`, logged, training continues |
| Stage raises | Propagates (stages are critical path) |
| `num_epochs = 0` | TrainLoop.run() exits immediately, empty history |
| `model.parameters()` empty | backward/step skipped, only forward+loss executed |
| `val_data_stream` is None | evaluate() raises informative error |

---

## 7. Test Strategy (TDD)

### Phase 0 amendment tests

Tests for `Loss`, updated `LossProtocol`, `PipelineState.val_data_stream`, `Metrics` type
on `PipelineState`, `BasePipeline.hooks` @property, and `BasePipeline.train()` default
implementation.

### Phase 1 new tests

| Module | Test file | Key cases |
|--------|-----------|-----------|
| `CsvDataSource` | `tests/unit/test_csv_source.py` | reads CSV, yields Batches, batch_size boundary, empty CSV, shuffle reproducibility, lazy load |
| `train_test_split` | `tests/unit/test_split.py` | correct ratio, shuffle reproducibility, both streams iterable, edge: ratio=0.99 |
| `TrainLoop` | `tests/unit/test_train_loop.py` | runs correct num epochs, writes history, skips backward when params empty, respects should_stop, hook dispatch, hook exception recovery |
| `Metrics` | `tests/unit/test_metrics.py` | compute returns correct values, __getitem__, empty metrics, all 5 pure functions against known numpy inputs |
| `ProgressHook` | `tests/unit/test_progress.py` | creates pbar on epoch_start, updates on batch_end, closes on epoch_end |
| `to_csv` | `tests/unit/test_to_csv.py` | writes file, round-trip read, columns header |
| `__main__` | `tests/unit/test_main.py` | prints version + components |

### Test doubles

```python
class FakeModel(ModelProtocol):
    """Returns configurable predictions, has empty parameters()."""
    def __init__(self, output=None):
        self._output = output
    def forward(self, inputs): return self._output or inputs
    def parameters(self): return []
    def train_mode(self): pass
    def eval_mode(self): pass

class FakeModelWithParams(ModelProtocol):
    """Same but returns non-empty parameters() for backward-path coverage."""
    def parameters(self): return [Parameter(data=np.array([1.0]), name="w")]

class FakeLoss(LossProtocol):
    def forward(self, preds, targets): return Loss(float(np.mean((preds - targets)**2)))

class FakeOptimizer(OptimizerProtocol):
    def step(self): pass
    def zero_grad(self): pass
```

### Test data

- `tests/fixtures/tiny_titanic.csv` — 10 rows, 3 features + 1 target, committed to git
- `data/titanic/` — full dataset, gitignored, download instructions in README

---

## 8. Phase 1 Completion Gate

- [ ] `python -m pipeline` prints version + available components + device info
- [ ] `python run.py --config config.yaml --mode train` runs full pipeline
      (CSV → TrainTestSplit → TrainLoop → Metrics → to_csv) using fake models
- [ ] `python run.py --config config.yaml --mode infer` runs inference-only path
- [ ] `python run.py --help` documents all CLI options
- [ ] All new code covered by TDD tests (Phase 0 amendment tests + Phase 1 new tests)
- [ ] `uv run ruff check .` clean
- [ ] `uv run mypy pipeline/` clean (numpy stub warnings on py3.13 expected)

---

## 9. Future Extensions (not in Phase 1)

- PipelineState: fine-grained stage state (idle/loading/training/evaluating/done),
  state transition validation, `on_state_transition` hook point
- Metrics: streaming accumulation via `MetricProtocol.update()/compute()`
- TrainLoop: gradient accumulation steps, mixed precision support
- Data: streaming train_test_split for large datasets
