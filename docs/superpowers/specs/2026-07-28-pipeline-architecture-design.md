# Universal_DL_Pipeline Architecture Design

Date: 2026-07-28
Status: approved, pending implementation plan

## 1. Overview

Universal_DL_Pipeline is a **teaching-first, framework-agnostic deep learning pipeline library**.
It serves two roles:

1. **Infrastructure**: a Python package (`pipeline/`) providing abstract pipeline
   interfaces and composable components.
2. **Code companion**: an `examples/` directory aligned with the AI syllabus ("From Prompt
   to Transistor and Back", `materials/ml-materials/AI/`), where each example demonstrates
   one module using this pipeline.

### Design Principles

- **Teach, don't compete.** Code is readable first, functional second. Reference mature
  libraries; don't rebuild them.
- **Framework-agnostic core.** The pipeline defines *when* things happen. Concrete
  implementations define *how*. Protocols (not PyTorch classes) are the contract.
- **Pipeline is not a black box.** Every stage can be inspected, replaced, or skipped.
  Students descend the abstraction ladder one rung at a time.
- **Spiral pedagogy.** Same pipeline structure, progressively richer implementations:
  sklearn → numpy → PyTorch → custom.

### Relationship to AI Syllabus

```
AI Syllabus                     Universal_DL_Pipeline
─────────────────────────────────────────────────────
M0 (LLM hook)                   (no pipeline dependency)
M1 (sklearn)                    Phase 2: SklearnModel adapter + Titanic example
M2 (numpy math)                 Phase 2: sympy→numpy gradient descent
M3 (NN from scratch)            Phase 2: numpy MLP in Titanic
M4 (PyTorch + DL)               Phase 3: TorchAdapter + MNIST/CIFAR-10
M5 (sequences)                  Phase 6: NLP adapter
M6 (min LLM)                    examples/ only; may add pipeline modules, no API changes
M7 (LLM landscape)              examples/ only; may add pipeline modules, no API changes
```

---

## 2. Coding Standards

### Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Files | `snake_case` | `train_loop.py` |
| Classes | `PascalCase` | `BasePipeline`, `CsvDataSource` |
| Abstract bases | `Base` prefix or `Protocol` suffix | `BaseHook`, `ModelProtocol` |
| Functions/methods | `snake_case`, verb-first | `build_dataloader`, `compute_metrics` |
| Private members | single underscore `_` | `_notify()`, `_hooks` |

### Documentation

- **All public API**: Google-style docstrings (parsed by mkdocstrings).
- **Module-level docstring**: position in pipeline + minimal usage example.
- **Teaching comments**: `# WHY:` explains design decisions; `# NOTE:` marks common pitfalls.
- **All comments in English**. Chinese explanations live in `examples/` notebooks.

### Complexity Limits

| Metric | Limit | Rationale |
|--------|-------|-----------|
| Lines per file | ≤ 300 | One teaching unit per file |
| Lines per function | ≤ 50 | Fits on screen |
| Public methods per class | ≤ 5 | Student mental budget |
| Nesting depth | ≤ 3 | Readability |
| Cyclomatic complexity | ≤ 10 | One clear path |

Forbidden patterns:
- Closures capturing mutable state (see dl_framework `hpo.py` lesson)
- `if task_type == "..."` branching (SOLID violation)
- Module-level global mutable state

### Tooling

- **Environment**: `uv` (lockfile, editable install, `link-mode = "copy"` for WSL2)
- **Build**: `hatchling`
- **Lint/Format**: `ruff` (replaces flake8 + isort + black)
- **Type check**: `mypy` (strict mode)
- **Docs**: Material for MkDocs + mkdocstrings + `mkdocs-static-i18n` (en/zh toggle)
- **CI**: GitHub Actions, `uv run pytest` (unit < 60s, slow on main only)

### Project Structure

```
Universal_DL_Pipeline/
├── pyproject.toml
├── mkdocs.yml
├── CLAUDE.md
├── .gitignore
├── pipeline/                   # framework core
│   ├── __init__.py
│   ├── config.py
│   ├── protocols.py            # abstract interfaces
│   ├── pipeline.py             # BasePipeline ABC + PipelineState
│   ├── registry.py             # DI: register() + build()
│   ├── hooks.py                # BaseHook ABC + built-in hooks
│   ├── data/                   # data sources, splits
│   ├── training/               # TrainLoop, optimizers, losses
│   ├── evaluation/             # metrics
│   ├── export/                 # CSV, checkpoint
│   ├── inference/              # Inferencer, TTA
│   ├── ensemble/               # Stacker
│   ├── adapters/               # sklearn, numpy, torch adapters
│   ├── hpo.py                  # outer-loop HPO script
│   └── utils/                  # device, seed
├── examples/                   # AI syllabus code companion
│   ├── m1-sklearn/
│   ├── m2-numpy-math/
│   ├── m3-nn-scratch/
│   ├── m4-pytorch/
│   ├── m5-sequences/
│   ├── m6-min-llm/
│   └── m7-landscape/
├── tests/
│   ├── contract/               # protocol contract tests
│   ├── unit/                   # per-module unit tests
│   ├── integration/            # cross-module integration tests
│   └── fixtures/               # tiny committed datasets
├── data/                       # test datasets (gitignored)
│   └── */README.md             # download instructions
└── docs/                       # MkDocs source
```

---

## 3. Core Architecture

### 3.1 Abstract Protocols (Phase 0)

The pipeline depends on protocols, not frameworks. Any framework (numpy, PyTorch, JAX)
can satisfy these interfaces via an adapter.

```python
from dataclasses import dataclass, field
from typing import Protocol, Iterator, Iterable, Literal, Any
import numpy as np

ArrayLike = np.ndarray | Any

@dataclass
class Parameter:
    """A trainable parameter. Wraps framework-specific tensor."""
    data: ArrayLike
    grad: ArrayLike | None = None
    name: str = ""

@dataclass
class Batch:
    """One batch of data. Framework-agnostic."""
    inputs: ArrayLike
    targets: ArrayLike

class DataStream(ABC):
    """Iterable producing Batch objects."""
    @abstractmethod
    def __iter__(self) -> Iterator[Batch]: ...
    @abstractmethod
    def __len__(self) -> int: ...

class ModelProtocol(ABC):
    """A model: forward pass + trainable parameters."""
    @abstractmethod
    def forward(self, inputs: ArrayLike) -> ArrayLike: ...
    @abstractmethod
    def parameters(self) -> Iterable[Parameter]: ...
    @abstractmethod
    def train_mode(self) -> None: ...
    @abstractmethod
    def eval_mode(self) -> None: ...

class LossProtocol(ABC):
    """Loss function. Callable returns scalar float."""
    @abstractmethod
    def forward(self, predictions: ArrayLike, targets: ArrayLike) -> float: ...
    def __call__(self, predictions: ArrayLike, targets: ArrayLike) -> float:
        return self.forward(predictions, targets)

class OptimizerProtocol(ABC):
    """Updates model parameters using accumulated gradients."""
    @abstractmethod
    def step(self) -> None: ...
    @abstractmethod
    def zero_grad(self) -> None: ...
```

### 3.2 BasePipeline (Phase 0)

The pipeline is the template that orchestrates stages. Subclasses provide concrete
implementations via dependency injection.

```python
@dataclass
class PipelineState:
    """Shared state bus. Each stage reads/writes its fields."""
    config: "Config"
    mode: Literal["train", "infer"] = "train"
    data_stream: DataStream | None = None
    features: ArrayLike | None = None
    model: ModelProtocol | None = None
    loss_fn: LossProtocol | None = None
    optimizer: OptimizerProtocol | None = None
    history: dict | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: ArrayLike | None = None
    current_epoch: int = 0
    should_stop: bool = False

class BasePipeline(ABC):
    """Universal DL pipeline template.
    
    Defines *when* stages run. Subclasses define *how* via DI.
    Hooks inject cross-cutting concerns (logging, checkpoint, early-stop).
    
    Usage:
        pipeline = SomePipeline(config)
        pipeline.add_hook(ProgressHook())
        pipeline.add_hook(EarlyStopHook(patience=5))
        result = pipeline.run("train")
    """
    
    def __init__(self, config: "Config"):
        self.config = config
        self._hooks: list[BaseHook] = []
    
    def add_hook(self, hook: "BaseHook") -> None:
        self._hooks.append(hook)
    
    def run(self, mode: Literal["train", "infer"] = "train") -> PipelineState:
        state = PipelineState(config=self.config, mode=mode)
        
        self._run_stage("load_data", state, self.load_data)
        
        if mode == "train":
            self._run_stage("extract_features", state, self.extract_features)
            self._run_stage("build_model", state, self.build_model)
            self._run_stage("train", state, self.train)
            self._run_stage("evaluate", state, self.evaluate)
        
        self._run_stage("export", state, self.export)
        return state
    
    # ── Abstract stages (subclass or DI) ────────
    @abstractmethod
    def load_data(self, state: PipelineState) -> None: ...
    
    def extract_features(self, state: PipelineState) -> None:
        """Default: pass-through. DL models extract features internally."""
        pass
    
    @abstractmethod
    def build_model(self, state: PipelineState) -> None: ...
    @abstractmethod
    def train(self, state: PipelineState) -> None: ...
    @abstractmethod
    def evaluate(self, state: PipelineState) -> None: ...
    @abstractmethod
    def export(self, state: PipelineState) -> None: ...
    
    # ── Hook dispatch ───────────────────────────
    def _run_stage(self, name: str, state: PipelineState, stage_fn):
        self._notify("on_stage_start", name, state)
        stage_fn(state)
        self._notify("on_stage_end", name, state)
    
    def _notify(self, event: str, *args) -> None:
        for hook in self._hooks:
            getattr(hook, event)(*args)
```

### 3.3 Hook System (Phase 0/1)

Cross-cutting concerns are hooks, not pipeline stages. Students add them one at a time.

```python
class BaseHook(ABC):
    """Hook into pipeline lifecycle events. Five extension points."""
    def on_stage_start(self, stage: str, state: PipelineState) -> None: ...
    def on_stage_end(self, stage: str, state: PipelineState) -> None: ...
    def on_epoch_start(self, epoch: int, state: PipelineState) -> None: ...
    def on_epoch_end(self, epoch: int, state: PipelineState) -> None: ...
    def on_batch_end(self, batch: int, loss: float, state: PipelineState) -> None: ...
```

Built-in hooks ship progressively:
- Phase 1: `ProgressHook` (tqdm + batch loss)
- Phase 3: `CheckpointHook`, `EarlyStoppingHook`, `VisualizerHook` (loss/acc curves)

### 3.4 DI Registry (Phase 0)

Minimal registry: one dict, two functions. No framework magic.

```python
_REGISTRY: dict[str, dict[str, type]] = {}

def register(kind: str, name: str):
    """Decorator: register a class under (kind, name)."""
    def decorator(cls):
        _REGISTRY.setdefault(kind, {})[name] = cls
        return cls
    return decorator

def build(kind: str, name: str, **kwargs):
    """Build a registered component by (kind, name)."""
    return _REGISTRY[kind][name](**kwargs)
```

### 3.5 Design Decisions

| Decision | Rationale |
|----------|-----------|
| `extract_features` defaults to pass-through | DL fuses feature extraction into the model. The slot exists so M1 (sklearn) examples can demonstrate explicit feature engineering. |
| Hooks, not callbacks | Simpler concept. Five events cover all cross-cutting needs. |
| `PipelineState` as data bus | Avoids parameter explosion. Each stage writes, next stage reads. Students can inspect every intermediate value. |
| Registry as a flat dict | MMEngine's hierarchical registries are correct but opaque. A dict teaches the concept first. |
| HPO is an outer loop | HPO = "run the pipeline N times with different configs." Not a pipeline stage. A separate script wraps `pipeline.run()`. |
| Federated learning is deferred | FL changes the training topology, not the pipeline structure. Future: a `DistributedTrainLoop` that replaces the default one. |

---

## 4. Implementation Phases

### Phase 0: Skeleton (zero dependencies, zero ML frameworks)

**Written from scratch. No reference to dl_framework.**

```
pipeline/
├── __init__.py
├── config.py           # Config dataclass (~20 fields)
├── protocols.py        # ModelProtocol, DataStream, Batch,
│                       #   LossProtocol, OptimizerProtocol, Parameter
├── pipeline.py         # BasePipeline ABC + PipelineState
├── registry.py         # register() + build()
├── hooks.py            # BaseHook ABC
└── utils/
    ├── device.py       # get_device(), device_info()
    └── seed.py         # set_seed()
```

No imports of torch, sklearn, or sympy. Only `abc`, `dataclasses`, `typing`, `numpy` (for `ArrayLike` type alias).

### Phase 1: Pipeline Framework (operates on protocols only)

```
pipeline/data/
    - CsvDataSource: CSV → Batch iterator
    - TrainTestSplit: (train_stream, val_stream)
    - Shuffle, Batch wrapper utilities

pipeline/training/
    - TrainLoop: forward → loss → backward → step, <80 lines
    - Skips backward/step when model.parameters() is empty

pipeline/evaluation/
    - metrics.py: accuracy, precision, recall, F1, confusion_matrix
    - Pure numpy, ArrayLike input

pipeline/hooks/
    - ProgressHook: tqdm + batch loss

pipeline/export/
    - to_csv.py (basic)
```

Test data introduced: `data/titanic/` (train.csv only, gitignored).

Tests use **fake** ModelProtocol implementations (return fixed predictions) to validate
TrainLoop iteration count, metrics correctness, hook dispatch order.

### Phase 2: Table Adapters (M1/M2/M3)

```
pipeline/adapters/sklearn_adapter.py
    - SklearnModel(ModelProtocol): forward→predict_proba, parameters→[]

pipeline/adapters/numpy_adapter.py
    - NumpyModel(ModelProtocol): Parameter list + np.dot forward
    - NumpyOptimizer(OptimizerProtocol): step() + zero_grad()

pipeline/training/optimizers.py
    - SGD, Adam (pure numpy, OptimizerProtocol)

pipeline/training/losses.py
    - MSELoss, CrossEntropyLoss (pure numpy, LossProtocol)

examples/
    examples/m1-sklearn/titanic.py         # sklearn LogisticRegression → submission
    examples/m2-numpy-math/gd_demo.py      # sympy gradient → lambdify → numpy GD
    examples/m3-nn-scratch/mlp_titanic.py  # manual MLP on Titanic
```

Key test: the same Titanic contract test runs on both SklearnModel and NumpyModel,
verifying the protocol design is correct.

### Phase 3: Image + TorchAdapter (M4)

```
pipeline/adapters/torch_adapter.py
    - TorchModel(ModelProtocol): nn.Module → ModelProtocol
    - TorchDataStream(DataStream): DataLoader → DataStream
    - TorchOptimizer(OptimizerProtocol): torch.optim → OptimizerProtocol

pipeline/data/image.py
    - ImageFolderDataSource

pipeline/models/
    - build_backbone(), SimpleCNN, ResNet18 (3 models to start)
    - Registry-backed: @register("backbone", "resnet18")

pipeline/hooks/
    - CheckpointHook, EarlyStoppingHook, VisualizerHook

examples/m4-pytorch/
    - mnist_cnn.py
    - cifar10_resnet.py
```

Test data: `data/digit-recognizer/`, `data/cifar-10/`.

### Phase 4: Export + Inference (production-grade)

```
pipeline/export/
    - save_checkpoint, load_checkpoint (framework-agnostic metadata)
    - to_submission_csv

pipeline/inference/
    - Inferencer: predict_logits, predict_proba, predict
    - TTAInferencer: test-time augmentation
```

dl_framework sources to extract and simplify:
- `export/exporter.py` (9/10 readability)
- `inference/inferencer.py` (8/10 readability)

### Phase 5: HPO + Ensemble (pipeline capability completion)

```
pipeline/hpo.py
    - Standalone Optuna script. Outer loop that calls pipeline.run() N times.
    - Not a pipeline stage. No closures, no config mutation.

pipeline/ensemble/stacker.py
    - collect_oof(), fit_meta(), predict()
    - Two-level stacking: base models → meta-learner
```

dl_framework sources:
- `ensemble/stacker.py` (8/10) — extract and simplify
- `hpo.py` (5/10) — DO NOT extract; rewrite from scratch

### Phase 6: Sequences / NLP (M5)

```
pipeline/data/text.py
pipeline/adapters/nlp_adapter.py

examples/m5-sequences/
    - nlp_disaster.py
    - timeseries_forecast.py
```

Test data: `data/spaceship-titanic/`, `data/nlp-getting-started/`.

### M6/M7: LLM Modules

Principle: **may add pipeline modules, must not modify existing APIs.**

Examples only; framework additions only if genuinely needed by the LLM examples.

---

## 5. dl_framework Extraction Plan

Only extract from dl_framework starting Phase 1 (Phase 0 is pure from-scratch).

| dl_framework source | Score | Extract? | Phase | Treatment |
|---------------------|-------|----------|-------|-----------|
| `utils/device.py` | 10/10 | Reference only | 0 | Trivial utility (38 lines). Write from scratch, same API. |
| `utils/seed.py` | 10/10 | Reference only | 0 | Trivial utility (32 lines). Write from scratch, same API. |
| `data/dataset.py` | 8/10 | Pattern only | 1 | Extract `LabelledDataset` pattern, rewrite |
| `data/dataloader.py` | 7/10 | Pattern only | 1 | Extract 3 split modes, simplify (one class = one split mode) |
| `data/transforms.py` | - | Pattern only | 3 | Extract `get_train_transform` pattern |
| `models/backbones.py` | 9/10 | Extract+simplify | 3 | Registry pattern + 3 models, not 17 |
| `training/trainer.py` | 6/10 | DO NOT extract | 1 | Too complex. Rewrite as <80-line `TrainLoop` |
| `training/early_stopping.py` | - | Extract | 3 | Logic is simple, extract |
| `training/optimizers.py` | - | Pattern only | 2 | Factory pattern, rewrite |
| `training/losses.py` | - | Pattern only | 2 | Factory pattern, rewrite |
| `evaluation/metrics.py` | 7/10 | Extract+simplify | 1 | Confusion matrix → derived metrics |
| `export/exporter.py` | 9/10 | Extract+simplify | 4 | Clean SRP, mostly usable |
| `inference/inferencer.py` | 8/10 | Extract+simplify | 4 | Clean API, simplify TTA params |
| `ensemble/stacker.py` | 8/10 | Extract+simplify | 5 | Good pattern, reduce params |
| `hpo.py` | 5/10 | DO NOT extract | 5 | Closures + config mutation. Rewrite as clean outer loop. |
| `tasks/base.py` | 9/10 | Reference only | - | Adapter pattern reference, not extracted |
| `tasks/tabular_classification.py` | 6/10 | Reference only | 6 | Adapter reference, rewrite for simplicity |
| `config/config.py` | 10/10 | Reference only | 0 | Documentation gold standard, but start with 20 fields |
| `core/component.py` | 7/10 | Reference only | - | Component ABCs reference, superseded by our simpler protocols |
| `core/pipeline.py` | 8/10 | Reference only | 0 | Pipeline stage table reference, rewrite |

---

## 6. Test Strategy (TDD)

**Iron law: no production code without a failing test first.**

### Test Dimensions

Every module's tests must consider (where applicable):

| Dimension | Coverage target |
|-----------|----------------|
| Happy Path | Normal inputs, normal flow |
| Boundary | min/max values, dimensions=1, edge of valid range |
| Empty/Null | Empty arrays, None inputs, zero-length iterables |
| Type Error | Wrong dtype, wrong container type |
| Shape Mismatch | Incompatible dimensions between inputs |
| Overflow/Underflow | NaN, inf, zero-division in metrics |
| Reproducibility | Same seed → same result |
| Stress | Large inputs (10k+ samples), many iterations |
| Concurrency | Repeated runs don't share state |
| Error Recovery | Exceptions in one hook don't kill pipeline |

### Contract Tests

Same test suite runs against all protocol implementations:

```python
@pytest.mark.parametrize("adapter", ["SklearnModel", "NumpyModel", "TorchModel"])
def test_model_forward_shape(adapter, batch):
    """Every ModelProtocol implementation must satisfy this contract."""
    model = make_model(adapter)
    output = model.forward(batch.inputs)
    assert output.shape[0] == len(batch.targets)
```

### Test Data

```
tests/fixtures/
    tiny_titanic.csv       # 10 rows, committed to git (fast unit tests)

data/                      # full datasets, gitignored
    titanic/README.md      # download instructions
    digit-recognizer/README.md
    cifar-10/README.md
    ...
```

### CI

```yaml
uv run pytest                     # unit + contract, <60s
uv run pytest -m "slow"           # integration + E2E, main branch only
```

---

## 7. Documentation (MkDocs)

- **Framework**: Material for MkDocs + mkdocstrings (Google-style)
- **i18n**: `mkdocs-static-i18n` plugin — English source, Simplified Chinese translation
- **Structure**:
  - Tutorials: step-by-step walkthroughs (linked from AI syllabus)
  - How-To Guides: "Use the pipeline for a Kaggle competition"
  - API Reference: auto-generated from docstrings
  - Explanation: architecture decisions, design rationale

---

## 8. Migration from Old Code

The existing `Universal_DL_Pipeline/` flat scripts are superseded. They serve as a
reference for ONNX export (the one feature dl_framework lacks), which will be added
as `pipeline/export/onnx.py` during Phase 4.

The existing code is **not deleted during Phase 0-6**. It is archived or removed only
after the new pipeline achieves feature parity.
