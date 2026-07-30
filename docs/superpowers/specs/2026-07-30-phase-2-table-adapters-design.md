# Phase 2: Table Adapters Design

Date: 2026-07-30
Status: approved, plan written → execution

## 1. Overview

Phase 2 builds the framework adapters and training components that support AI syllabus
milestones M1 (sklearn), M2 (numpy math), and M3 (NN from scratch). The **Titanic dataset**
is the common target across all three milestones — the same contract test runs on both
SklearnModel and NumpyModel, verifying the protocol design is correct.

### Milestone mapping

```
AI Syllabus                     Phase 2 deliverable
─────────────────────────────────────────────────────
M1 (sklearn)                    SklearnModel adapter + StubLoss/StubOptimizer
                                → examples/m1-sklearn/titanic.ipynb

M2 (numpy math)                 NumpyModel (single-neuron), SGD/Adam, MSELoss
                                sympy→numpy gradient descent (notebook only)
                                ↘
M3 (NN from scratch)            NumpyModel (multi-layer), CrossEntropyLoss
                                → examples/m2-m3-numpy/gradient_to_mlp.ipynb
```

M2 and M3 are one continuous notebook arc: single-neuron → sympy gradient → lambdify →
manual GD → multi-layer → automated backprop.

### Design principles (from architecture spec)

- **No framework in core.** `pipeline/` never imports sklearn or sympy at module level.
- **Protocols are the boundary.** Every component goes through ModelProtocol, LossProtocol, OptimizerProtocol.
- **Sklearn passes through the same pipeline stages** with stub Loss/Optimizer, but
  overrides `train()` — teaching the difference between `.fit()` and gradient descent
  is the point.
- **M2→M3 is a ladder.** Single neuron math first, then multi-neuron connection.

---

## 2. Module Layout

```
pipeline/
├── adapters/                      # NEW package
│   ├── __init__.py                # exports SklearnModel, NumpyModel, NumpyOptimizer
│   ├── sklearn_adapter.py         # SklearnModel(ModelProtocol) + StubLoss + StubOptimizer
│   └── numpy_adapter.py           # NumpyModel(ModelProtocol) + NumpyOptimizer(OptimizerProtocol)
├── training/
│   ├── train_loop.py              # (unchanged)
│   ├── optimizers.py              # NEW — SGD, Adam (pure numpy)
│   └── losses.py                  # NEW — MSELoss, CrossEntropyLoss (pure numpy, LossProtocol)
└── pipeline.py                    # (unchanged — train() default already supports override)

examples/
├── m1-sklearn/
│   └── titanic.ipynb              # LogisticRegression → Kaggle submission
└── m2-m3-numpy/
    └── gradient_to_mlp.ipynb      # sympy single-neuron GD → numpy multi-layer MLP

tests/
├── contract/                      # NEW
│   └── test_model_contract.py     # parametrized: SklearnModel + NumpyModel on Titanic
├── unit/
│   ├── test_sklearn_adapter.py    # NEW
│   ├── test_numpy_adapter.py      # NEW
│   ├── test_optimizers.py         # NEW
│   └── test_losses.py             # NEW
└── fixtures/
    └── tiny_titanic.csv           # (unchanged, 10 rows)
```

### Lazy imports

`pipeline/adapters/__init__.py` lazy-imports sklearn and sympy. The core `pipeline/`
package imports without either installed.

```python
# pipeline/adapters/__init__.py
def get_sklearn_model(estimator):
    from pipeline.adapters.sklearn_adapter import SklearnModel
    return SklearnModel(estimator)

def get_numpy_model(layers):
    from pipeline.adapters.numpy_adapter import NumpyModel
    return NumpyModel(layers)
```

---

## 3. SklearnModel Adapter

### Design

`SklearnModel` wraps any sklearn estimator. It is **passive** — it satisfies the
protocol but does not call `.fit()` itself. The pipeline subclass calls `.fit()` in
an overridden `train()`.

```python
# pipeline/adapters/sklearn_adapter.py

class SklearnModel(ModelProtocol):
    """Wraps any sklearn estimator as a ModelProtocol.

    forward() delegates to predict_proba() for classifiers (returning
    class probabilities) or predict() for regressors.
    parameters() returns [] — sklearn models don't expose gradient params.
    train_mode()/eval_mode() are no-ops.
    """

    def __init__(self, estimator):
        self._estimator = estimator

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(inputs)
        return self._estimator.predict(inputs)

    def parameters(self) -> Iterable[Parameter]:
        return []

    def train_mode(self) -> None:
        pass

    def eval_mode(self) -> None:
        pass
```

### Stub Loss/Optimizer

Two no-op classes satisfy the protocol so `build_model()` can always assign
`state.loss_fn` and `state.optimizer`:

```python
class StubLoss(LossProtocol):
    """No-op loss. Always returns 0.0."""
    def forward(self, predictions, targets) -> Loss:
        return Loss(value=0.0)

class StubOptimizer(OptimizerProtocol):
    """No-op optimizer."""
    def step(self) -> None: pass
    def zero_grad(self) -> None: pass
```

### Pipeline usage

```python
class TitanicSklearnPipeline(BasePipeline):
    def load_data(self, state):
        source = CsvDataSource("data/titanic/train.csv", batch_size=32)
        train, val = train_test_split(source, train_ratio=0.8)
        state.data_stream = train
        state.val_data_stream = val

    def build_model(self, state):
        from sklearn.linear_model import LogisticRegression
        state.model = SklearnModel(LogisticRegression(max_iter=1000))
        state.loss_fn = StubLoss()
        state.optimizer = StubOptimizer()

    def train(self, state):
        """Override: sklearn fits in one call, not per-batch."""
        from pipeline.data.utils import collect_arrays
        X, y = collect_arrays(state.data_stream)
        state.model._estimator.fit(X, y)
        state.history = {"loss": [0.0]}  # sklearn doesn't expose per-epoch loss

    def evaluate(self, state): ...
    def export(self, state): ...
```

`collect_arrays()` is a utility that drains a DataStream into (X, y) numpy arrays.
Lives in `pipeline/data/` since it's useful beyond sklearn.

**Why override `train()` instead of making TrainLoop branch?**
The difference between "sklearn fits in one call" and "numpy does per-batch gradient
descent" is the pedagogical point of the spiral curriculum. Hiding it behind a uniform
interface or an empty loop misses the lesson.

---

## 4. NumpyModel + NumpyOptimizer

### NumpyModel

Backed by numpy arrays with explicit `Parameter` objects. Supports both single-layer
(M2) and multi-layer (M3) by passing different layer configs.

```python
# pipeline/adapters/numpy_adapter.py

class NumpyModel(ModelProtocol):
    """Model backed by numpy arrays with explicit Parameter objects.

    Supports configurable layer stack. parameters() returns live Parameter
    references — the optimizer mutates them in-place.
    """

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]], activation: str = "relu"):
        """Build a model from (weight, bias) pairs.

        layers[0] is (W1, b1) for input→hidden, layers[1] is (W2, b2) for
        hidden→output, etc. Each pair becomes two Parameter objects.
        """
        self._params: list[Parameter] = []
        for i, (W, b) in enumerate(layers):
            self._params.append(Parameter(data=W, grad=np.zeros_like(W), name=f"W{i}"))
            self._params.append(Parameter(data=b, grad=np.zeros_like(b), name=f"b{i}"))
        self._activation = activation
        self._training = True

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        x = np.asarray(inputs, dtype=np.float64)
        # Iterate (W, b) pairs: z = x @ W.T + b; x = activation(z)
        # Last layer: no activation (raw logits)
        n_layers = len(self._params) // 2
        for i in range(n_layers):
            W = self._params[2 * i].data
            b = self._params[2 * i + 1].data
            x = x @ W.T + b
            if i < n_layers - 1:
                x = self._apply_activation(x)
        return x

    def parameters(self) -> Iterable[Parameter]:
        yield from self._params

    def train_mode(self) -> None:
        self._training = True

    def eval_mode(self) -> None:
        self._training = False
```

### NumpyOptimizer

Takes a list of `Parameter` refs and an update rule. `step()` reads `param.grad`,
updates `param.data` in-place, then calls the rule.

```python
class NumpyOptimizer(OptimizerProtocol):
    """Optimizer that mutates Parameter objects in-place.

    Delegates the actual update formula to an OptimizerRule (SGD, Adam).
    """

    def __init__(self, parameters: Iterable[Parameter], rule):
        self._params = list(parameters)
        self._rule = rule

    def step(self) -> None:
        for p in self._params:
            self._rule.update(p)

    def zero_grad(self) -> None:
        for p in self._params:
            if p.grad is not None:
                p.grad.fill(0.0)
```

---

## 5. Optimizers and Losses

### Optimizer rules (pure numpy)

```python
# pipeline/training/optimizers.py

class SGD:
    """Vanilla SGD: w = w - lr * grad."""
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, param: Parameter) -> None:
        param.data -= self.lr * param.grad

class Adam:
    """Adam: m = b1*m + (1-b1)*grad; v = b2*v + (1-b2)*grad^2;
       w = w - lr * m_hat / (sqrt(v_hat) + eps)."""
    def __init__(self, lr: float = 0.001, betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}
        self._t = 0

    def update(self, param: Parameter) -> None:
        self._t += 1
        key = id(param)
        if key not in self._m:
            self._m[key] = np.zeros_like(param.data)
            self._v[key] = np.zeros_like(param.data)
        self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * param.grad
        self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * param.grad ** 2
        m_hat = self._m[key] / (1 - self.beta1 ** self._t)
        v_hat = self._v[key] / (1 - self.beta2 ** self._t)
        param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

### Loss functions (pure numpy, return Loss objects)

```python
# pipeline/training/losses.py

class MSELoss(LossProtocol):
    """Mean squared error: mean((pred - target)^2)."""
    def forward(self, predictions, targets) -> Loss:
        pred = np.asarray(predictions)
        targ = np.asarray(targets)
        diff = pred - targ
        value = float(np.mean(diff ** 2))
        n = diff.size
        def _backward():
            ...  # dL/dpred = 2 * diff / n
        return Loss(value=value, _backward_fn=_backward)

class CrossEntropyLoss(LossProtocol):
    """Softmax + negative log-likelihood."""
    def forward(self, predictions, targets) -> Loss:
        ...  # softmax + NLL
        def _backward():
            ...  # dL/dlogits = softmax_probs - one_hot_targets
        return Loss(value=value, _backward_fn=_backward)
```

The `_backward_fn` closure captures forward-pass intermediates. This is the manual
backprop that M2/M3 students inspect and eventually trust.

---

## 6. M2→M3 Notebook Arc

One notebook (`examples/m2-m3-numpy/gradient_to_mlp.ipynb`), two acts:

### Act 1 (M2): "Where do gradients come from?"

1. Define a single neuron in sympy: `y = σ(w·x + b)`, `L = (y - t)²`
2. Call `sympy.diff(L, w)`, `sympy.diff(L, b)` → see gradient formulas
3. `lambdify` into numpy functions
4. Manual training loop: forward → compute grad via lambdified function →
   `param.grad = grad_value` → `SGD.update(param)` → repeat
5. Plot loss curve — student sees gradient descent from first principles

### Act 2 (M3): "Now connect them"

6. "We could derive these gradients by hand for a 2-layer network... or let backprop do it"
7. Stack two layers: `h = relu(W1·x + b1)`, `y = softmax(W2·h + b2)`
8. Use `NumpyModel([(W1, b1), (W2, b2)])` + `CrossEntropyLoss._backward_fn` →
   the student has earned the right to use automated gradients
9. `TrainLoop` → training curve → Kaggle submission CSV
10. Compare: sklearn accuracy vs numpy MLP accuracy on the same Titanic test split

**The sympy→lambdify step lives only in the notebook.** It doesn't need a pipeline
module. The pipeline provides `NumpyModel`, `SGD`, `Parameter`; the notebook shows *why*
they work.

---

## 7. Training Path Design

Two training paths coexist, distinguished by whether the model has gradient parameters:

### Path 1: sklearn (non-gradient)

```
train() override → collect all data → _estimator.fit(X, y)
```
- No TrainLoop involved
- Stub Loss/Optimizer satisfy build_model() assignments
- `collect_arrays(data_stream)` utility drains a DataStream into (X, y)

### Path 2: numpy (gradient-based)

```
train() default → TrainLoop → forward → loss → backward → step
```
- Uses BasePipeline's default `train()` — no override needed
- Real Loss (MSELoss, CrossEntropyLoss) with `_backward_fn`
- Real Optimizer (SGD, Adam) mutating Parameter refs in-place

### Utility: `_collect_arrays`

```python
# pipeline/data/utils.py  (or in data/__init__.py)

def collect_arrays(stream: DataStream) -> tuple[np.ndarray, np.ndarray]:
    """Drain a DataStream into (X, y) numpy arrays.
    
    Useful for sklearn models that need the full dataset at once,
    and for evaluation loops that aggregate predictions.
    """
    xs, ys = [], []
    for batch in stream:
        xs.append(np.asarray(batch.inputs))
        ys.append(np.asarray(batch.targets))
    return np.concatenate(xs), np.concatenate(ys)
```

---

## 8. Contract Tests

One test file parametrized over both adapters. Verifies protocol compliance,
not benchmark scores.

```python
# tests/contract/test_model_contract.py

@pytest.mark.parametrize("adapter", ["sklearn", "numpy"])
class TestModelContract:
    """Every ModelProtocol implementation must pass these."""

    def test_forward_shape(self, adapter, batch):
        """forward(inputs).shape[0] == batch_size."""
        ...

    def test_train_eval_mode_cycle(self, adapter):
        """train_mode() → eval_mode() → train_mode() without error."""
        ...

    def test_full_pipeline_on_titanic(self, adapter, tiny_titanic_path):
        """Same Titanic task, same evaluation metric.
        Both adapters produce valid predictions through the same
        pipeline stages."""
        ...
```

### Test dimensions (from architecture spec)

| Dimension | SklearnModel | NumpyModel |
|-----------|-------------|------------|
| Happy Path | LogisticRegression on Titanic | 2-layer MLP on Titanic |
| Boundary | Single sample, single feature | Single neuron, no hidden layer |
| Empty/Null | Empty parameter list | Zero-gradient edge case |
| Type Error | Non-array input to forward | Wrong dtype grad |
| Shape Mismatch | Wrong feature count | Layer dimension mismatch |
| Reproducibility | Same seed → same result | Same seed → same result |

---

## 9. Files Created/Modified Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `pipeline/adapters/__init__.py` | NEW | ~15 |
| `pipeline/adapters/sklearn_adapter.py` | NEW | ~60 |
| `pipeline/adapters/numpy_adapter.py` | NEW | ~80 |
| `pipeline/training/optimizers.py` | NEW | ~80 |
| `pipeline/training/losses.py` | NEW | ~80 |
| `pipeline/data/utils.py` | NEW | ~20 (collect_arrays) |
| `examples/m1-sklearn/titanic.ipynb` | NEW | notebook |
| `examples/m2-m3-numpy/gradient_to_mlp.ipynb` | NEW | notebook |
| `tests/contract/__init__.py` | NEW | empty |
| `tests/contract/test_model_contract.py` | NEW | ~80 |
| `tests/unit/test_sklearn_adapter.py` | NEW | ~80 |
| `tests/unit/test_numpy_adapter.py` | NEW | ~80 |
| `tests/unit/test_optimizers.py` | NEW | ~80 |
| `tests/unit/test_losses.py` | NEW | ~80 |
| `pipeline/__init__.py` | MODIFIED | +adapter exports |

### Out of scope (Phase 3+)

- `CheckpointHook`, `EarlyStoppingHook` — Phase 3
- `TorchAdapter`, `TorchDataStream` — Phase 3
- `ImageFolderDataSource` — Phase 3
- `save_checkpoint`, `load_checkpoint` — Phase 4
- HPO, ensemble — Phase 5

---

## 10. Design Decisions

| Decision | Rationale |
|----------|-----------|
| `train()` override for sklearn, not TrainLoop branching | Teaches the real difference between `.fit()` and gradient descent |
| Stub Loss/Optimizer instead of optional fields | build_model() always assigns all three; no None-checking in later stages |
| `_collect_arrays` as a utility, not a DataStream method | DataStream is an iterator protocol; draining it is a consumer concern |
| Single `NumpyModel` class, not separate SingleNeuron/MLP | The architecture is the same (layers → forward); the difference is in the notebook story |
| `OptimizerRule` separate from `OptimizerProtocol` | Protocol handles the pipeline contract (step/zero_grad); rules are pure math |
| `_backward_fn` closure captures intermediates | Explicit — students can inspect what the gradient depends on. No autograd magic |
| sympy code lives in the notebook, not in `pipeline/` | The pipeline provides components; the notebook teaches why they work |
