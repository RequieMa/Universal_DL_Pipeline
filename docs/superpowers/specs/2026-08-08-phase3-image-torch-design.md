# Phase 3 Design — Image + TorchAdapter → M4

Date: 2026-08-08 | Status: approved

## Scope

Phase 3 adds image data support and a PyTorch adapter to the Universal DL Pipeline.
Five new source modules + two example notebooks.

## Components

### 1. `pipeline/data/image_folder.py` — ImageFolderDataSource

DataStream implementation that reads images from class-labeled folder trees.

```
root_dir/
├── class_0/       # sorted → label 0
│   ├── img_01.jpg
│   └── img_02.png
├── class_1/       # sorted → label 1
│   └── img_03.jpg
└── ...
```

**Design:**

| Field | Detail |
|-------|--------|
| `__init__` | `root_dir: str\|Path`, `batch_size: int`, `shuffle: bool`, `seed: int` |
| `_load()` | lazy scan subfolders once, collect `(path, label_idx)` pairs; `class_names` = sorted folder names |
| `__iter__` | yield `Batch(inputs=list[PIL.Image], targets=np.ndarray)` — inputs are PIL Image objects, targets are int labels |
| `__len__` | `ceil(n_samples / batch_size)` |
| Imports | `from PIL import Image` inside `__iter__` (lazy) |
| Edge: empty folder | `_load()` raises `FileNotFoundError` if `root_dir` missing; `ValueError` if zero images found |
| Edge: non-image files | `Image.open()` raises `OSError` → wrapped in `_load()` with clear message listing bad paths |

**Pattern matches `CsvDataSource`:** lazy-load, batch_size + shuffle + seed, `__len__` returns batch count.

### 2. `pipeline/data/transforms.py` — TransformedDataStream

Decorator DataStream that applies a user-provided transform to each Batch.

**Design:**

| Field | Detail |
|-------|--------|
| `__init__` | `stream: DataStream`, `transform: Callable[[Batch], Batch]` |
| `__iter__` | `for batch in self._stream: yield self._transform(batch)` |
| `__len__` | delegates to `self._stream.__len__()` |

The transform is fully opaque — it can be `torchvision.transforms.Compose`, a custom PIL→numpy pipeline, or any callable. The pipeline core does not care what happens inside.

### 3. `pipeline/adapters/torch_adapter.py` — TorchModel

Thin `ModelProtocol` wrapper around `torch.nn.Module`.

```python
class TorchModel(ModelProtocol):
    def __init__(self, module: Any): ...
    def forward(self, inputs: ArrayLike) -> ArrayLike:
        # torch.Tensor → pass through (zero-copy)
        # numpy/other → torch.as_tensor (no copy if possible)
        # output: same format as input
    def parameters(self) -> Iterable[Parameter]:
        # live refs — optimizer mutates underlying torch tensor in-place
    def train_mode(self) -> None: ...
    def eval_mode(self) -> None: ...
```

**Input/output format rules:**
- `torch.Tensor` in → `torch.Tensor` out (zero overhead)
- `np.ndarray` in → `np.ndarray` out (as_tensor on input, detach+cpu+numpy on output)
- `list[PIL.Image]` in → that goes through a torchvision transform BEFORE reaching forward(), so forward() always sees tensors or arrays

The TensorFlow path is the hot one: `ImageFolderDataSource` → `TransformedDataStream(torchvision transforms)` → `TorchModel.forward(torch.Tensor)` — zero conversion.

### 4. `pipeline/adapters/torch_adapter.py` — TorchLoss

Generic config-driven wrapper for `torch.nn.*` loss functions.

```python
class TorchLoss(LossProtocol):
    def __init__(self, loss_name: str, **kwargs):
        # import torch.nn; loss_cls = getattr(nn, loss_name); self._loss = loss_cls(**kwargs)
    def forward(self, predictions, targets) -> Loss:
        # torch.as_tensor both → self._loss(p, t) → Loss(value, _backward_fn=result.backward)
```

Config: `loss: {name: CrossEntropyLoss}` or `loss: {name: MSELoss}`.

### 5. `pipeline/adapters/torch_adapter.py` — TorchOptimizer

Generic config-driven wrapper for `torch.optim.*` optimizers.

```python
class TorchOptimizer(OptimizerProtocol):
    def __init__(self, parameters, optimizer_name: str, **kwargs):
        # extract torch tensors from Parameter.data
        # import torch.optim; opt_cls = getattr(optim, optimizer_name)
        # self._opt = opt_cls(torch_params, **kwargs)
    def step(self): self._opt.step()
    def zero_grad(self): self._opt.zero_grad()
```

Config: `optimizer: {name: Adam, lr: 0.001}` or `optimizer: {name: SGD, lr: 0.01, momentum: 0.9}`.

### 6. Example notebooks

**M4a — `examples/m4a-torch-mnist/mnist.ipynb`:**
- Simple CNN: Conv2d→ReLU→MaxPool→Conv2d→ReLU→MaxPool→FC→FC
- `ImageFolderDataSource` + `TorchModel` + `TorchLoss` + `TorchOptimizer`
- Dataset: MNIST (downloaded via torchvision, saved to disk as folders)
- < 2 min training, > 98% test accuracy

**M4b — `examples/m4b-torch-cifar10/cifar10.ipynb`:**
- Deeper CNN: 3×Conv2d→ReLU→MaxPool → FC → FC
- Same adapter pattern
- Dataset: CIFAR-10
- ~5 min training, > 75% test accuracy

---

## Test strategy

### Unit tests

| File | Tests | Coverage |
|------|-------|----------|
| `test_image_folder.py` | 10+ tests | construction, lazy load, __len__, __iter__ shape/content, empty dir, bad files, shuffle determinism, single-class, class name ordering |
| `test_transforms.py` | 6+ tests | pass-through (identity), PIL→tensor conversion, augmentation no-op on targets, __len__ delegation, empty stream, chained transforms (Compose) |
| `test_torch_adapter.py` | 25+ tests | TorchModel: forward(tensor), forward(numpy), parameter shape/names, train/eval toggle, module state reflects toggle. TorchLoss: CrossEntropyLoss, MSELoss, Loss.value is float, Loss.backward() populates grad. TorchOptimizer: SGD step, Adam step, zero_grad, lr effect, parameter mutation verified. |
| `test_torch_adapter_config.py` | 8+ tests | TorchLoss config construction (loss_name + kwargs), TorchOptimizer config construction (optimizer_name + kwargs), invalid names raise, kwargs passthrough |

### Contract tests (added to existing `test_model_contract.py`)

| Test | Detail |
|------|--------|
| `test_forward_shape_matches_batch_size` | parametrize → add `"torch"` |
| `test_train_eval_mode_cycle_no_error` | parametrize → add `"torch"` |
| `test_full_pipeline_on_titanic` | parametrize → add `"torch"` |
| `test_parameters_are_mutable` | NEW: verify optimizer step changes Parameter.data |
| `test_torch_tensor_passthrough` | NEW: torch in → torch out, zero-copy |

### Integration test

| Test | Detail |
|------|--------|
| `test_torch_mnist_pipeline` | Full 6-stage pipeline with TorchAdapter on synthetic 4×4 "images" (tiny_tensor_mnist.csv equivalent) |
| `test_image_folder_to_torch_e2e` | ImageFolderDataSource → TransformedDataStream → TorchModel → train 1 epoch |

### Fixtures

- `tests/fixtures/tiny_image_folder/` — 3 classes, 2 images each (4×4 PNG, generated programmatically in test or committed)
  ```
  tiny_image_folder/
  ├── cat/
  │   ├── 01.png
  │   └── 02.png
  ├── dog/
  │   ├── 01.png
  │   └── 02.png
  └── bird/
      ├── 01.png
      └── 02.png
  ```
- Generated in `conftest.py` via `pytest.fixture` using PIL, NOT committed as binary files (keep repo small)

---

## Format conversion rules (by path)

| Path | Input → Transform → Model | Overhead |
|------|---------------------------|----------|
| Image + torch (hot) | PIL → torchvision → torch.Tensor → TorchModel → torch.Tensor | zero |
| CSV + torch | np.ndarray → TorchModel.forward → torch.as_tensor → .numpy() | 1 copy in, 1 copy out |
| Image + numpy | PIL → numpy transform → np.ndarray → NumpyModel | zero |
| CSV + sklearn | np.ndarray → SklearnModel → np.ndarray | zero |

---

## Non-goals (explicitly out of scope)

- Torch JIT / TorchScript export — Phase 4
- GPU device placement — already exists in `pipeline/utils/device.py`, TorchModel can use `module.to(device)` before wrapping
- Data augmentation as a pipeline stage — TransformedDataStream handles it declaratively
- torchvision.datasets integration — we use our own ImageFolderDataSource, user downloads data externally
- Mixed-precision training — Phase 4

---

## Files changed / created

```
NEW:
  pipeline/data/image_folder.py
  pipeline/data/transforms.py
  pipeline/adapters/torch_adapter.py
  tests/unit/test_image_folder.py
  tests/unit/test_transforms.py
  tests/unit/test_torch_adapter.py
  tests/fixtures/tiny_image_folder/   (generated by fixture, not committed)
  examples/m4a-torch-mnist/mnist.ipynb
  examples/m4b-torch-cifar10/cifar10.ipynb

MODIFIED:
  tests/contract/test_model_contract.py   (add "torch" parametrization)
  tests/unit/conftest.py                  (add torch fixtures)
  pipeline/__init__.py                    (export new public API)
  mkdocs.yml                              (add new module API pages)
```

---

## Self-review

- [x] No placeholders or TODOs
- [x] All imports in pipeline/ are lazy (torch, PIL imported inside functions)
- [x] No framework import at module level in pipeline/
- [x] Type conversion rules are explicit per path
- [x] Test counts are estimates — actual counts determined by TDD
- [x] Contract test parametrization consistent with existing pattern
- [x] M4 notebooks follow M1/M2-M3 structure
- [x] No API changes to existing modules (add-only)
