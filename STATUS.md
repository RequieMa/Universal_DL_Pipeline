# Project Status — 2026-08-08


## Where we are

Phase 4 (Checkpoint + Inference) **complete**. Phase 3 (Image + TorchAdapter) **complete**. Phase 2 (Table Adapters) **complete**.
331 tests passing, 5 skipped (sklearn on system python). Pipeline `v0.2.0` — sklearn,
numpy, and torch adapters ready, M1/M2/M3/M4 example notebooks written.

## What was built (Phase 1 + 2)

```
pipeline/
├── __init__.py           # package exports, v0.2.0
├── __main__.py           # python -m pipeline → version + device + components
├── protocols.py          # Loss, ArrayLike, Parameter, all protocol ABCs
├── pipeline.py           # BasePipeline ABC + PipelineState + hooks @property
├── registry.py           # flat-dict DI: @register / build
├── config.py             # Config dataclass (YAML, env, kwargs)
├── data/
│   ├── csv_source.py     # CsvDataSource
│   ├── image_folder.py   # NEW (Phase 3) — ImageFolderDataSource
│   ├── transforms.py     # NEW (Phase 3) — TransformedDataStream — CSV → Batch iterator
│   ├── split.py          # train_test_split() → (train_stream, val_stream)
│   └── utils.py          # collect_arrays() — drain DataStream → (X, y)
├── training/
│   ├── train_loop.py     # TrainLoop — epoch/batch loop + training hook dispatch
│   ├── optimizers.py     # SGD, Adam — pure-numpy update rules
│   └── losses.py         # MSELoss, CrossEntropyLoss — model-aware backward
├── evaluation/
│   └── metrics.py        # Metrics container + accuracy/precision/recall/F1/confusion_matrix
├── export/
│   └── to_csv.py         # to_csv() — numpy array → CSV
├── hooks/
│   ├── base.py           # BaseHook — five hook points, all no-ops
│   └── progress.py       # ProgressHook — tqdm progress bar
├── adapters/             # NEW (Phase 2+3)
│   ├── sklearn_adapter.py # SklearnModel, StubLoss, StubOptimizer
│   ├── numpy_adapter.py   # NumpyModel, NumpyOptimizer
│   └── torch_adapter.py   # NEW (Phase 3) — TorchModel, TorchLoss, TorchOptimizer
  │   └── torch_adapter.py   # NEW (Phase 3) — TorchModel, TorchLoss, TorchOptimizer
└── utils/                # device.py, seed.py
run.py                    # CLI: --config --mode [train|infer]
```

## Test suite

283 passed, 1 skipped (torch). 67 new tests added in Phase 2.

```
tests/
├── unit/                     # per-module unit tests
│   ├── test_protocols.py     # +Loss tests
│   ├── test_pipeline.py      # +hooks @property, +train() assert guards
│   ├── test_pipeline_state.py
│   ├── test_metrics.py       # 28 tests: 5 pure functions + Metrics container
│   ├── test_train_loop.py    # 25 tests: hooks, backward skip, early stop
│   ├── test_csv_source.py    # 13 tests: CSV loading, shuffle, edge cases
│   ├── test_split.py         # 12 tests: ratio, reproducibility, non-overlap
│   ├── test_progress.py      # 5 tests: tqdm mock-based
│   ├── test_to_csv.py        # 6 tests: round-trip, edge cases
│   ├── test_main.py          # 4 tests: CLI output
│   ├── test_data_utils.py    # 3 tests: collect_arrays (Phase 2)
│   ├── test_sklearn_adapter.py # 10 tests: StubLoss/Optimizer + SklearnModel (Phase 2)
│   ├── test_optimizers.py    # 10 tests: 5 SGD + 5 Adam (Phase 2)
│   ├── test_losses.py        # 15 tests: 7 MSE + 8 CrossEntropy (Phase 2)
│   ├── test_numpy_adapter.py # 17 tests: 12 NumpyModel + 5 NumpyOptimizer (Phase 2)
│   ├── test_image_folder.py  # NEW (Phase 3) — 12 tests
│   ├── test_transforms.py    # NEW (Phase 3) — 6 tests
│   ├── test_torch_adapter.py # NEW (Phase 3) — 31 tests
│   └── conftest.py           # shared test doubles
├── contract/                 # (Phase 2)
│   └── test_model_contract.py # 6 tests: SklearnModel + NumpyModel contract
├── fixtures/
│   └── tiny_titanic.csv      # 10 rows, committed
└── integration/
    ├── test_pipeline_e2e.py  # 2 tests: train mode + infer mode
    └── test_torch_e2e.py     # NEW (Phase 3) — 3 tests: torch pipeline + image folder E2E
```

## Examples (Phase 2)

```
examples/
├── m1-sklearn/
│   └── titanic.ipynb          # sklearn LogisticRegression on Titanic
├── m2-m3-numpy/
    │   └── gradient_to_mlp.ipynb
├── m4a-torch-mnist/
    │   └── mnist.ipynb     # NEW (Phase 3)
└── m4b-torch-cifar10/
        └── cifar10.ipynb  # NEW (Phase 3)
```

## Tooling

- **Env:** `uv` (`uv sync --dev`, `uv run pytest`, `uv run ruff check .`)
- **Build:** hatchling (`uv build --wheel` → only `pipeline/` in wheel)
- **Docs:** Material for MkDocs + mkdocstrings (`uv run mkdocs build`)
- **CI:** 331 tests in ~1.9s — well under 60s target

## Completion gate ✅

- [x] `python -m pipeline` prints version 0.2.0 + device info
- [x] `python run.py --config config.yaml --mode train` runs full pipeline (fake model)
- [x] `python run.py --config config.yaml --mode infer` runs inference path
- [x] All Phase 2 imports: `SklearnModel`, `NumpyModel`, `NumpyOptimizer`, `SGD`, `Adam`, `MSELoss`, `CrossEntropyLoss`, `collect_arrays`
- [x] Contract tests parametrized over SklearnModel and NumpyModel (6 tests)
- [x] Core works without sklearn: `python -c "from pipeline import BasePipeline"`
- [x] 100% TDD coverage for all new Phase 2 modules
- [x] ruff clean, mypy strict (stub warnings expected on py3.13)
- [x] All Phase 3 imports: `ImageFolderDataSource`, `TransformedDataStream`, `TorchModel`, `TorchLoss`, `TorchOptimizer`
- [x] Contract tests parametrized over SklearnModel, NumpyModel, TorchModel (9 tests)
- [x] Core works without torch: `python -c "from pipeline.data.image_folder import ImageFolderDataSource"`
- [x] MNIST notebook executes (>95%) | CIFAR-10 notebook executes (>70%)
- [x] TDD coverage for all new Phase 3 modules
- [x] TorchCheckpoint save/load with model+optimizer+epoch+history
- [x] BasePipeline default export() writes predictions CSV
- [x] load_checkpoint stage added to infer mode flow
- [x] CheckpointHook saves latest.pt and best.pt per epoch
- [x] EarlyStopHook stops training when metric plateaus
- [x] 26 new tests (9 unit checkpoint + 7 early_stop + 4 checkpoint hook + 3 contract + 3 integration)
- [x] ruff clean, backward compatible (existing subclasses unchanged)

## Phase roadmap

```
Phase 0  ✅ DONE — Skeleton (protocols, BasePipeline, registry, hooks, config, utils)
Phase 1  ✅ DONE — Pipeline framework (CsvDataSource, TrainTestSplit, TrainLoop,
                  Metrics, ProgressHook, to_csv, CLI)
Phase 2  ✅ DONE — Table adapters: sklearn + numpy → M1/M2/M3
                  (SklearnModel, NumpyModel, SGD, Adam, MSELoss, CrossEntropyLoss,
                   NumpyOptimizer, StubLoss/Optimizer, collect_arrays, contract tests)
Phase 3  ✅ DONE — Image + TorchAdapter → M4
Phase 4  ✅ DONE — Checkpoint + Inference (TorchCheckpoint, CheckpointHook, EarlyStopHook,
                  load_checkpoint stage, default export)
Phase 5         — HPO + Ensemble
Phase 6         — Sequences / NLP → M5
M6/M7           — LLM, add-only (no API changes)
```
