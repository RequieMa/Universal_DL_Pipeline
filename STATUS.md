# Project Status — 2026-07-30

## Where we are

Phase 1 (Pipeline Framework) **complete**. 225 tests passing. Framework layer built on Phase 0
protocols — no torch/sklearn/sympy dependencies.

## What was built (Phase 1)

```
pipeline/
├── __init__.py          # package exports (updated)
├── __main__.py          # python -m pipeline → version + device + components
├── protocols.py         # +Loss class, +val_data_stream, +Metrics type
├── pipeline.py          # +hooks @property, +train() default impl (TrainLoop delegate)
├── registry.py          # unchanged
├── config.py            # unchanged
├── data/                # NEW
│   ├── csv_source.py    # CsvDataSource — CSV → Batch iterator
│   └── split.py         # train_test_split() → (train_stream, val_stream)
├── training/            # NEW
│   └── train_loop.py    # TrainLoop — epoch/batch loop + training hook dispatch
├── evaluation/          # NEW
│   └── metrics.py       # Metrics container + accuracy/precision/recall/F1/confusion_matrix
├── export/              # NEW
│   └── to_csv.py        # to_csv() — numpy array → CSV
├── hooks/               # restructured: module → package
│   ├── base.py          # BaseHook (moved from hooks.py)
│   └── progress.py      # ProgressHook — tqdm progress bar
└── utils/               # unchanged
run.py                   # NEW — CLI: --config --mode [train|infer]
```

## Test suite

225 passed, 1 skipped (torch). Integration E2E test exercises the full pipeline:
CSV → TrainTestSplit → TrainLoop → Metrics → to_csv.

```
tests/
├── unit/                # per-module unit tests
│   ├── test_protocols.py      # +Loss tests
│   ├── test_pipeline.py       # +hooks @property, +train() default tests
│   ├── test_pipeline_state.py # PipelineState tests (split out)
│   ├── test_metrics.py        # 28 tests: 5 pure functions + Metrics container
│   ├── test_train_loop.py     # 25 tests: hooks, backward skip, early stop, etc.
│   ├── test_csv_source.py     # 13 tests: CSV loading, shuffle, edge cases
│   ├── test_split.py          # 12 tests: ratio, reproducibility, non-overlap
│   ├── test_progress.py       # 5 tests: tqdm mock-based
│   ├── test_to_csv.py         # 6 tests: round-trip, edge cases
│   ├── test_main.py           # 4 tests: CLI output
│   └── conftest.py            # shared test doubles
├── fixtures/
│   └── tiny_titanic.csv       # 10 rows, committed
└── integration/
    └── test_pipeline_e2e.py   # 2 tests: train mode + infer mode
```

## Tooling

- **Env:** `uv` (`uv sync --dev`, `uv run pytest`, `uv run ruff check .`)
- **Build:** hatchling (`uv build --wheel` → only `pipeline/` in wheel)
- **Docs:** Material for MkDocs + mkdocstrings (`uv run mkdocs build`)
- **CI:** 225 tests in 0.3s — unit < 60s target met

## Completion gate ✅

- [x] `python -m pipeline` prints version + device info
- [x] `python run.py --config config.yaml --mode train` runs full pipeline (fake model)
- [x] `python run.py --config config.yaml --mode infer` runs inference path
- [x] `python run.py --help` documents all CLI options
- [x] 100% new code TDD-covered
- [x] ruff clean, mypy (stub warnings expected on py3.13)

## Phase roadmap

```
Phase 0  ✅ DONE — Skeleton (protocols, BasePipeline, registry, hooks, config, utils)
Phase 1  ✅ DONE — Pipeline framework (CsvDataSource, TrainTestSplit, TrainLoop,
                  Metrics, ProgressHook, to_csv, CLI)
Phase 2  NEXT  — Table adapters: sklearn + sympy→numpy → M1/M2/M3
Phase 3         — Image + TorchAdapter → M4
Phase 4         — Export + Inference (production-grade)
Phase 5         — HPO + Ensemble
Phase 6         — Sequences / NLP → M5
M6/M7           — LLM, add-only (no API changes)
```
