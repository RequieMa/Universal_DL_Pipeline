<sub>🌐 <b>English</b> · <a href="README.zh.md">中文</a></sub>

# Universal DL Pipeline

> *"A deep learning pipeline that teaches you how it works — not just that it works."*

[![Ko-fi](https://img.shields.io/badge/Support-ko--fi-FF5E5B?style=flat&logo=ko-fi&logoColor=white)](https://ko-fi.com/requiema)
[![Afdian](https://img.shields.io/badge/Support-爱发电-946CE6?style=flat)](https://afdian.com/a/requiema)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://pypi.org/project/universal-dl-pipeline/)
[![Tests](https://img.shields.io/badge/Tests-416_passing-5a5?style=flat)](https://github.com/RequieMa/Universal_DL_Pipeline)

<br>

**A teaching-first, framework-agnostic deep learning pipeline library. Stages are inspectable, replaceable, skippable — no black boxes.**

The pipeline defines *when* things happen. Concrete implementations — sklearn, numpy, PyTorch, or your own — define *how*. Protocols are the contract, not framework classes. Students descend the abstraction ladder one rung at a time: same pipeline structure, progressively richer implementations.

[Quick Start](#quick-start) · [What's Here](#whats-here) · [Repository Structure](#repository-structure)

---

## Quick Start

```bash
pip install universal-dl-pipeline
python -m pipeline                    # → version + device + registered components
```

Or from source:

```bash
git clone https://github.com/RequieMa/Universal_DL_Pipeline
cd Universal_DL_Pipeline
pip install -e .
python run.py --config config.yaml --mode train
```

---

## What's Here

| Capability | What It Does | Example |
|------------|-------------|---------|
| **Protocols** | Framework-agnostic interfaces: ModelProtocol, DataStream, LossProtocol, OptimizerProtocol | `model.forward(batch.inputs)` |
| **Pipeline** | 7-stage template method with hook dispatch — you implement the stages, the pipeline orchestrates | `pipeline.run("train")` |
| **Registry** | Flat-dict DI: `register("kind", "name")` + `build(kind, name)` | `@register("backbone", "resnet18")` |
| **Hooks** | Cross-cutting concerns — progress bars, checkpointing, early stopping | `pipeline.add_hook(CheckpointHook())` |
| **TrainLoop** | Epoch/batch loop with hook dispatch — skips backward for non-gradient models | `TrainLoop(model, stream, opt, loss, epochs=10)` |
| **Data** | CSV, image folder, text sources — lazy-load, reproducible shuffles | `CsvDataSource("train.csv", batch_size=32)` |
| **Transforms** | Batch transform decorator — plug any preprocessing pipeline | `TransformedDataStream(source, to_tensor)` |
| **Metrics** | Pure numpy metric functions in a replaceable container | `Metrics(accuracy=acc, f1=f1).compute(y_true, y_pred)` |
| **Config** | Single dataclass, 17 fields, YAML serializable | `Config.from_yaml("config.yaml")` |
| **Adapters** | sklearn, numpy, PyTorch wrappers — all satisfy the same protocols | `TorchModel(nn.Linear(784, 10))` |
| **Checkpoint** | Framework-specific save/load — model + optimizer + epoch + history | `TorchCheckpoint.save(state, "best.pt")` |
| **HPO** | Grid search, random search over Config hyperparameters | `GridSearch(MyPipeline, config, grid).run()` |
| **Ensemble** | Voting (hard/soft) + stacking meta-model over trained models | `VotingEnsemble(models, mode="soft").predict(x)` |
| **Tokenizer** | Pure-Python whitespace tokenizer — fit, encode, decode, OOV handling | `SimpleTokenizer().fit(texts).encode("hello")` |
| **Text Data** | CSV text column → Batch iterator with string inputs | `TextDataSource("reviews.csv", text_column="review")` |

---

## Repository Structure

```
Universal_DL_Pipeline/
├── pipeline/                 # framework core (what goes into the wheel)
│   ├── protocols.py          #   abstract interfaces (framework-agnostic)
│   ├── pipeline.py           #   BasePipeline ABC + PipelineState
│   ├── registry.py           #   DI: register() + build()
│   ├── config.py             #   Config dataclass + YAML loading
│   ├── hooks/                #   BaseHook + ProgressHook + CheckpointHook + EarlyStopHook
│   ├── data/                 #   CsvDataSource + ImageFolderDataSource +
│   │                         #   TransformedDataStream + TextDataSource + tokenizer + split
│   ├── training/             #   TrainLoop + SGD/Adam optimizers + MSE/CrossEntropyLoss
│   ├── evaluation/           #   Metrics container + accuracy/precision/recall/F1
│   ├── export/               #   to_csv + TorchCheckpoint
│   ├── adapters/             #   SklearnModel + NumpyModel + TorchModel/Loss/Optimizer
│   ├── hpo/                  #   GridSearch + RandomSearch + VotingEnsemble + StackingEnsemble
│   └── utils/                #   device, seed
├── tests/                    # unit + contract + integration (416 tests, TDD)
├── examples/                 # AI syllabus code companion
│   ├── m1-sklearn/           #   Titanic logistic regression
│   ├── m2-m3-numpy/          #   gradient → MLP (sympy + numpy)
│   ├── m4a-torch-mnist/      #   CNN on MNIST (99.1% accuracy)
│   ├── m4b-torch-cifar10/    #   CNN on CIFAR-10 (76.2% accuracy)
│   ├── m5-text-imdb/         #   LSTM on IMDB sentiment (80.7% accuracy)
│   └── m6-llm-distilgpt2/    #   distilgpt2 LoRA fine-tune + FastAPI serve
├── docs/                     # MkDocs Material source + superpowers specs/plans
├── legacy/                   # pre-framework flat scripts (reference only)
├── data/                     # datasets (gitignored, download instructions in docs)
├── run.py                    # CLI: --config --mode [train|infer]
└── pyproject.toml            # hatchling build, ruff, mypy, pytest
```

---

## Phase Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Skeleton — protocols, BasePipeline, registry, hooks, config, utils | ✅ Done |
| 1 | Framework — CsvDataSource, TrainLoop, Metrics, ProgressHook, CLI | ✅ Done |
| 2 | Table adapters — sklearn + numpy (M1, M2, M3) | ✅ Done |
| 3 | Image + TorchAdapter — ImageFolderDataSource, TorchModel/Loss/Optimizer, MNIST, CIFAR-10 (M4) | ✅ Done |
| 4 | Checkpoint + Inference — TorchCheckpoint, CheckpointHook, EarlyStopHook, load_checkpoint stage, default export | ✅ Done |
| 5 | HPO + Ensemble — GridSearch, RandomSearch, VotingEnsemble, StackingEnsemble | ✅ Done |
| 6 | Text + NLP — TextDataSource, SimpleTokenizer, IMDB LSTM (M5) | ✅ Done |
| M6/M7 | LLM — distilgpt2 LoRA fine-tune + FastAPI OpenAI-compatible serving | ✅ Done |

---

## Verified Matrix

Each adapter is tested against the same contract and integration suite:

| Adapter | Framework | Contract Tests | Unit Tests | Notebook |
|---------|-----------|----------------|------------|----------|
| `SklearnModel` | scikit-learn | ✅ forward shape, mode cycle, full pipeline | 10 | M1 Titanic |
| `NumpyModel` | numpy | ✅ forward shape, mode cycle, full pipeline | 17 | M2-M3 Gradient→MLP |
| `TorchModel` | PyTorch | ✅ forward shape, mode cycle, full pipeline | 31 | M4a MNIST, M4b CIFAR-10, M5 IMDB, M6 distilgpt2 |

**All adapters pass the same 3 contract tests.** Switch between them by changing one import — the pipeline doesn't care.

### Data sources verified:

| Source | Input Type | Backend | Tests |
|--------|-----------|---------|-------|
| `CsvDataSource` | CSV → numpy arrays | pandas (lazy) | 13 |
| `ImageFolderDataSource` | Folder → PIL Images | PIL (lazy) | 12 |
| `TextDataSource` | CSV → list[str] | pandas (lazy) | 12 |

### Cross-cutting concerns verified:

| Component | Tests | Description |
|-----------|-------|-------------|
| `CheckpointHook` | 4 | Saves best + latest checkpoints per epoch |
| `EarlyStopHook` | 7 | Stops training when metric plateaus |
| `TorchCheckpoint` | 9 + 3 contract | Save/load model + optimizer + epoch + history |
| `GridSearch` | 18 | Cartesian product, run completes, best trial |
| `RandomSearch` | — (shared with GridSearch) | Sampling without replacement, reproducibility |
| `VotingEnsemble` | 9 | Hard/soft voting, evaluate on stream |
| `StackingEnsemble` | 5 | Meta-model fit + predict |
| `SimpleTokenizer` | 12 | Fit/encode/decode, OOV, min_freq, max_vocab |

---

## Test Suite

```bash
uv sync --dev
uv run pytest                    # 416 passed, 5 skipped (~2s)
uv run pytest -m slow            # 15 contract + integration tests
uv run ruff check .              # clean
uv run mypy                      # strict mode
```

**Test layers:**
- **Unit** (`tests/unit/`) — per-module, TDD, fakes only (no frameworks)
- **Contract** (`tests/contract/`) — same tests run on SklearnModel, NumpyModel, TorchModel
- **Integration** (`tests/integration/`) — full pipeline E2E with real adapters

**What is NOT tested / intentionally skipped:**
- `torch` tests skip on torch-less environments (5 skips — sklearn on system python)
- GPU-specific behavior — tests run on CPU by default; notebooks use GPU when available
- Network-dependent operations — HuggingFace downloads in notebooks need internet

---

## Limitations

- **Python 3.11+ only.** Uses `from __future__ import annotations` and modern typing.
- **Teaching scale, not production scale.** Streaming for large datasets, distributed training, and mixed-precision are deferred.
- **PyTorch tested; JAX/TensorFlow not yet.** The TorchAdapter covers PyTorch; protocols are framework-agnostic but only PyTorch has a full adapter implementation.
- **LLM serving is a demo.** The distilgpt2 fine-tune + FastAPI serve proves the architecture works end-to-end, but the 82M model cannot produce useful code. Larger models (Qwen 0.5B, SmolLM2 1.7B) fit the same pattern with better quality.

---

## Connect

<div align="center">

| | | |
|---|---|---|
| 📧 | Email | [mazengou@gmail.com](mailto:mazengou@gmail.com) |
| 🌐 | Personal Site | [requiema.github.io](https://requiema.github.io) |
| 📝 | dev.to | [dev.to/requiema](https://dev.to/requiema) |
| 𝕏 | X | [x.com/mazengou](https://x.com/mazengou) |
| 👾 | Reddit | [u/Leather_Rip7919](https://www.reddit.com/user/Leather_Rip7919/) |
| 🔖 | 掘金 | [juejin.cn/user/76300220645242](https://juejin.cn/user/76300220645242) |
| 📦 | Gitee | [gitee.com/requiema](https://gitee.com/requiema) |
| 📖 | 知乎 | [zhihu.com/people/consilivm](https://www.zhihu.com/people/consilivm) |
| 🎬 | Bilibili | 镇魂曲麦 |
| 📱 | 公众号 | 镇魂曲麦 |

</div>
