<sub>🌐 <b>English</b> · <a href="README.zh.md">中文</a></sub>

# Universal DL Pipeline

> *"A deep learning pipeline that teaches you how it works — not just that it works."*

[![Ko-fi](https://img.shields.io/badge/Support-ko--fi-FF5E5B?style=flat&logo=ko-fi&logoColor=white)](https://ko-fi.com/requiema)
[![Afdian](https://img.shields.io/badge/Support-爱发电-946CE6?style=flat)](https://afdian.com/a/requiema)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://pypi.org/project/universal-dl-pipeline/)

<br>

**A teaching-first, framework-agnostic deep learning pipeline library. Stages are inspectable, replaceable, skippable — no black boxes.**

The pipeline defines *when* things happen (six stages: load → extract → build → train → evaluate → export). Concrete implementations — sklearn, numpy, PyTorch, or your own — define *how*. Protocols are the contract, not framework classes. Students descend the abstraction ladder one rung at a time: same pipeline structure, progressively richer implementations.

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
| **Pipeline** | 6-stage template method with hook dispatch — you implement the stages, the pipeline orchestrates | `pipeline.run("train")` |
| **Registry** | Flat-dict dependency injection: `register("kind", "name")` + `build(kind, name)` | `@register("backbone", "resnet18")` |
| **Hooks** | Cross-cutting concerns (logging, progress, early-stop) that don't touch pipeline logic | `pipeline.add_hook(ProgressHook())` |
| **TrainLoop** | Epoch/batch loop with hook dispatch — skips backward for non-gradient models | `TrainLoop(model, stream, opt, loss, epochs=10)` |
| **Data** | CSV data source + train/validation split — pandas-backed, reproducible shuffles | `CsvDataSource("train.csv", batch_size=32)` |
| **Metrics** | Pure numpy metric functions in a replaceable container — add new metrics without touching evaluate() | `Metrics(accuracy=acc, f1=f1).compute(y_true, y_pred)` |
| **Config** | Single dataclass, 17 fields, sensible defaults — YAML serializable | `Config.from_yaml("config.yaml")` |

---

## Repository Structure

```
Universal_DL_Pipeline/
├── pipeline/                 # framework core (what goes into the wheel)
│   ├── protocols.py          #   abstract interfaces (framework-agnostic)
│   ├── pipeline.py           #   BasePipeline ABC + PipelineState
│   ├── registry.py           #   DI: register() + build()
│   ├── config.py             #   Config dataclass + YAML loading
│   ├── hooks/                #   BaseHook + ProgressHook
│   ├── data/                 #   CsvDataSource + train_test_split
│   ├── training/             #   TrainLoop
│   ├── evaluation/           #   Metrics container + pure functions
│   ├── export/               #   to_csv
│   └── utils/                #   device, seed
├── tests/                    # unit + integration (225 tests, TDD)
├── examples/                 # AI syllabus code companion (Phase 2+)
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
| 2 | Table adapters — sklearn + sympy→numpy | 📋 Planned |
| 3 | Image + TorchAdapter | 📋 Planned |
| 4 | Export + Inference (production-grade) | 📋 Planned |
| 5 | HPO + Ensemble | 📋 Planned |
| 6 | Sequences / NLP | 📋 Planned |

---

## Limitations

- **No torch / sklearn / sympy in core.** Phase 1 operates purely on protocols with fake models. Real framework adapters arrive in Phase 2-3.
- **Not production-grade yet.** The pipeline structure is solid, but checkpointing, distributed training, and production inference are Phase 4+.
- **Python 3.11+ only.** Uses `from __future__ import annotations` and modern typing syntax.
- **Titanic-sized data only.** In-memory splits. Streaming for large datasets is deferred.

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
