<sub>🌐 <a href="README.md">English</a> · <b>中文</b></sub>

# Universal DL Pipeline

> *"一个让你真正理解它如何工作的深度学习管线 — 而不只是知道它在工作。"*

[![Ko-fi](https://img.shields.io/badge/Support-ko--fi-FF5E5B?style=flat&logo=ko-fi&logoColor=white)](https://ko-fi.com/requiema)
[![Afdian](https://img.shields.io/badge/Support-爱发电-946CE6?style=flat)](https://afdian.com/a/requiema)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://pypi.org/project/universal-dl-pipeline/)

<br>

**一个教学优先、框架无关的深度学习管线库。每个阶段都可检查、可替换、可跳过 — 没有黑盒。**

管线定义了*什么时候*发生（六个阶段：加载 → 特征 → 构建 → 训练 → 评估 → 导出）。具体实现 — sklearn、numpy、PyTorch 或你自己的 — 定义*怎么做*。Protocol 是契约，不是框架类。学生可以逐级下沉抽象层次：同样的管线结构，渐进丰富的实现方式。

[快速开始](#快速开始) · [能做什么](#能做什么) · [仓库结构](#仓库结构)

---

## 快速开始

```bash
pip install universal-dl-pipeline
python -m pipeline                    # → 版本 + 设备 + 已注册组件
```

从源码安装：

```bash
git clone https://github.com/RequieMa/Universal_DL_Pipeline
cd Universal_DL_Pipeline
pip install -e .
python run.py --config config.yaml --mode train
```

---

## 能做什么

| 能力 | 功能 | 示例 |
|------|------|------|
| **协议层** | 框架无关的抽象接口：ModelProtocol、DataStream、LossProtocol、OptimizerProtocol | `model.forward(batch.inputs)` |
| **管线** | 6 阶段模板方法 + Hook 调度 — 你实现阶段逻辑，管线来编排 | `pipeline.run("train")` |
| **注册表** | 扁平字典依赖注入：`register("kind", "name")` + `build(kind, name)` | `@register("backbone", "resnet18")` |
| **Hook 系统** | 横切关注点（日志、进度条、早停）不侵入管线逻辑 | `pipeline.add_hook(ProgressHook())` |
| **训练循环** | Epoch/Batch 循环 + Hook 调度 — 无梯度模型自动跳过 backward | `TrainLoop(model, stream, opt, loss, epochs=10)` |
| **数据层** | CSV 数据源 + 训练/验证集拆分 — pandas 支持，可复现的随机打乱 | `CsvDataSource("train.csv", batch_size=32)` |
| **指标** | 纯 numpy 指标函数 + 可替换容器 — 新增指标不需要改 evaluate() | `Metrics(accuracy=acc, f1=f1).compute(y_true, y_pred)` |
| **配置** | 单个 dataclass，17 个字段，合理默认值 — 支持 YAML | `Config.from_yaml("config.yaml")` |

---

## 仓库结构

```
Universal_DL_Pipeline/
├── pipeline/                 # 框架核心（打包进 wheel 的内容）
│   ├── protocols.py          #   抽象接口（框架无关）
│   ├── pipeline.py           #   BasePipeline ABC + PipelineState
│   ├── registry.py           #   依赖注入：register() + build()
│   ├── config.py             #   Config dataclass + YAML 加载
│   ├── hooks/                #   BaseHook + ProgressHook
│   ├── data/                 #   CsvDataSource + train_test_split
│   ├── training/             #   TrainLoop
│   ├── evaluation/           #   Metrics 容器 + 纯函数指标
│   ├── export/               #   to_csv
│   └── utils/                #   device, seed
├── tests/                    # 单元测试 + 集成测试（225 个测试，TDD）
├── examples/                 # AI 教学大纲配套代码（Phase 2+）
├── docs/                     # MkDocs Material 源文件 + 设计文档
├── legacy/                   # 旧框架脚本（仅供参考）
├── data/                     # 数据集（gitignored，下载说明见 docs）
├── run.py                    # CLI：--config --mode [train|infer]
└── pyproject.toml            # hatchling 构建, ruff, mypy, pytest
```

---

## 阶段路线图

| 阶段 | 范围 | 状态 |
|------|------|------|
| 0 | 骨架 — protocols、BasePipeline、registry、hooks、config、utils | ✅ 已完成 |
| 1 | 框架 — CsvDataSource、TrainLoop、Metrics、ProgressHook、CLI | ✅ 已完成 |
| 2 | 表格适配器 — sklearn + sympy→numpy | 📋 计划中 |
| 3 | 图像 + TorchAdapter | 📋 计划中 |
| 4 | 导出 + 推理（生产级） | 📋 计划中 |
| 5 | 超参优化 + 集成 | 📋 计划中 |
| 6 | 序列 / NLP | 📋 计划中 |

---

## 局限性

- **核心不依赖 torch / sklearn / sympy。** Phase 1 纯基于 Protocol 运行，使用假模型。真正的框架适配器在 Phase 2-3 引入。
- **目前非生产级。** 管线结构已完备，但 checkpoint、分布式训练、生产推理在 Phase 4+ 实现。
- **仅支持 Python 3.11+。** 使用了 `from __future__ import annotations` 和现代 typing 语法。
- **仅适配 Titanic 量级数据。** 拆分操作在内存中进行。大数据集的流式拆分会后续实现。

---

## Connect · 关于作者

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
