<sub>🌐 <a href="README.md">English</a> · <b>中文</b></sub>

# Universal DL Pipeline

> *"一个让你真正理解它如何工作的深度学习管线 — 而不只是知道它在工作。"*

[![Ko-fi](https://img.shields.io/badge/Support-ko--fi-FF5E5B?style=flat&logo=ko-fi&logoColor=white)](https://ko-fi.com/requiema)
[![Afdian](https://img.shields.io/badge/Support-爱发电-946CE6?style=flat)](https://afdian.com/a/requiema)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://pypi.org/project/universal-dl-pipeline/)
[![Tests](https://img.shields.io/badge/Tests-416_passing-5a5?style=flat)](https://github.com/RequieMa/Universal_DL_Pipeline)

<br>

**一个教学优先、框架无关的深度学习管线库。每个阶段都可检查、可替换、可跳过 — 没有黑盒。**

管线定义了*什么时候*发生。具体实现 — sklearn、numpy、PyTorch 或你自己的 — 定义*怎么做*。Protocol 是契约，不是框架类。学生可以逐级下沉抽象层次：同样的管线结构，渐进丰富的实现方式。

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
| **管线** | 7 阶段模板方法 + Hook 调度 — 你实现阶段逻辑，管线来编排 | `pipeline.run("train")` |
| **注册表** | 扁平字典依赖注入：`register("kind", "name")` + `build(kind, name)` | `@register("backbone", "resnet18")` |
| **Hook 系统** | 横切关注点（进度条、checkpoint、早停）不侵入管线逻辑 | `pipeline.add_hook(CheckpointHook())` |
| **训练循环** | Epoch/Batch 循环 + Hook 调度 — 无梯度模型自动跳过 backward | `TrainLoop(model, stream, opt, loss, epochs=10)` |
| **数据层** | CSV、图像文件夹、文本数据源 — 惰性加载，可复现的随机打乱 | `CsvDataSource("train.csv", batch_size=32)` |
| **变换** | Batch 变换装饰器 — 接入任意预处理管线 | `TransformedDataStream(source, to_tensor)` |
| **指标** | 纯 numpy 指标函数 + 可替换容器 — 新增指标不需要改 evaluate() | `Metrics(accuracy=acc, f1=f1).compute(y_true, y_pred)` |
| **配置** | 单个 dataclass，17 个字段，合理默认值 — 支持 YAML | `Config.from_yaml("config.yaml")` |
| **适配器** | sklearn、numpy、PyTorch 包装器 — 全部满足同一套 Protocol | `TorchModel(nn.Linear(784, 10))` |
| **Checkpoint** | 框架相关的保存/加载 — 模型 + 优化器 + epoch + 历史 | `TorchCheckpoint.save(state, "best.pt")` |
| **超参优化** | Grid search + random search over Config 超参数 | `GridSearch(MyPipeline, config, grid).run()` |
| **集成** | 投票（硬/软）+ stacking 元模型 | `VotingEnsemble(models, mode="soft").predict(x)` |
| **分词器** | 纯 Python 空格分词器 — fit、encode、decode、OOV 处理 | `SimpleTokenizer().fit(texts).encode("hello")` |
| **文本数据** | CSV 文本列 → Batch 迭代器，string 输入 | `TextDataSource("reviews.csv", text_column="review")` |

---

## 仓库结构

```
Universal_DL_Pipeline/
├── pipeline/                 # 框架核心（打包进 wheel 的内容）
│   ├── protocols.py          #   抽象接口（框架无关）
│   ├── pipeline.py           #   BasePipeline ABC + PipelineState
│   ├── registry.py           #   依赖注入：register() + build()
│   ├── config.py             #   Config dataclass + YAML 加载
│   ├── hooks/                #   BaseHook + ProgressHook + CheckpointHook + EarlyStopHook
│   ├── data/                 #   CsvDataSource + ImageFolderDataSource +
│   │                         #   TransformedDataStream + TextDataSource + tokenizer + split
│   ├── training/             #   TrainLoop + SGD/Adam 优化器 + MSE/CrossEntropyLoss
│   ├── evaluation/           #   Metrics 容器 + accuracy/precision/recall/F1
│   ├── export/               #   to_csv + TorchCheckpoint
│   ├── adapters/             #   SklearnModel + NumpyModel + TorchModel/Loss/Optimizer
│   ├── hpo/                  #   GridSearch + RandomSearch + VotingEnsemble + StackingEnsemble
│   └── utils/                #   device, seed
├── tests/                    # 单元 + 契约 + 集成测试（416 个测试，TDD）
├── examples/                 # AI 教学大纲配套代码
│   ├── m1-sklearn/           #   Titanic 逻辑回归
│   ├── m2-m3-numpy/          #   梯度 → MLP（sympy + numpy）
│   ├── m4a-torch-mnist/      #   CNN 手写数字识别（99.1% 准确率）
│   ├── m4b-torch-cifar10/    #   CNN 图像分类（76.2% 准确率）
│   ├── m5-text-imdb/         #   LSTM 电影评论情感分析（80.7% 准确率）
│   └── m6-llm-distilgpt2/    #   distilgpt2 LoRA 微调 + FastAPI 服务
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
| 2 | 表格适配器 — sklearn + numpy（M1、M2、M3） | ✅ 已完成 |
| 3 | 图像 + TorchAdapter — ImageFolderDataSource、TorchModel/Loss/Optimizer、MNIST、CIFAR-10（M4） | ✅ 已完成 |
| 4 | Checkpoint + 推理 — TorchCheckpoint、CheckpointHook、EarlyStopHook、load_checkpoint 阶段、默认 export | ✅ 已完成 |
| 5 | 超参优化 + 集成 — GridSearch、RandomSearch、VotingEnsemble、StackingEnsemble | ✅ 已完成 |
| 6 | 文本 + NLP — TextDataSource、SimpleTokenizer、IMDB LSTM（M5） | ✅ 已完成 |
| M6/M7 | LLM — distilgpt2 LoRA 微调 + FastAPI OpenAI 兼容服务 | ✅ 已完成 |

---

## 已验证矩阵

每个适配器都通过了相同的契约测试和集成测试：

| 适配器 | 框架 | 契约测试 | 单元测试 | Notebook |
|---------|------|----------|----------|----------|
| `SklearnModel` | scikit-learn | ✅ forward 形状、模式切换、全管线 | 10 | M1 Titanic |
| `NumpyModel` | numpy | ✅ forward 形状、模式切换、全管线 | 17 | M2-M3 梯度→MLP |
| `TorchModel` | PyTorch | ✅ forward 形状、模式切换、全管线 | 31 | M4a MNIST、M4b CIFAR-10、M5 IMDB、M6 distilgpt2 |

**三个适配器通过同一套 3 个契约测试。** 换一行 import 就能切换框架 — 管线不在乎。

### 已验证的数据源：

| 数据源 | 输入类型 | 后端 | 测试数 |
|--------|----------|------|--------|
| `CsvDataSource` | CSV → numpy 数组 | pandas（惰性） | 13 |
| `ImageFolderDataSource` | 文件夹 → PIL Image | PIL（惰性） | 12 |
| `TextDataSource` | CSV → list[str] | pandas（惰性） | 12 |

### 已验证的横切关注点：

| 组件 | 测试数 | 描述 |
|------|--------|------|
| `CheckpointHook` | 4 | 每 epoch 保存最优 + 最新 checkpoint |
| `EarlyStopHook` | 7 | 指标不再改善时停止训练 |
| `TorchCheckpoint` | 9 + 3 契约 | 保存/加载模型 + 优化器 + epoch + 历史 |
| `GridSearch` | 18 | 笛卡尔积搜索、完整运行、选择最优 trial |
| `RandomSearch` | —（合并在 GridSearch） | 无放回随机采样、可复现 |
| `VotingEnsemble` | 9 | 硬/软投票、在数据流上评估 |
| `StackingEnsemble` | 5 | 元模型拟合 + 预测 |
| `SimpleTokenizer` | 12 | 拟合/编码/解码、OOV、最小频率、最大词表 |

---

## 测试套件

```bash
uv sync --dev
uv run pytest                    # 416 通过，5 跳过（~2s）
uv run pytest -m slow            # 15 个契约 + 集成测试
uv run ruff check .              # 格式检查通过
uv run mypy                      # 严格模式
```

**测试层次：**
- **单元测试** (`tests/unit/`) — 逐模块，TDD，仅使用假组件（无框架依赖）
- **契约测试** (`tests/contract/`) — 同一组测试运行在 SklearnModel、NumpyModel、TorchModel 上
- **集成测试** (`tests/integration/`) — 真实适配器完整管线端到端

**未测试 / 有意跳过：**
- `torch` 相关测试在无 torch 环境下跳过（5 个跳过 — 系统 Python 缺 sklearn）
- GPU 特定行为 — 测试默认在 CPU 运行；notebook 在 GPU 可用时使用 GPU
- 依赖网络的操作 — notebook 中的 HuggingFace 下载需要互联网

---

## 局限性

- **仅支持 Python 3.11+。** 使用了 `from __future__ import annotations` 和现代 typing 语法。
- **教学规模，非生产规模。** 大数据集的流式处理、分布式训练、混合精度训练暂时没有包含。
- **PyTorch 已验证；JAX/TensorFlow 未覆盖。** TorchAdapter 覆盖了 PyTorch；Protocol 是框架无关的，但目前只有 PyTorch 有完整适配器实现。
- **LLM 服务是演示级别。** distilgpt2 微调 + FastAPI 服务证明架构端到端可行，但 82M 的模型无法生成有用的代码。更大的模型（Qwen 0.5B、SmolLM2 1.7B）可以同一流程下获得更好质量。

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
