# Universal DL Pipeline

A **teaching-first, framework-agnostic deep learning pipeline library**.

The pipeline defines *when* things happen. Concrete implementations define *how*.
Protocols (not PyTorch classes) are the contract.

## Quick Start

```python
from pipeline import BasePipeline, PipelineState, Config
from pipeline.protocols import ModelProtocol, DataStream, Batch
from pipeline.registry import register, build
from pipeline.evaluation.metrics import Metrics, accuracy, f1_score

# Build a pipeline
class MyPipeline(BasePipeline):
    def load_data(self, state): ...
    def build_model(self, state): ...
    def evaluate(self, state): ...
    def export(self, state): ...

pipeline = MyPipeline(Config(batch_size=64, num_epochs=10))
state = pipeline.run("train")
print(state.metrics["accuracy"])
```

## Design Principles

- **Teach, don't compete.** Code is readable first, functional second.
- **Framework-agnostic core.** Protocols define the contract, not PyTorch classes.
- **Pipeline is not a black box.** Every stage can be inspected, replaced, or skipped.
- **Spiral pedagogy.** Same pipeline structure, progressively richer implementations:
  sklearn → numpy → PyTorch → custom.

## Package Structure

| Package | Purpose |
|---------|---------|
| `pipeline.protocols` | Abstract interfaces (ModelProtocol, DataStream, Batch, ...) |
| `pipeline.pipeline` | BasePipeline ABC + PipelineState data bus |
| `pipeline.registry` | Dependency injection: `register()` + `build()` |
| `pipeline.hooks` | Hook system: BaseHook + ProgressHook |
| `pipeline.config` | Configuration dataclass with YAML loading |
| `pipeline.data` | Data sources (CsvDataSource) and train/test split |
| `pipeline.training` | TrainLoop with training-time hook dispatch |
| `pipeline.evaluation` | Metrics container + accuracy, precision, recall, F1 |
| `pipeline.export` | Export utilities (to_csv) |
| `pipeline.utils` | Device detection and random seed utilities |
