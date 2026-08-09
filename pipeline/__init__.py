"""Universal DL Pipeline — a teaching-first, framework-agnostic deep learning pipeline.

The pipeline defines *when* stages run via :class:`BasePipeline`.
Concrete implementations define *how* by satisfying the protocols in :mod:`pipeline.protocols`.

Quick start::

    from pipeline import BasePipeline, PipelineState, Config
    from pipeline.protocols import ModelProtocol, DataStream, Batch
    from pipeline.registry import register, build

Subpackages:
    - :mod:`pipeline.protocols` — abstract interfaces (framework-agnostic)
    - :mod:`pipeline.pipeline` — BasePipeline ABC + PipelineState
    - :mod:`pipeline.registry` — dependency injection registry
    - :mod:`pipeline.hooks` — hook system for cross-cutting concerns
    - :mod:`pipeline.config` — configuration dataclass
    - :mod:`pipeline.data` — data sources (CsvDataSource, ImageFolderDataSource,
      TransformedDataStream) and split utilities
    - :mod:`pipeline.training` — train loop, optimizers (SGD, Adam), losses (MSE, CrossEntropy)
    - :mod:`pipeline.evaluation` — metrics container and pure metric functions
    - :mod:`pipeline.export` — CSV and checkpoint export utilities
    - :mod:`pipeline.adapters` — framework adapters (SklearnModel, NumpyModel,
      NumpyOptimizer, TorchModel, TorchLoss, TorchOptimizer)
    - :mod:`pipeline.utils` — device, seed utilities
"""

from pipeline.config import Config
from pipeline.pipeline import BasePipeline, PipelineState

__all__ = ["BasePipeline", "PipelineState", "Config"]
__version__ = "0.2.0"
