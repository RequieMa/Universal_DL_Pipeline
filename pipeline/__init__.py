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
    - :mod:`pipeline.utils` — device, seed utilities
"""

from pipeline.config import Config
from pipeline.pipeline import BasePipeline, PipelineState

__all__ = ["BasePipeline", "PipelineState", "Config"]
__version__ = "0.1.0"
