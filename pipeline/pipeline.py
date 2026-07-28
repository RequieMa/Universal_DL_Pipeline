"""Base pipeline orchestration.

Defines the :class:`PipelineState` data bus and the :class:`BasePipeline`
abstract class that orchestrates the six pipeline stages.

The pipeline defines *when* stages run. Concrete implementations define
*how* each stage works by satisfying the protocols in :mod:`pipeline.protocols`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from pipeline.config import Config
    from pipeline.protocols import DataStream, LossProtocol, ModelProtocol, OptimizerProtocol

ArrayLike = np.ndarray | Any


@dataclass
class PipelineState:
    """Shared state bus passed through all pipeline stages.

    Each stage reads from and writes to specific fields. Students
    can inspect ``state`` after each stage to understand what happened.

    Attributes:
        config: The pipeline configuration (read-only after construction).
        mode: ``"train"`` runs all 6 stages; ``"infer"`` skips training.
        data_stream: Set by Stage 1 (load_data).
        features: Set by Stage 2 (extract_features). None if pass-through.
        model: Set by Stage 3 (build_model).
        loss_fn: Set by Stage 3.
        optimizer: Set by Stage 3.
        history: Set by Stage 4 (train). A dict with loss/accuracy curves.
        metrics: Set by Stage 5 (evaluate). e.g., ``{"accuracy": 0.92}``.
        predictions: Set by Stage 6 (export). Model predictions on test data.
        current_epoch: Tracked by the training loop; hooks can read it.
        should_stop: Early-stopping hooks set this to True.
    """

    config: Config
    mode: Literal["train", "infer"] = "train"

    # Stage outputs (None until the stage runs)
    data_stream: DataStream | None = None
    features: ArrayLike | None = None
    model: ModelProtocol | None = None
    loss_fn: LossProtocol | None = None
    optimizer: OptimizerProtocol | None = None
    history: dict[str, Any] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: ArrayLike | None = None

    # Control signals
    current_epoch: int = 0
    should_stop: bool = False

    # -- Derived properties --------------------------------------------------

    @property
    def is_training(self) -> bool:
        """True if the pipeline is in training mode.

        Derived from :attr:`mode`. Safe to query in any stage or hook.
        """
        return self.mode == "train"


class BasePipeline(ABC):
    """Abstract base for concrete pipeline implementations.

    Subclasses implement each of the six stages. The public
    entry points are :meth:`run` and :meth:`infer`.
    """

    @abstractmethod
    def run(self, config: Config) -> PipelineState:
        """Execute the full training pipeline."""
        ...

    @abstractmethod
    def infer(self, state: PipelineState) -> PipelineState:
        """Execute the inference pipeline."""
        ...
