"""Base pipeline orchestration.

Defines the :class:`PipelineState` data bus and the :class:`BasePipeline`
abstract class that orchestrates the six pipeline stages.

The pipeline defines *when* stages run. Concrete implementations define
*how* each stage works by satisfying the protocols in :mod:`pipeline.protocols`.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pipeline.evaluation import Metrics

if TYPE_CHECKING:
    from pipeline.config import Config
    from pipeline.hooks import BaseHook
    from pipeline.protocols import DataStream, LossProtocol, ModelProtocol, OptimizerProtocol

_logger = logging.getLogger(__name__)

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
        val_data_stream: Validation data stream, set by Stage 1.
        features: Set by Stage 2 (extract_features). None if pass-through.
        model: Set by Stage 3 (build_model).
        loss_fn: Set by Stage 3.
        optimizer: Set by Stage 3.
        history: Set by Stage 4 (train). A dict with loss/accuracy curves.
        metrics: Set by Stage 5 (evaluate). A :class:`Metrics` instance.
        predictions: Set by Stage 6 (export). Model predictions on test data.
        current_epoch: Tracked by the training loop; hooks can read it.
        should_stop: Early-stopping hooks set this to True.
    """

    config: Config
    mode: Literal["train", "infer"] = "train"

    # Stage outputs (None until the stage runs)
    data_stream: DataStream | None = None
    val_data_stream: DataStream | None = None
    features: ArrayLike | None = None
    model: ModelProtocol | None = None
    loss_fn: LossProtocol | None = None
    optimizer: OptimizerProtocol | None = None
    history: dict[str, Any] | None = None
    metrics: Metrics = field(default_factory=Metrics)
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
    """Universal deep learning pipeline template.

    Defines *when* six stages run. Subclasses define *how* each stage
    works by implementing the abstract methods. Hooks inject
    cross-cutting concerns (logging, checkpoint, early-stop).

    The six stages are:
        1. :meth:`load_data` -- build :class:`DataStream`
        2. :meth:`extract_features` -- optional feature engineering
        3. :meth:`build_model` -- construct model, loss, optimizer
        4. :meth:`train` -- run the training loop
        5. :meth:`evaluate` -- compute validation metrics
        6. :meth:`export` -- save model and predictions

    Usage::

        class ImagePipeline(BasePipeline):
            def load_data(self, state): ...
            def build_model(self, state): ...
            def train(self, state): ...
            def evaluate(self, state): ...
            def export(self, state): ...

        pipeline = ImagePipeline(config)
        pipeline.add_hook(ProgressHook())
        result = pipeline.run("train")
    """

    def __init__(self, config: Config) -> None:
        """Store config and initialize an empty hook list.

        Args:
            config: Pipeline configuration. Stored as ``self.config``
                for subclasses to read.
        """
        self.config = config
        self._hooks: list[BaseHook] = []

    @property
    def hooks(self) -> list[BaseHook]:
        """Registered hooks (read-only).

        Use :meth:`add_hook` to register. The returned list is the live
        internal list -- modifications to it affect the pipeline.

        TrainLoop reads this property to borrow hooks for training-time
        events (epoch start/end, batch end).
        """
        return self._hooks

    def add_hook(self, hook: BaseHook) -> None:
        """Register a hook to receive pipeline lifecycle events.

        Hooks fire in registration order. Adding the same hook twice
        causes it to fire twice (no deduplication).

        Args:
            hook: Any object implementing :class:`BaseHook`.
        """
        self._hooks.append(hook)

    # ── Template method ────────────────────────────────────────────
    def run(self, mode: Literal["train", "infer"] = "train") -> PipelineState:
        """Execute the pipeline.

        In train mode, runs all six stages. In infer mode, skips
        feature extraction, model building, training, and evaluation
        -- only loads data and exports predictions.

        Args:
            mode: ``"train"`` or ``"infer"``.

        Returns:
            The :class:`PipelineState` with all stage outputs populated.
        """
        state = PipelineState(config=self.config, mode=mode)

        # Stage 1: always needed (both train and infer need data)
        self._run_stage("load_data", state, self.load_data)

        if mode == "train":
            # Stage 2: feature extraction (default: pass-through)
            self._run_stage("extract_features", state, self.extract_features)
            # Stage 3: model construction (DI entry point)
            self._run_stage("build_model", state, self.build_model)
            # Stage 4: training loop
            self._run_stage("train", state, self.train)
            # Stage 5: evaluation on validation set
            self._run_stage("evaluate", state, self.evaluate)

        # Stage 6: export (save model + write predictions)
        self._run_stage("export", state, self.export)

        return state

    # ── Abstract stages ─────────────────────────────────────────────
    @abstractmethod
    def load_data(self, state: PipelineState) -> None:
        """Stage 1: Build a :class:`DataStream` and assign to ``state.data_stream``.

        Called in both train and infer modes.
        """

    def extract_features(self, state: PipelineState) -> None:
        """Stage 2: Optional feature engineering.

        Default is pass-through (no-op). Override in traditional ML
        pipelines (e.g., sklearn) that need explicit feature extraction.
        In deep learning, the model extracts features internally.
        """

    @abstractmethod
    def build_model(self, state: PipelineState) -> None:
        """Stage 3: Construct model, loss, and optimizer.

        Assign ``state.model``, ``state.loss_fn``, and ``state.optimizer``.
        """

    def train(self, state: PipelineState) -> None:
        """Stage 4: Run the training loop.

        Default implementation delegates to :class:`TrainLoop`.
        Subclasses may override for non-standard training (GAN, meta-learning).

        Iterate over ``state.data_stream``, forward -- loss -- backward -- step,
        populating ``state.history`` with loss/accuracy curves.
        """
        # WHY: Lazy import avoids circular imports -- TrainLoop imports
        # PipelineState from pipeline.pipeline, but at this point both
        # modules are already loaded.
        from pipeline.training.train_loop import TrainLoop

        # Assert non-None: build_model() is called before train() in the
        # Template Method sequence; None values indicate a subclass bug.
        assert state.model is not None, "state.model must be set before train()"
        assert state.data_stream is not None, "state.data_stream must be set before train()"
        assert state.optimizer is not None, "state.optimizer must be set before train()"
        assert state.loss_fn is not None, "state.loss_fn must be set before train()"

        loop = TrainLoop(
            model=state.model,
            data_stream=state.data_stream,
            optimizer=state.optimizer,
            loss_fn=state.loss_fn,
            num_epochs=state.config.num_epochs,
            hooks=self._hooks,
        )
        loop.run(state)

    @abstractmethod
    def evaluate(self, state: PipelineState) -> None:
        """Stage 5: Evaluate the trained model on validation data.

        Compute metrics (accuracy, F1, etc.) and write them to
        ``state.metrics``.
        """

    @abstractmethod
    def export(self, state: PipelineState) -> None:
        """Stage 6: Save model checkpoints and write predictions.

        Assign ``state.predictions`` with model outputs on test data.
        """

    # ── Hook dispatch ───────────────────────────────────────────────
    def _run_stage(
        self,
        name: str,
        state: PipelineState,
        stage_fn: Callable[[PipelineState], None],
    ) -> None:
        """Execute one stage, wrapping it with hook notifications.

        Args:
            name: Stage label (e.g., ``"load_data"``).
            state: Shared pipeline state.
            stage_fn: The stage method to execute.
        """
        self._notify("on_stage_start", name, state)
        try:
            stage_fn(state)
        finally:
            self._notify("on_stage_end", name, state)

    def _notify(self, event: str, *args: object) -> None:
        """Dispatch an event to all registered hooks.

        Hook exceptions are caught and logged but do not interrupt
        the pipeline -- hooks are non-critical by design.

        Args:
            event: Hook method name (e.g., ``"on_stage_start"``).
            *args: Arguments forwarded to the hook method.
        """
        for hook in self._hooks:
            try:
                getattr(hook, event)(*args)
            except Exception:
                _logger.exception(
                    "Hook %s.%s raised an exception (ignored)",
                    hook.__class__.__name__,
                    event,
                )
