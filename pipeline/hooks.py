"""Hook system for cross-cutting concerns in the pipeline.

Hooks observe pipeline lifecycle events without modifying the pipeline
stages themselves. Students add one hook at a time to layer on logging,
checkpointing, and early stopping.

Five hook points:
    - :meth:`on_stage_start` — before a stage executes
    - :meth:`on_stage_end` — after a stage completes
    - :meth:`on_epoch_start` — before each training epoch
    - :meth:`on_epoch_end` — after each training epoch
    - :meth:`on_batch_end` — after each training batch

Usage::

    class MyLogger(BaseHook):
        def on_epoch_end(self, epoch, state):
            print(f"Epoch {epoch}: loss={state.history['loss'][-1]:.4f}")

    pipeline.add_hook(MyLogger())
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.pipeline import PipelineState


class BaseHook:
    """Base class for pipeline hooks. All five methods are no-ops by default.

    Subclasses override only the events they care about. The pipeline
    calls hooks in registration order. Exceptions in hooks are caught
    and logged — they never interrupt the pipeline.

    NOTE: This is NOT an ABC. All methods have default no-op
    implementations so students can write a hook with a single method.
    """

    def on_stage_start(self, stage: str, state: PipelineState) -> None:
        """Called immediately before a stage begins execution.

        Args:
            stage: Stage name — ``"load_data"``, ``"build_model"``,
                ``"train"``, ``"evaluate"``, or ``"export"``.
            state: Current pipeline state (read/write).
        """

    def on_stage_end(self, stage: str, state: PipelineState) -> None:
        """Called immediately after a stage completes (even if it raised).

        Args:
            stage: Stage name.
            state: Current pipeline state.
        """

    def on_epoch_start(self, epoch: int, state: PipelineState) -> None:
        """Called before each training epoch.

        Only fires during the train stage.

        Args:
            epoch: Zero-based epoch index.
            state: Current pipeline state (``state.current_epoch`` is set).
        """

    def on_epoch_end(self, epoch: int, state: PipelineState) -> None:
        """Called after each training epoch.

        Use this to check validation metrics and set
        ``state.should_stop = True`` for early stopping.

        Args:
            epoch: Zero-based epoch index.
            state: Current pipeline state.
        """

    def on_batch_end(
        self, batch: int, loss: float, state: PipelineState
    ) -> None:
        """Called after each training batch.

        Use for progress bars and per-batch logging.

        Args:
            batch: Zero-based batch index within the epoch.
            loss: Scalar loss value for this batch.
            state: Current pipeline state.
        """
