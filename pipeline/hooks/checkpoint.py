"""Model checkpointing hook.

Provides :class:`CheckpointHook` — a concrete :class:`BaseHook`
that saves model checkpoints at the end of each epoch, optionally
keeping only the best-scoring checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.hooks.base import BaseHook

if TYPE_CHECKING:
    from pipeline.pipeline import PipelineState


class CheckpointHook(BaseHook):
    """Save model checkpoints at epoch boundaries.

    On each epoch end this hook writes a checkpoint to
    ``{checkpoint_dir}/latest.pt``. When ``save_best_only`` is disabled
    (default) it also tracks the monitored metric and keeps a
    ``{checkpoint_dir}/best.pt`` that only ever holds the best score seen
    so far. When ``save_best_only`` is enabled only the best checkpoint is
    written.

    Checkpoint serialization is delegated to ``TorchCheckpoint``, so torch
    is imported lazily. If ``state.checkpoint_dir`` is not set the hook
    falls back to ``{output_dir}/checkpoints``.

    Usage::

        pipeline.add_hook(CheckpointHook(save_best_only=True, monitor="loss", mode="min"))
    """

    def __init__(
        self,
        save_best_only: bool = False,
        monitor: str = "loss",
        mode: str = "min",
    ) -> None:
        """Initialize the checkpoint hook.

        Args:
            save_best_only: If True, only write ``best.pt`` (skip ``latest.pt``).
            monitor: History key to monitor for best-score tracking.
            mode: Whether lower (``"min"``) or higher (``"max"``) is better.
        """
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self._best_score: float | None = None

    def on_epoch_end(self, epoch: int, state: PipelineState) -> None:
        """Save checkpoints at the end of an epoch.

        This is a no-op when ``state.model`` is None, ``state.history`` is
        None, or no checkpoint directory can be resolved from the state.

        Args:
            epoch: Zero-based epoch index.
            state: Current pipeline state.
        """
        if state.model is None or state.history is None:
            return

        checkpoint_dir = self._resolve_checkpoint_dir(state)
        if checkpoint_dir is None:
            return

        from pipeline.export.checkpoint import TorchCheckpoint

        # Always write the latest checkpoint (unless best-only).
        if not self.save_best_only:
            TorchCheckpoint.save(state, checkpoint_dir / "latest.pt")

        if self._is_best(state):
            self._best_score = self._current_score(state)
            TorchCheckpoint.save(state, checkpoint_dir / "best.pt")

    def _resolve_checkpoint_dir(self, state: PipelineState) -> Path | None:
        """Resolve the checkpoint directory from the pipeline state.

        Args:
            state: Current pipeline state.

        Returns:
            The checkpoint directory, or None if it cannot be determined.
        """
        cfg = state.config
        if cfg is None:
            return None
        if cfg.checkpoint_dir:
            return Path(cfg.checkpoint_dir)
        if cfg.output_dir:
            return Path(cfg.output_dir) / "checkpoints"
        return None

    def _current_score(self, state: PipelineState) -> float:
        """Read the current monitored score from history.

        Args:
            state: Current pipeline state.

        Returns:
            The latest value of ``state.history[self.monitor]``.
        """
        metrics = state.history[self.monitor] if state.history else None
        if metrics is None:
            return float("inf")
        if isinstance(metrics, list):
            return float(metrics[-1]) if metrics else float("inf")
        return float(metrics)

    def _is_best(self, state: PipelineState) -> bool:
        """Return True when the current score is the best seen so far.

        Args:
            state: Current pipeline state.

        Returns:
            True on the first epoch or when the current score improves on
            the stored best score per ``self.mode``.
        """
        if self._best_score is None:
            return True
        score = self._current_score(state)
        if self.mode == "max":
            return score > self._best_score
        return score < self._best_score
