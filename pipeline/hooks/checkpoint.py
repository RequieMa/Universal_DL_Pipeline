"""Model checkpointing hook.

Provides :class:`CheckpointHook` — a concrete :class:`BaseHook`
that saves model checkpoints at the end of each epoch, optionally
keeping only the best-scoring checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.hooks.base import BaseHook

if TYPE_CHECKING:
    from pipeline.pipeline import PipelineState

_logger = logging.getLogger(__name__)


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
        self._consumed = 0
        self._warned_missing = False

    def on_epoch_end(self, epoch: int, state: PipelineState) -> None:
        """Save checkpoints at the end of an epoch.

        This is a no-op when ``state.model`` is None, ``state.history`` is
        None, or no checkpoint directory can be resolved from the state.

        The monitored metric is the mean over the batches added since the
        previous epoch (``state.history[monitor]`` is a flat, per-batch
        list), so best-tracking compares epoch means rather than the noisy
        final batch. In ``save_best_only`` mode the checkpoint is written
        only when that epoch metric strictly improves.

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

        # Always write the latest checkpoint (unless best-only); this is
        # independent of the monitored metric.
        if not self.save_best_only:
            TorchCheckpoint.save(state, checkpoint_dir / "latest.pt")

        # Best-tracking needs the monitored metric. If it's absent, warn
        # once and skip best-tracking (never crash the loop).
        if self.monitor not in state.history:
            self._warn_missing(state)
            return

        score = self._epoch_metric(state)
        if score is None:
            return

        if self._is_best(score):
            self._best_score = score
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

    def _epoch_metric(self, state: PipelineState) -> float | None:
        """Return the mean of the monitored series over the current epoch.

        ``state.history[monitor]`` is a flat, per-batch list. Each call
        averages only the entries appended since the previous epoch and
        advances an internal cursor. When no new entries were added (e.g.
        the hook is driven over a pre-filled history) it falls back to the
        mean of the whole series, so a one-value-per-epoch metric still maps
        to that single value.

        The caller guarantees ``self.monitor`` is present in
        ``state.history`` before this runs.

        Args:
            state: Current pipeline state.

        Returns:
            The epoch mean, or None when the series is empty or not a list.
        """
        series = state.history[self.monitor] if state.history else None
        if not isinstance(series, list) or not series:
            return None

        new = series[self._consumed :]
        self._consumed = len(series)
        window = new if new else series
        return sum(float(v) for v in window) / len(window)

    def _warn_missing(self, state: PipelineState) -> None:
        """Log a single warning naming the missing monitored key.

        Args:
            state: Current pipeline state.
        """
        if self._warned_missing:
            return
        self._warned_missing = True
        available = sorted(state.history) if state.history else []
        _logger.warning(
            "CheckpointHook: monitored key %r not found in state.history; "
            "available keys: %s. Best-checkpoint tracking is disabled.",
            self.monitor,
            available,
        )

    def _is_best(self, score: float) -> bool:
        """Return True when ``score`` is the best seen so far.

        Args:
            score: The current epoch metric.

        Returns:
            True on the first epoch or when the score improves on the
            stored best score per ``self.mode``.
        """
        if self._best_score is None:
            return True
        if self.mode == "max":
            return score > self._best_score
        return score < self._best_score
