"""Early-stopping hook.

Provides :class:`EarlyStopHook` — a concrete :class:`BaseHook`
that monitors a metric across epochs and sets ``state.should_stop = True``
when the metric stops improving for a given number of epochs
("patience").
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pipeline.hooks.base import BaseHook

if TYPE_CHECKING:
    from pipeline.pipeline import PipelineState

_logger = logging.getLogger(__name__)


class EarlyStopHook(BaseHook):
    """Stop training when a monitored metric stops improving.

    At the end of each epoch this hook compares the epoch's mean of the
    monitored metric to the best value seen so far. When the metric fails
    to improve for ``patience`` consecutive epochs it sets
    ``state.should_stop = True`` so the training loop can halt early.

    ``state.history[monitor]`` is a flat, per-batch list -- the training
    loop appends one value per batch across every epoch. The hook averages
    only the batches added since the previous epoch, so the decision is
    driven by the epoch mean rather than the noisy final batch.

    If the monitored key is missing from ``state.history`` the hook logs a
    single warning and then behaves as a no-op (it never crashes training).

    The first epoch always records a baseline and never stops. Works with
    any framework: it only reads scalar values from ``state.history``, so
    numpy, torch, and sklearn pipelines all behave identically.

    Usage::

        pipeline.add_hook(EarlyStopHook(patience=5, monitor="val_loss", mode="min"))

    Args:
        patience: Number of consecutive non-improving epochs before
            stopping. ``0`` stops at the first non-improving epoch.
        monitor: Key in ``state.history`` to monitor.
        mode: Whether lower (``"min"``) or higher (``"max"``) is better.
        min_delta: Minimum absolute change required to count as
            improvement. Values within this threshold are treated as no
            improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        """Initialize the early-stopping hook.

        Args:
            patience: Consecutive non-improving epochs before stopping.
            monitor: History key to monitor.
            mode: Whether lower (``"min"``) or higher (``"max"``) is better.
            min_delta: Minimum change to count as improvement.
        """
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self._counter = 0
        self._best: float | None = None
        self._consumed = 0
        self._warned_missing = False

    def on_epoch_end(self, epoch: int, state: PipelineState) -> None:
        """Check the monitored metric and decide whether to stop.

        No-op when ``state.history`` is None. If ``self.monitor`` is absent
        a single warning is logged and the hook then no-ops. Epoch 0 records
        a baseline and never stops.

        Args:
            epoch: Zero-based epoch index.
            state: Current pipeline state.
        """
        if state.history is None:
            return

        if self.monitor not in state.history:
            self._warn_missing(state)
            return

        current = self._epoch_metric(state)
        if current is None:
            return

        if self._best is None:
            # First observation: baseline, never stop.
            self._best = current
            self._counter = 0
            return

        assert self._best is not None
        if self._is_improved(current, self._best):
            self._best = current
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                state.should_stop = True

    def _is_improved(self, current: float, best: float) -> bool:
        """Return True when ``current`` is a meaningful improvement.

        Args:
            current: The latest monitored value.
            best: The best value recorded so far.

        Returns:
            True if the value improves on ``best`` by at least
            ``min_delta`` per ``self.mode``.
        """
        if self.mode == "max":
            return current - best > self.min_delta
        return best - current > self.min_delta

    def _epoch_metric(self, state: PipelineState) -> float | None:
        """Return the mean of the monitored series over the current epoch.

        ``state.history[monitor]`` is a flat, per-batch list. Each call
        averages only the entries appended since the previous epoch and
        advances an internal cursor. When no new entries were added (e.g.
        the hook is driven over a pre-filled history) it falls back to the
        mean of the whole series, so a one-value-per-epoch metric still maps
        to that single value.

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
            "EarlyStopHook: monitored key %r not found in state.history; "
            "available keys: %s. Early stopping is disabled.",
            self.monitor,
            available,
        )
