"""Early-stopping hook.

Provides :class:`EarlyStopHook` — a concrete :class:`BaseHook`
that monitors a metric across epochs and sets ``state.should_stop = True``
when the metric stops improving for a given number of epochs
("patience").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.hooks.base import BaseHook

if TYPE_CHECKING:
    from pipeline.pipeline import PipelineState


class EarlyStopHook(BaseHook):
    """Stop training when a monitored metric stops improving.

    At the end of each epoch this hook reads the latest value of a key in
    ``state.history`` and compares it to the best value seen so far. When
    the metric fails to improve for ``patience`` consecutive epochs it
    sets ``state.should_stop = True`` so the training loop can halt early.

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

    def on_epoch_end(self, epoch: int, state: PipelineState) -> None:
        """Check the monitored metric and decide whether to stop.

        No-op when ``state.history`` is None or ``self.monitor`` is not a
        key in it. Epoch 0 records a baseline and never stops.

        Args:
            epoch: Zero-based epoch index.
            state: Current pipeline state.
        """
        if state.history is None or self.monitor not in state.history:
            return

        series = state.history[self.monitor]
        if not isinstance(series, list) or not series:
            return

        current = float(series[-1])

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
