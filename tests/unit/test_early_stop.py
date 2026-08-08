"""Unit tests for pipeline.hooks.early_stop — EarlyStopHook."""

from __future__ import annotations

from pipeline.config import Config
from pipeline.hooks.early_stop import EarlyStopHook
from pipeline.pipeline import PipelineState


def _state(history: dict | None = None) -> PipelineState:
    """Build a minimal PipelineState with the given history (or None)."""
    return PipelineState(config=Config(), history=history)


class TestEarlyStop:
    """Tests for EarlyStopHook."""

    def test_stops_after_patience_exhausted(self) -> None:
        """3 epochs without improvement sets should_stop=True."""
        state = _state(history={"loss": [1.0, 1.0, 1.0, 1.0]})
        hook = EarlyStopHook(patience=3)

        hook.on_epoch_end(0, state)
        hook.on_epoch_end(1, state)
        hook.on_epoch_end(2, state)
        hook.on_epoch_end(3, state)

        assert state.should_stop is True

    def test_resets_on_improvement(self) -> None:
        """An improvement resets the patience counter."""
        state = _state(history={"loss": [1.0, 1.0, 0.5, 0.5, 0.5, 0.5]})
        hook = EarlyStopHook(patience=3)

        hook.on_epoch_end(0, state)  # baseline 1.0
        hook.on_epoch_end(1, state)  # no improvement (1.0) -> counter 1
        hook.on_epoch_end(2, state)  # improvement (0.5) -> reset to 0
        hook.on_epoch_end(3, state)  # no improvement -> counter 1
        hook.on_epoch_end(4, state)  # no improvement -> counter 2
        hook.on_epoch_end(5, state)  # no improvement -> counter 3 -> stop

        assert state.should_stop is True

    def test_skips_when_no_history(self) -> None:
        """When state.history is None the hook is a no-op."""
        state = _state(history=None)
        hook = EarlyStopHook(patience=3)

        for epoch in range(5):
            hook.on_epoch_end(epoch, state)

        assert state.should_stop is False

    def test_mode_max(self) -> None:
        """mode='max' tracks higher-is-better metrics."""
        state = _state(history={"accuracy": [0.6, 0.6, 0.6, 0.6]})
        hook = EarlyStopHook(patience=3, monitor="accuracy", mode="max")

        hook.on_epoch_end(0, state)
        hook.on_epoch_end(1, state)
        hook.on_epoch_end(2, state)
        hook.on_epoch_end(3, state)

        assert state.should_stop is True

    def test_min_delta(self) -> None:
        """Small changes below min_delta don't count as improvement."""
        state = _state(history={"loss": [1.0, 0.99, 0.98, 0.97]})
        hook = EarlyStopHook(patience=2, min_delta=0.05)

        hook.on_epoch_end(0, state)  # baseline 1.0
        hook.on_epoch_end(1, state)  # 0.99 -> delta 0.01 < 0.05, no improve
        hook.on_epoch_end(2, state)  # 0.98 -> no improve -> counter 2 -> stop

        assert state.should_stop is True

    def test_patience_zero_stops_immediately(self) -> None:
        """patience=0 stops on the first non-improvement."""
        state = _state(history={"loss": [1.0, 1.0, 1.0]})
        hook = EarlyStopHook(patience=0)

        hook.on_epoch_end(0, state)  # baseline
        hook.on_epoch_end(1, state)  # no improvement -> counter 0 >= 0 -> stop

        assert state.should_stop is True

    def test_first_epoch_never_stops(self) -> None:
        """Epoch 0 records a baseline and never stops."""
        state = _state(history={"loss": [0.5]})
        hook = EarlyStopHook(patience=0)

        hook.on_epoch_end(0, state)

        assert state.should_stop is False
