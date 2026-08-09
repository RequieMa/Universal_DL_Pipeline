"""Unit tests for pipeline.hooks.early_stop — EarlyStopHook."""

from __future__ import annotations

import logging

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

    def test_decision_uses_epoch_mean_not_last_batch(self) -> None:
        """The decision is driven by the epoch MEAN, not the last batch.

        Simulates a flat per-batch history appended by TrainLoop:
          - epoch 0 batches [0.0, 1.0] -> mean 0.5, last 1.0
          - epoch 1 batches [1.0, 0.0] -> mean 0.5, last 0.0
        By last-batch logic epoch 1 (0.0) would "improve" on epoch 0 (1.0),
        resetting patience and NOT stopping. By epoch-mean logic both epochs
        are 0.5 -> no improvement -> patience=0 stops on epoch 1.
        """
        state = _state(history={"loss": []})
        hook = EarlyStopHook(patience=0)

        # Epoch 0: two batches, mean 0.5 (baseline).
        state.history["loss"].extend([0.0, 1.0])
        hook.on_epoch_end(0, state)
        assert state.should_stop is False

        # Epoch 1: two batches, mean 0.5 again (last batch 0.0 is a red herring).
        state.history["loss"].extend([1.0, 0.0])
        hook.on_epoch_end(1, state)

        # Mean did not improve -> patience=0 stops. Last-batch logic would not.
        assert state.should_stop is True

    def test_epoch_mean_patience_over_multiple_epochs(self) -> None:
        """Epoch-mean based patience still honours consecutive-epoch semantics."""
        state = _state(history={"loss": []})
        hook = EarlyStopHook(patience=2)

        # Epoch 0: mean 1.0 baseline.
        state.history["loss"].extend([1.0, 1.0])
        hook.on_epoch_end(0, state)
        assert state.should_stop is False

        # Epoch 1: mean 0.5 -> improvement, counter reset.
        state.history["loss"].extend([0.4, 0.6])
        hook.on_epoch_end(1, state)
        assert state.should_stop is False

        # Epoch 2: mean 0.5 -> no improvement, counter 1.
        state.history["loss"].extend([0.6, 0.4])
        hook.on_epoch_end(2, state)
        assert state.should_stop is False

        # Epoch 3: mean 0.5 -> no improvement, counter 2 >= patience -> stop.
        state.history["loss"].extend([0.5, 0.5])
        hook.on_epoch_end(3, state)
        assert state.should_stop is True

    def test_missing_key_warns_once_and_no_stop(self, caplog) -> None:
        """A missing monitored key logs exactly one warning and never stops."""
        state = _state(history={"loss": [0.5, 0.4]})
        hook = EarlyStopHook(patience=0, monitor="val_loss")

        with caplog.at_level(logging.WARNING, logger="pipeline.hooks.early_stop"):
            hook.on_epoch_end(0, state)
            state.history["loss"].append(0.3)
            hook.on_epoch_end(1, state)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "val_loss" in warnings[0].getMessage()
        assert state.should_stop is False
