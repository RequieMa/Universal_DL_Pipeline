"""Unit tests for pipeline.hooks.progress — ProgressHook."""

from __future__ import annotations

from unittest.mock import patch

from pipeline.config import Config
from pipeline.hooks.progress import ProgressHook
from pipeline.pipeline import PipelineState


class TestProgressHook:
    """Tests for ProgressHook."""

    def test_creates_pbar_on_epoch_start(self):
        """Happy Path: on_epoch_start creates a tqdm progress bar."""
        hook = ProgressHook()
        state = PipelineState(config=Config(batch_size=2))
        state.data_stream = [1, 2, 3]  # len=3 for total

        with patch("tqdm.tqdm") as mock_tqdm:
            hook.on_epoch_start(0, state)
            mock_tqdm.assert_called_once_with(total=3, desc="Epoch 1")

    def test_updates_pbar_on_batch_end(self):
        """Happy Path: on_batch_end updates the tqdm bar."""
        hook = ProgressHook()
        state = PipelineState(config=Config())
        state.data_stream = [1, 2]

        with patch("tqdm.tqdm") as mock_tqdm:
            mock_pbar = mock_tqdm.return_value
            hook.on_epoch_start(0, state)
            hook.on_batch_end(0, 1.5, state)
            mock_pbar.update.assert_called_once_with(1)
            mock_pbar.set_postfix.assert_called_once_with(loss="1.5000")

    def test_closes_pbar_on_epoch_end(self):
        """Happy Path: on_epoch_end closes the tqdm bar."""
        hook = ProgressHook()
        state = PipelineState(config=Config())

        with patch("tqdm.tqdm") as mock_tqdm:
            mock_pbar = mock_tqdm.return_value
            hook.on_epoch_start(0, state)
            hook.on_batch_end(0, 0.5, state)
            hook.on_batch_end(1, 0.3, state)
            hook.on_epoch_end(0, state)
            mock_pbar.close.assert_called_once()

    def test_clears_losses_after_epoch(self):
        """Happy Path: loss accumulator is cleared after each epoch."""
        hook = ProgressHook()
        state = PipelineState(config=Config())
        state.data_stream = [1]

        with patch("tqdm.tqdm"):
            hook.on_epoch_start(0, state)
            hook.on_batch_end(0, 0.5, state)
            assert len(hook._losses) == 1
            hook.on_epoch_end(0, state)
            assert len(hook._losses) == 0

    def test_stage_events_are_noops(self):
        """Happy Path: on_stage_start/end are inherited no-ops."""
        hook = ProgressHook()
        state = PipelineState(config=Config())
        hook.on_stage_start("train", state)
        hook.on_stage_end("train", state)
