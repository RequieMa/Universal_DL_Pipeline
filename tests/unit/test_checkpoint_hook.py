"""Unit tests for pipeline.hooks.checkpoint — CheckpointHook."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import torch  # noqa: F401  # availability probe; only torch.nn is imported below

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


@requires_torch
class TestCheckpointHook:
    """Tests for CheckpointHook."""

    def test_saves_on_epoch_end(self, tmp_path: Path) -> None:
        """CheckpointHook saves a checkpoint at on_epoch_end."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.hooks.checkpoint import CheckpointHook
        from pipeline.pipeline import PipelineState

        module = nn.Linear(2, 1)
        state = PipelineState(
            config=Config(output_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"))
        )
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.01)
        state.history = {"loss": [0.5, 0.3]}

        hook = CheckpointHook()
        hook.on_epoch_end(0, state)

        assert (tmp_path / "ckpt" / "latest.pt").exists()
        assert (tmp_path / "ckpt" / "best.pt").exists()  # first epoch always saves best

    def test_best_only_mode(self, tmp_path: Path) -> None:
        """save_best_only=True only saves when metric improves."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.hooks.checkpoint import CheckpointHook
        from pipeline.pipeline import PipelineState

        module = nn.Linear(2, 1)
        state = PipelineState(
            config=Config(output_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"))
        )
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.01)
        state.history = {"loss": [0.5, 0.3]}

        hook = CheckpointHook(save_best_only=True)

        # Epoch 0: loss 0.3 → saves (first epoch)
        hook.on_epoch_end(0, state)
        best_path = tmp_path / "ckpt" / "best.pt"
        assert best_path.exists()
        best_mtime_0 = best_path.stat().st_mtime

        # Epoch 1: loss 0.5 → worse, should NOT overwrite best
        state.history["loss"].append(0.5)
        hook.on_epoch_end(1, state)
        assert best_path.stat().st_mtime == best_mtime_0  # unchanged

        # Epoch 2: loss 0.2 → better, SHOULD overwrite
        state.history["loss"].append(0.2)
        hook.on_epoch_end(2, state)
        assert best_path.stat().st_mtime > best_mtime_0  # updated

    def test_skips_when_model_is_none(self, tmp_path: Path) -> None:
        """on_epoch_end is no-op when state.model is None."""
        from pipeline.config import Config
        from pipeline.hooks.checkpoint import CheckpointHook
        from pipeline.pipeline import PipelineState

        state = PipelineState(config=Config(checkpoint_dir=str(tmp_path)))
        state.model = None

        hook = CheckpointHook()
        hook.on_epoch_end(0, state)  # should not raise

    def test_skips_when_history_is_none(self, tmp_path: Path) -> None:
        """on_epoch_end is no-op when state.history is None."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.hooks.checkpoint import CheckpointHook
        from pipeline.pipeline import PipelineState

        module = nn.Linear(2, 1)
        state = PipelineState(config=Config(output_dir=str(tmp_path)))
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.01)
        state.history = None

        hook = CheckpointHook()
        hook.on_epoch_end(0, state)  # should not raise
