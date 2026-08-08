"""Unit tests for pipeline.export.checkpoint — TorchCheckpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


@requires_torch
class TestTorchCheckpointSave:
    """Save tests."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """save() writes a .pt file to disk."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.export.checkpoint import TorchCheckpoint
        from pipeline.pipeline import PipelineState

        module = nn.Linear(4, 2)
        state = PipelineState(config=Config(output_dir=str(tmp_path)))
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.1)
        state.current_epoch = 3
        state.history = {"loss": [0.5, 0.3, 0.2]}

        path = tmp_path / "checkpoint.pt"
        TorchCheckpoint.save(state, path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save() creates parent directories if they don't exist."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.export.checkpoint import TorchCheckpoint
        from pipeline.pipeline import PipelineState

        module = nn.Linear(2, 1)
        state = PipelineState(config=Config(output_dir=str(tmp_path)))
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.01)

        deep_path = tmp_path / "sub" / "deep" / "model.pt"
        TorchCheckpoint.save(state, deep_path)
        assert deep_path.exists()

    def test_save_raises_on_non_torch_model(self, tmp_path: Path) -> None:
        """save() raises TypeError when model is not TorchModel."""
        from pipeline.config import Config
        from pipeline.export.checkpoint import TorchCheckpoint
        from pipeline.pipeline import PipelineState

        state = PipelineState(config=Config())
        state.model = "not_a_torch_model"

        with pytest.raises(TypeError, match="TorchModel"):
            TorchCheckpoint.save(state, tmp_path / "bad.pt")


@requires_torch
class TestTorchCheckpointLoad:
    """Load tests."""

    def test_load_restores_weights(self, tmp_path: Path) -> None:
        """load() restores model weights to saved values."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.export.checkpoint import TorchCheckpoint
        from pipeline.pipeline import PipelineState

        # Train a model to change weights from init
        module = nn.Linear(3, 2)
        state = PipelineState(config=Config(output_dir=str(tmp_path)))
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.1)

        # One training step to get non-init weights
        x = torch.randn(4, 3)
        out = module(x)
        loss = out.sum()
        loss.backward()
        state.optimizer.step()

        # Save weights
        saved_weight = module.weight.data.clone()
        path = tmp_path / "saved.pt"
        TorchCheckpoint.save(state, path)

        # Reset module to random init
        module = nn.Linear(3, 2)
        fresh_state = PipelineState(config=Config(output_dir=str(tmp_path)))
        fresh_state.model = TorchModel(module)
        fresh_state.optimizer = TorchOptimizer(fresh_state.model.parameters(), "SGD", lr=0.1)
        TorchCheckpoint.load(fresh_state, path)

        # Weights should match saved
        torch.testing.assert_close(module.weight.data, saved_weight)

    def test_load_restores_epoch(self, tmp_path: Path) -> None:
        """load() restores current_epoch."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.export.checkpoint import TorchCheckpoint
        from pipeline.pipeline import PipelineState

        module = nn.Linear(2, 1)
        state = PipelineState(config=Config(output_dir=str(tmp_path)))
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.01)
        state.current_epoch = 7
        state.history = {"loss": [1.0]}

        path = tmp_path / "epoch.pt"
        TorchCheckpoint.save(state, path)

        fresh = PipelineState(config=Config())
        fresh.model = TorchModel(nn.Linear(2, 1))
        TorchCheckpoint.load(fresh, path)
        assert fresh.current_epoch == 7

    def test_load_restores_optimizer_state(self, tmp_path: Path) -> None:
        """load() restores optimizer state dict."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.export.checkpoint import TorchCheckpoint
        from pipeline.pipeline import PipelineState

        module = nn.Linear(2, 2)
        state = PipelineState(config=Config(output_dir=str(tmp_path)))
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "Adam", lr=0.001)

        # Take a step so optimizer has state
        x = torch.randn(3, 2)
        out = module(x)
        loss = out.sum()
        loss.backward()
        state.optimizer.step()

        path = tmp_path / "opt_state.pt"
        TorchCheckpoint.save(state, path)

        # Fresh model + optimizer
        module2 = nn.Linear(2, 2)
        fresh = PipelineState(config=Config())
        fresh.model = TorchModel(module2)
        fresh.optimizer = TorchOptimizer(fresh.model.parameters(), "Adam", lr=0.001)
        TorchCheckpoint.load(fresh, path)

        # Optimizer should have state (not empty dicts)
        for group in fresh.optimizer._opt.param_groups:
            for p in group["params"]:
                param_state = fresh.optimizer._opt.state[p]
                assert len(param_state) > 0

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """load() raises FileNotFoundError for nonexistent path."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel
        from pipeline.config import Config
        from pipeline.export.checkpoint import TorchCheckpoint
        from pipeline.pipeline import PipelineState

        state = PipelineState(config=Config())
        state.model = TorchModel(nn.Linear(2, 1))

        with pytest.raises(FileNotFoundError):
            TorchCheckpoint.load(state, tmp_path / "nonexistent.pt")

    def test_save_load_round_trip_forward_output(self, tmp_path: Path) -> None:
        """Model produces identical output after save→load round-trip."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.export.checkpoint import TorchCheckpoint
        from pipeline.pipeline import PipelineState

        module = nn.Linear(4, 2)
        state = PipelineState(config=Config(output_dir=str(tmp_path)))
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.1)

        x = torch.randn(3, 4)
        out_before = module(x).clone()

        path = tmp_path / "roundtrip.pt"
        TorchCheckpoint.save(state, path)

        # Load into fresh module
        module2 = nn.Linear(4, 2)
        fresh = PipelineState(config=Config())
        fresh.model = TorchModel(module2)
        TorchCheckpoint.load(fresh, path)

        out_after = module2(x)
        torch.testing.assert_close(out_after, out_before)

    def test_load_without_optimizer_does_not_raise(self, tmp_path: Path) -> None:
        """load() works when state.optimizer is None (infer mode)."""
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.export.checkpoint import TorchCheckpoint
        from pipeline.pipeline import PipelineState

        # Save with optimizer
        module = nn.Linear(3, 1)
        state = PipelineState(config=Config(output_dir=str(tmp_path)))
        state.model = TorchModel(module)
        state.optimizer = TorchOptimizer(state.model.parameters(), "SGD", lr=0.1)
        path = tmp_path / "noopt.pt"
        TorchCheckpoint.save(state, path)

        # Load without optimizer
        fresh = PipelineState(config=Config())
        fresh.model = TorchModel(nn.Linear(3, 1))
        fresh.optimizer = None
        TorchCheckpoint.load(fresh, path)  # should not raise
        assert fresh.optimizer is None  # unchanged
