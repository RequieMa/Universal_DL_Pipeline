"""Contract tests: TorchCheckpoint save/load round-trip.

Verifies that :class:`TorchCheckpoint` transparently serializes and
restores a PyTorch model — identical forward output, identical model
parameters, and a fully restored optimizer (LR + internal state) after
a training step. These are the checkpoint contract guarantees any
framework implementation must honor, exercised here on torch.

Gated behind ``requires_torch`` and marked ``slow``, matching the
pattern used across the torch adapter contract tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _build_state(tmp_path: Path, *, lr: float = 0.1, name: str = "SGD"):
    """Build a PipelineState wired to a fresh torch Linear model + optimizer.

    Args:
        tmp_path: Directory for the Config output_dir.
        lr: Learning rate for the optimizer.
        name: torch.optim class name.

    Returns:
        The wired ``(state, module)`` tuple — ``module`` is the raw
        ``torch.nn.Linear`` so tests can step it directly.
    """
    import torch.nn as nn

    from pipeline.adapters.torch_adapter import TorchModel, TorchOptimizer
    from pipeline.config import Config
    from pipeline.pipeline import PipelineState

    module = nn.Linear(4, 2)
    state = PipelineState(config=Config(output_dir=str(tmp_path)))
    state.model = TorchModel(module)
    state.optimizer = TorchOptimizer(state.model.parameters(), name, lr=lr)
    return state, module


@requires_torch
@pytest.mark.slow
class TestCheckpointContract:
    """TorchCheckpoint save/load must preserve the trained model."""

    def test_checkpoint_roundtrip_preserves_forward_output(self, tmp_path: Path) -> None:
        """Model produces identical output after save→load round-trip."""
        from pipeline.export.checkpoint import TorchCheckpoint

        state, module = _build_state(tmp_path)

        # Take a training step so weights differ from init, then capture the
        # trained model's output — the contract is that a loaded model
        # reproduces the *saved* (trained) model's output.
        x = torch.randn(3, 4)
        module(x).sum().backward()
        state.optimizer.step()
        out_before = module(x).clone()

        path = tmp_path / "roundtrip.pt"
        TorchCheckpoint.save(state, path)

        # Load into a fresh module and compare forward output.
        fresh_state, module2 = _build_state(tmp_path)
        TorchCheckpoint.load(fresh_state, path)
        out_after = module2(x)
        torch.testing.assert_close(out_after, out_before)

    def test_checkpoint_parameters_equal_after_load(self, tmp_path: Path) -> None:
        """Weights and biases match the saved values after load."""
        from pipeline.export.checkpoint import TorchCheckpoint

        state, module = _build_state(tmp_path)

        # Step once so the saved parameters are non-init.
        x = torch.randn(3, 4)
        out = module(x)
        out.sum().backward()
        state.optimizer.step()

        saved_weight = module.weight.data.clone()
        saved_bias = module.bias.data.clone()

        path = tmp_path / "params.pt"
        TorchCheckpoint.save(state, path)

        # Fresh module must pick up the saved parameters exactly.
        fresh_state, module2 = _build_state(tmp_path)
        TorchCheckpoint.load(fresh_state, path)

        torch.testing.assert_close(module2.weight.data, saved_weight)
        torch.testing.assert_close(module2.bias.data, saved_bias)

    def test_checkpoint_optimizer_state_restored(self, tmp_path: Path) -> None:
        """Optimizer LR and internal state match after save→load.

        Uses Adam so the step accumulates non-trivial internal state
        (first/second moment estimates), then verifies a fresh optimizer
        loaded from the checkpoint reproduces the saved optimizer's full
        state dict — covering both learning rate and per-parameter state.
        """
        from pipeline.export.checkpoint import TorchCheckpoint

        state, module = _build_state(tmp_path, lr=0.001, name="Adam")

        # Step a few times so Adam builds up moments and a realized LR.
        rng = torch.Generator().manual_seed(0)
        for _ in range(3):
            x = torch.randn(4, 4, generator=rng)
            module(x).sum().backward()
            state.optimizer.step()

        saved_optimizer_dict = state.optimizer._opt.state_dict()

        path = tmp_path / "opt.pt"
        TorchCheckpoint.save(state, path)

        # Fresh model + optimizer must inherit the saved optimizer state.
        fresh_state, _module2 = _build_state(tmp_path, lr=0.001, name="Adam")
        TorchCheckpoint.load(fresh_state, path)

        restored_dict = fresh_state.optimizer._opt.state_dict()

        # Same learning rate and hyperparameters.
        assert restored_dict["param_groups"] == saved_optimizer_dict["param_groups"]
        # Same per-parameter moments / state tensors.
        assert restored_dict["state"].keys() == saved_optimizer_dict["state"].keys()
        for key in saved_optimizer_dict["state"]:
            saved_vals = saved_optimizer_dict["state"][key]
            restored_vals = restored_dict["state"][key]
            assert saved_vals.keys() == restored_vals.keys(), f"mismatched keys for param {key}"
            for state_key in saved_vals:
                torch.testing.assert_close(
                    restored_vals[state_key], saved_vals[state_key], msg=f"param {key} {state_key}"
                )
