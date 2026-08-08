"""Unit tests for pipeline.adapters.torch_adapter — TorchModel, TorchLoss, TorchOptimizer."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


# ═══════════════════════════════════════════════════════════════════════════
# TorchModel tests
# ═══════════════════════════════════════════════════════════════════════════

@requires_torch
class TestTorchModelConstruction:
    """Construction and basic inspection."""

    def test_wraps_nn_module(self) -> None:
        """TorchModel wraps an nn.Module without error."""
        from pipeline.adapters.torch_adapter import TorchModel

        model = TorchModel(torch.nn.Linear(4, 2))
        assert model._module is not None

    def test_parameters_returns_parameters(self) -> None:
        """parameters() yields Parameter objects matching module params."""
        from pipeline.adapters.torch_adapter import TorchModel
        from pipeline.protocols import Parameter

        module = torch.nn.Linear(4, 2)
        model = TorchModel(module)
        params = list(model.parameters())
        assert len(params) == 2  # weight + bias
        assert all(isinstance(p, Parameter) for p in params)
        assert params[0].name == "weight"
        assert params[1].name == "bias"

    def test_parameter_data_is_torch_tensor(self) -> None:
        """Parameter.data holds the underlying torch tensor."""
        from pipeline.adapters.torch_adapter import TorchModel

        module = torch.nn.Linear(4, 2)
        model = TorchModel(module)
        for p in model.parameters():
            assert isinstance(p.data, torch.Tensor)

    def test_parameter_grad_is_tensor_or_none(self) -> None:
        """Parameter.grad is torch.Tensor or None (before backward)."""
        from pipeline.adapters.torch_adapter import TorchModel

        module = torch.nn.Linear(4, 2)
        model = TorchModel(module)
        for p in model.parameters():
            # Before backward, grad may be None
            assert p.grad is None or isinstance(p.grad, torch.Tensor)

    def test_empty_module_returns_no_parameters(self) -> None:
        """Module with no nn.Parameter yields no Parameter objects."""
        from pipeline.adapters.torch_adapter import TorchModel

        module = torch.nn.Sequential()  # no params
        model = TorchModel(module)
        assert list(model.parameters()) == []


@requires_torch
class TestTorchModelForward:
    """Forward pass tests."""

    def test_forward_tensor_input_returns_tensor(self) -> None:
        """torch.Tensor input → torch.Tensor output (zero-copy path)."""
        from pipeline.adapters.torch_adapter import TorchModel

        module = torch.nn.Linear(4, 2)
        model = TorchModel(module)
        x = torch.randn(3, 4)
        out = model.forward(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 2)

    def test_forward_numpy_input_returns_numpy(self) -> None:
        """np.ndarray input → np.ndarray output."""
        from pipeline.adapters.torch_adapter import TorchModel

        module = torch.nn.Linear(4, 2)
        model = TorchModel(module)
        x = np.random.randn(3, 4).astype(np.float32)
        out = model.forward(x)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3, 2)

    def test_forward_preserves_batch_size(self) -> None:
        """Output batch size matches input batch size."""
        from pipeline.adapters.torch_adapter import TorchModel

        module = torch.nn.Linear(10, 5)
        model = TorchModel(module)
        x = torch.randn(7, 10)
        out = model.forward(x)
        assert out.shape[0] == 7
        assert out.shape[1] == 5

    def test_forward_tensor_same_values_as_module_direct(self) -> None:
        """TorchModel.forward(tensor) == module.forward(tensor)."""
        from pipeline.adapters.torch_adapter import TorchModel

        module = torch.nn.Linear(4, 2)
        # Freeze weights for deterministic test
        with torch.no_grad():
            module.weight.fill_(0.5)
            module.bias.fill_(0.1)

        model = TorchModel(module)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        via_model = model.forward(x)
        via_module = module(x)
        torch.testing.assert_close(via_model, via_module)

    def test_forward_numpy_same_values_as_module(self) -> None:
        """TorchModel.forward(numpy) matches module.forward(tensor) converted back."""
        from pipeline.adapters.torch_adapter import TorchModel

        module = torch.nn.Linear(3, 1)
        with torch.no_grad():
            module.weight.fill_(0.5)
            module.bias.fill_(0.1)

        model = TorchModel(module)
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x_t = torch.from_numpy(x_np)
        via_model = model.forward(x_np)
        via_module = module(x_t).detach().cpu().numpy()
        np.testing.assert_array_almost_equal(via_model, via_module)

    def test_forward_conv2d_module(self) -> None:
        """TorchModel works with Conv2d modules (multi-dim tensors)."""
        from pipeline.adapters.torch_adapter import TorchModel

        conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        model = TorchModel(conv)
        x = torch.randn(2, 3, 32, 32)  # (N, C, H, W)
        out = model.forward(x)
        assert out.shape == (2, 16, 32, 32)


@requires_torch
class TestTorchModelMode:
    """Train/eval mode tests."""

    def test_train_mode_sets_module_training(self) -> None:
        """train_mode() sets module to training state."""
        from pipeline.adapters.torch_adapter import TorchModel

        module = torch.nn.Linear(4, 2)
        model = TorchModel(module)
        model.train_mode()
        assert module.training is True

    def test_eval_mode_sets_module_eval(self) -> None:
        """eval_mode() sets module to eval state."""
        from pipeline.adapters.torch_adapter import TorchModel

        module = torch.nn.Linear(4, 2)
        model = TorchModel(module)
        model.eval_mode()
        assert module.training is False

    def test_train_eval_cycle(self) -> None:
        """train → eval → train toggles correctly."""
        from pipeline.adapters.torch_adapter import TorchModel

        module = torch.nn.Linear(4, 2)
        model = TorchModel(module)
        model.train_mode()
        assert module.training
        model.eval_mode()
        assert not module.training
        model.train_mode()
        assert module.training

    def test_batchnorm_behavior_differs_in_train_vs_eval(self) -> None:
        """train_mode affects BatchNorm layers (dropout/batchnorm)."""
        from pipeline.adapters.torch_adapter import TorchModel

        bn = torch.nn.BatchNorm1d(4)
        model = TorchModel(bn)
        x = torch.randn(8, 4)

        model.train_mode()
        out_train = model.forward(x)

        model.eval_mode()
        out_eval = model.forward(x)

        # BatchNorm uses population stats in eval, batch stats in train
        # Output should differ (unless weights happen to be identical)
        assert not torch.allclose(out_train, out_eval)
