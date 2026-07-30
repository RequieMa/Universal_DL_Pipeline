"""Unit tests for pipeline.adapters.numpy_adapter — NumpyModel, NumpyOptimizer."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.adapters.numpy_adapter import NumpyModel, NumpyOptimizer
from pipeline.protocols import Parameter


class TestNumpyModelConstruction:
    """Construction and basic properties."""

    def test_single_layer_construction(self) -> None:
        """NumpyModel with one (weight, bias) pair."""
        W = np.array([[0.5, 0.5]])
        b = np.array([0.0])
        model = NumpyModel([(W, b)])
        params = list(model.parameters())
        assert len(params) == 2  # W and b
        assert params[0].name == "W0"
        assert params[1].name == "b0"

    def test_two_layer_construction(self) -> None:
        """NumpyModel with two (weight, bias) pairs."""
        W1 = np.array([[0.5, 0.5], [0.3, 0.3]])
        b1 = np.array([0.0, 0.0])
        W2 = np.array([[1.0, 1.0]])
        b2 = np.array([0.0])
        model = NumpyModel([(W1, b1), (W2, b2)])
        params = list(model.parameters())
        assert len(params) == 4  # W1, b1, W2, b2

    def test_grad_initialized_to_zeros(self) -> None:
        """Each Parameter.grad starts as zeros matching data shape."""
        W = np.array([[0.5, 0.5, 0.5]])
        b = np.array([0.0])
        model = NumpyModel([(W, b)])
        for p in model.parameters():
            assert p.grad is not None
            assert p.grad.shape == p.data.shape
            np.testing.assert_array_equal(p.grad, np.zeros_like(p.data))

    def test_empty_layers_raises(self) -> None:
        """Empty layer list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            NumpyModel([])


class TestNumpyModelForward:
    """Forward pass tests."""

    def test_single_layer_forward_shape(self) -> None:
        """Single layer: (batch, in) -> (batch, out)."""
        W = np.array([[1.0, 0.0]])
        b = np.array([0.0])
        model = NumpyModel([(W, b)])
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = model.forward(x)
        assert out.shape == (2, 1)

    def test_two_layer_forward_shape(self) -> None:
        """Two layers: (batch, in) -> (batch, hidden) -> (batch, out)."""
        W1 = np.ones((4, 3)) * 0.1
        b1 = np.zeros(4)
        W2 = np.ones((2, 4)) * 0.1
        b2 = np.zeros(2)
        model = NumpyModel([(W1, b1), (W2, b2)])
        x = np.array([[1.0, 2.0, 3.0]])
        out = model.forward(x)
        assert out.shape == (1, 2)

    def test_forward_is_deterministic(self) -> None:
        """Same input → same output (no randomness in forward)."""
        rng = np.random.default_rng(42)
        W = rng.standard_normal((3, 4))
        b = rng.standard_normal(3)
        model = NumpyModel([(W, b)])
        x = rng.standard_normal((10, 4))
        out1 = model.forward(x)
        out2 = model.forward(x)
        np.testing.assert_array_equal(out1, out2)

    def test_last_layer_no_activation(self) -> None:
        """Last layer outputs raw logits (no activation applied)."""
        W1 = np.eye(3)
        b1 = np.ones(3)
        W2 = np.eye(3)
        b2 = np.ones(3)
        model = NumpyModel([(W1, b1), (W2, b2)], activation="relu")
        x = np.array([[1.0, 2.0, 3.0]])
        out = model.forward(x)
        # With relu: layer1 = relu(x + 1) = x + 1 (since all positive)
        # layer2 = (x+1) + 1 = x + 2
        expected = x + 2.0
        np.testing.assert_array_almost_equal(out, expected)

    def test_train_eval_mode_toggle(self) -> None:
        """train_mode() and eval_mode() toggle without error."""
        W = np.eye(2)
        b = np.zeros(2)
        model = NumpyModel([(W, b)])
        model.train_mode()
        model.eval_mode()
        model.train_mode()


class TestNumpyModelBackward:
    """Backward pass tests."""

    def test_backward_populates_grads_single_layer(self) -> None:
        """backward() sets param.grad for single-layer model."""
        W = np.array([[1.0, 2.0]])
        b = np.array([0.5])
        model = NumpyModel([(W, b)])
        x = np.array([[1.0, 3.0]])
        model.forward(x)  # populates cache
        dL_dout = np.array([[0.5]])
        model.backward(dL_dout)
        # dL/db = sum(dL/dout) = 0.5
        np.testing.assert_array_almost_equal(
            list(model.parameters())[1].grad, np.array([0.5])
        )
        # dL/dW = dL/dout.T @ x = [[0.5]] @ [[1,3]] = [[0.5, 1.5]]
        expected_dW = dL_dout.T @ x  # (1,1) @ (1,2) = (1,2)
        np.testing.assert_array_almost_equal(
            list(model.parameters())[0].grad, expected_dW
        )

    def test_backward_propagates_through_two_layers(self) -> None:
        """Gradients flow through both layers."""
        rng = np.random.default_rng(42)
        W1 = rng.standard_normal((4, 3)) * 0.1
        b1 = np.zeros(4)
        W2 = rng.standard_normal((2, 4)) * 0.1
        b2 = np.zeros(2)
        model = NumpyModel([(W1, b1), (W2, b2)])
        x = rng.standard_normal((5, 3))
        model.forward(x)
        dL_dout = rng.standard_normal((5, 2))
        model.backward(dL_dout)
        params = list(model.parameters())
        for p in params:
            assert p.grad is not None
            assert np.any(p.grad != 0), f"{p.name} grad is all zeros"


class TestNumpyOptimizer:
    """Tests for NumpyOptimizer adapter."""

    def test_step_delegates_to_rule(self) -> None:
        """step() calls rule.update() for each parameter."""
        from pipeline.training.optimizers import SGD

        p1 = Parameter(data=np.array([1.0, 2.0]), grad=np.array([0.1, 0.2]), name="w")
        p2 = Parameter(data=np.array([3.0]), grad=np.array([0.3]), name="b")
        opt = NumpyOptimizer([p1, p2], SGD(lr=1.0))
        opt.step()
        np.testing.assert_array_almost_equal(p1.data, np.array([0.9, 1.8]))
        np.testing.assert_array_almost_equal(p2.data, np.array([2.7]))

    def test_zero_grad_resets_all_grads(self) -> None:
        """zero_grad() sets all param.grad to zero."""
        from pipeline.training.optimizers import SGD

        p1 = Parameter(data=np.array([1.0]), grad=np.array([0.5]), name="w")
        p2 = Parameter(data=np.array([2.0]), grad=np.array([0.3]), name="b")
        opt = NumpyOptimizer([p1, p2], SGD(lr=0.1))
        opt.zero_grad()
        np.testing.assert_array_equal(p1.grad, np.array([0.0]))
        np.testing.assert_array_equal(p2.grad, np.array([0.0]))

    def test_zero_grad_handles_none_grad(self) -> None:
        """zero_grad() doesn't crash when param.grad is None."""
        from pipeline.training.optimizers import SGD

        p = Parameter(data=np.array([1.0]), grad=None, name="w")
        opt = NumpyOptimizer([p], SGD(lr=0.1))
        opt.zero_grad()  # should not raise
        assert p.grad is None  # unchanged

    def test_step_zero_grad_step_cycle(self) -> None:
        """Full cycle: zero_grad → backward sets grad → step → zero_grad."""
        from pipeline.training.optimizers import SGD

        p = Parameter(data=np.array([5.0]), grad=np.array([0.0]), name="w")
        opt = NumpyOptimizer([p], SGD(lr=1.0))

        # Simulate a training step
        opt.zero_grad()
        np.testing.assert_array_equal(p.grad, np.array([0.0]))
        p.grad = np.array([2.0])  # simulate backward
        opt.step()
        assert p.data[0] == 3.0  # 5.0 - 1.0*2.0
        opt.zero_grad()
        np.testing.assert_array_equal(p.grad, np.array([0.0]))
