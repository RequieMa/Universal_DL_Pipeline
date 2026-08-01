"""Unit tests for pipeline.training.losses — MSELoss, CrossEntropyLoss."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.protocols import Loss, Parameter


class TestMSELoss:
    """Tests for MSELoss."""

    def test_forward_returns_loss_object(self) -> None:
        """forward() returns a Loss instance."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        result = loss_fn.forward(pred, target)
        assert isinstance(result, Loss)

    def test_perfect_prediction_zero_loss(self) -> None:
        """MSE = 0 when predictions equal targets."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        loss = loss_fn.forward(pred, target)
        assert float(loss) == 0.0

    def test_positive_loss_for_errors(self) -> None:
        """MSE > 0 when predictions differ from targets."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()
        pred = np.array([1.0, 2.0])
        target = np.array([3.0, 4.0])
        loss = loss_fn.forward(pred, target)
        # MSE = mean((1-3)^2 + (2-4)^2) = mean(4+4) = 4
        assert float(loss) == pytest.approx(4.0)

    def test_call_delegates_to_forward(self) -> None:
        """__call__ returns same as forward()."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()
        pred = np.array([1.0, 2.0])
        target = np.array([1.0, 3.0])
        result_call = loss_fn(pred, target)
        result_fwd = loss_fn.forward(pred, target)
        assert float(result_call) == float(result_fwd)

    def test_backward_noop_without_model(self) -> None:
        """backward() is a no-op when no model is attached."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()  # no model
        pred = np.array([1.0, 2.0])
        target = np.array([0.0, 1.0])
        loss = loss_fn.forward(pred, target)
        loss.backward()  # should not raise

    def test_backward_with_model_populates_grads(self) -> None:
        """backward() propagates dL/dpred through attached model."""
        from pipeline.training.losses import MSELoss

        class _MinimalModel:
            """Single-layer linear model for MSELoss gradient test."""

            def __init__(self) -> None:
                self.W = Parameter(
                    data=np.array([[2.0]]),
                    grad=np.zeros((1, 1)),
                    name="W",
                )
                self.b = Parameter(
                    data=np.array([0.0]),
                    grad=np.zeros(1),
                    name="b",
                )
                self._params = [self.W, self.b]
                self._cache: dict[str, np.ndarray] = {}

            def forward(self, x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=np.float64)
                self._cache["a0"] = x
                return x @ self.W.data.T + self.b.data

            def backward(self, dL_doutput: np.ndarray) -> None:
                self.W.grad = dL_doutput.T @ self._cache["a0"]
                self.b.grad = dL_doutput.sum(axis=0)

        model = _MinimalModel()
        loss_fn = MSELoss(model=model)
        x = np.array([[1.0], [2.0]])
        y = np.array([[3.0], [5.0]])
        pred = model.forward(x)
        loss = loss_fn.forward(pred, y)
        loss.backward()
        assert model.W.grad is not None
        assert model.b.grad is not None
        assert np.any(model.W.grad != 0)

    def test_2d_output(self) -> None:
        """MSE works with 2D predictions (batch_size, output_dim)."""
        from pipeline.training.losses import MSELoss

        loss_fn = MSELoss()
        pred = np.array([[0.5, 0.5], [0.2, 0.8]])
        target = np.array([[1.0, 0.0], [0.0, 1.0]])
        loss = loss_fn.forward(pred, target)
        expected = float(np.mean((pred - target) ** 2))
        assert float(loss) == pytest.approx(expected)


class TestCrossEntropyLoss:
    """Tests for CrossEntropyLoss."""

    def test_forward_returns_loss_object(self) -> None:
        """forward() returns a Loss instance."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[2.0, 1.0], [0.5, 1.5]])
        targets = np.array([0, 1])
        result = loss_fn.forward(logits, targets)
        assert isinstance(result, Loss)

    def test_perfect_prediction_low_loss(self) -> None:
        """Loss is near zero for confident correct predictions."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[10.0, -10.0], [-10.0, 10.0]])
        targets = np.array([0, 1])
        loss = loss_fn.forward(logits, targets)
        assert float(loss) < 0.001

    def test_wrong_prediction_high_loss(self) -> None:
        """Loss is high for confident wrong predictions."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[-10.0, 10.0], [10.0, -10.0]])
        targets = np.array([0, 1])
        loss = loss_fn.forward(logits, targets)
        assert float(loss) > 10.0

    def test_call_delegates_to_forward(self) -> None:
        """__call__ returns same as forward()."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[1.0, 2.0], [3.0, 1.0]])
        targets = np.array([1, 0])
        assert float(loss_fn(logits, targets)) == float(loss_fn.forward(logits, targets))

    def test_backward_noop_without_model(self) -> None:
        """backward() is no-op when no model attached."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[1.0, 2.0]])
        targets = np.array([0])
        loss = loss_fn.forward(logits, targets)
        loss.backward()  # should not raise

    def test_backward_with_model_populates_grads(self) -> None:
        """backward() propagates through attached model."""
        from pipeline.training.losses import CrossEntropyLoss

        class _MinimalModel:
            """Two-class linear model for CrossEntropyLoss gradient test."""

            def __init__(self) -> None:
                self.W = Parameter(
                    data=np.array([[0.5, 0.5], [0.5, 0.5]]),
                    grad=np.zeros((2, 2)),
                    name="W",
                )
                self.b = Parameter(
                    data=np.array([0.0, 0.0]),
                    grad=np.zeros(2),
                    name="b",
                )
                self._params = [self.W, self.b]
                self._cache: dict[str, np.ndarray] = {}

            def forward(self, x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=np.float64)
                self._cache["a0"] = x
                return x @ self.W.data.T + self.b.data

            def backward(self, dL_doutput: np.ndarray) -> None:
                self.W.grad = dL_doutput.T @ self._cache["a0"]
                self.b.grad = dL_doutput.sum(axis=0)

        model = _MinimalModel()
        loss_fn = CrossEntropyLoss(model=model)
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        pred = model.forward(x)
        loss = loss_fn.forward(pred, y)
        loss.backward()
        assert model.W.grad is not None
        assert model.b.grad is not None
        assert np.any(model.W.grad != 0)

    def test_batch_independence(self) -> None:
        """Loss for a batch equals mean of per-sample losses."""
        from pipeline.training.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = np.array([[3.0, 1.0], [1.0, 3.0]])
        targets = np.array([0, 1])
        loss_batch = float(loss_fn.forward(logits, targets))

        loss0 = float(loss_fn.forward(np.array([[3.0, 1.0]]), np.array([0])))
        loss1 = float(loss_fn.forward(np.array([[1.0, 3.0]]), np.array([1])))
        assert loss_batch == pytest.approx((loss0 + loss1) / 2.0)
