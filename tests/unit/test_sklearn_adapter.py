"""Unit tests for pipeline.adapters.sklearn_adapter."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.adapters.sklearn_adapter import SklearnModel, StubLoss, StubOptimizer
from pipeline.protocols import Loss

try:
    import sklearn  # noqa: F401

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

requires_sklearn = pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")


class TestStubLoss:
    """Tests for StubLoss."""

    def test_forward_returns_loss_zero(self) -> None:
        """StubLoss always returns Loss(0.0)."""
        loss_fn = StubLoss()
        result = loss_fn.forward(np.array([1.0, 2.0]), np.array([0.0, 1.0]))
        assert isinstance(result, Loss)
        assert float(result) == 0.0

    def test_call_delegates_to_forward(self) -> None:
        """__call__ returns same as forward()."""
        loss_fn = StubLoss()
        result = loss_fn(np.array([1.0]), np.array([0.0]))
        assert float(result) == 0.0

    def test_backward_is_noop(self) -> None:
        """Loss.backward() does not raise (no-op)."""
        loss_fn = StubLoss()
        loss = loss_fn.forward(np.array([1.0]), np.array([0.0]))
        loss.backward()  # should not raise


class TestStubOptimizer:
    """Tests for StubOptimizer."""

    def test_step_is_noop(self) -> None:
        """step() does nothing, does not raise."""
        opt = StubOptimizer()
        opt.step()  # should not raise

    def test_zero_grad_is_noop(self) -> None:
        """zero_grad() does nothing, does not raise."""
        opt = StubOptimizer()
        opt.zero_grad()  # should not raise

    def test_multiple_calls_do_not_raise(self) -> None:
        """Calling step/zero_grad repeatedly is safe."""
        opt = StubOptimizer()
        for _ in range(10):
            opt.zero_grad()
            opt.step()


@requires_sklearn
class TestSklearnModel:
    """Tests for SklearnModel adapter."""

    def test_parameters_returns_empty(self) -> None:
        """Sklearn models have no gradient parameters."""
        from sklearn.linear_model import LogisticRegression

        model = SklearnModel(LogisticRegression())
        params = list(model.parameters())
        assert params == []

    def test_train_eval_mode_noop(self) -> None:
        """train_mode() and eval_mode() do not raise."""
        from sklearn.linear_model import LogisticRegression

        model = SklearnModel(LogisticRegression())
        model.train_mode()
        model.eval_mode()

    def test_forward_shape_after_fit(self) -> None:
        """forward() returns predictions with correct shape after fit."""
        from sklearn.linear_model import LogisticRegression

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([0, 1, 0, 1])

        est = LogisticRegression(max_iter=1000)
        est.fit(X, y)

        model = SklearnModel(est)
        preds = model.forward(np.array([[1.0, 2.0], [3.0, 4.0]]))
        # LogisticRegression with 2 classes → predict_proba returns (n, 2)
        assert preds.shape == (2, 2)
        # Probabilities sum to ~1 per row
        assert np.allclose(preds.sum(axis=1), 1.0, atol=0.01)

    def test_forward_delegates_to_predict_proba(self) -> None:
        """forward() uses predict_proba when available (classifier)."""
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression(max_iter=1000)
        est.fit(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            np.array([0, 1, 0, 1]),
        )
        model = SklearnModel(est)
        preds = model.forward(np.array([[1.0, 2.0]]))
        # Should match predict_proba output
        expected = est.predict_proba(np.array([[1.0, 2.0]]))
        np.testing.assert_array_almost_equal(preds, expected)

    def test_forward_uses_predict_when_no_proba(self) -> None:
        """forward() falls back to predict() for regressors."""
        from sklearn.linear_model import LinearRegression

        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        est = LinearRegression()
        est.fit(X, y)

        model = SklearnModel(est)
        preds = model.forward(np.array([[1.5], [2.5]]))
        assert preds.shape == (2,)
        expected = est.predict(np.array([[1.5], [2.5]]))
        np.testing.assert_array_almost_equal(preds, expected)
