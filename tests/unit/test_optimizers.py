"""Unit tests for pipeline.training.optimizers — SGD, Adam."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.protocols import Parameter


class TestSGD:
    """Tests for SGD optimizer rule."""

    def test_update_reduces_param_by_lr_times_grad(self) -> None:
        """SGD: data = data - lr * grad."""
        from pipeline.training.optimizers import SGD

        param = Parameter(
            data=np.array([1.0, 2.0, 3.0]),
            grad=np.array([0.1, 0.2, 0.3]),
            name="w",
        )
        sgd = SGD(lr=0.1)
        sgd.update(param)
        expected = np.array([1.0, 2.0, 3.0]) - 0.1 * np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(param.data, expected)

    def test_update_with_default_lr(self) -> None:
        """Default lr=0.01."""
        from pipeline.training.optimizers import SGD

        param = Parameter(
            data=np.array([5.0]),
            grad=np.array([2.0]),
            name="w",
        )
        sgd = SGD()
        sgd.update(param)
        np.testing.assert_array_almost_equal(param.data, np.array([4.98]))

    def test_zero_gradient_no_change(self) -> None:
        """When grad is all zeros, data is unchanged."""
        from pipeline.training.optimizers import SGD

        param = Parameter(
            data=np.array([1.0, 2.0]),
            grad=np.array([0.0, 0.0]),
            name="w",
        )
        original = param.data.copy()
        sgd = SGD(lr=0.1)
        sgd.update(param)
        np.testing.assert_array_equal(param.data, original)

    def test_lr_zero_raises(self) -> None:
        """lr=0.0 raises ValueError (useless and likely a mistake)."""
        from pipeline.training.optimizers import SGD

        with pytest.raises(ValueError, match="positive"):
            SGD(lr=0.0)

    def test_negative_lr_raises(self) -> None:
        """Negative learning rate should raise ValueError."""
        from pipeline.training.optimizers import SGD

        with pytest.raises(ValueError, match="positive"):
            SGD(lr=-0.1)

    def test_missing_grad_raises_valueerror(self) -> None:
        """A param with grad=None raises ValueError (survives ``python -O``)."""
        from pipeline.training.optimizers import SGD

        param = Parameter(data=np.array([1.0, 2.0]), grad=None, name="w")
        with pytest.raises(ValueError, match="no gradient"):
            SGD(lr=0.1).update(param)


class TestAdam:
    """Tests for Adam optimizer rule."""

    def test_update_reduces_loss_direction(self) -> None:
        """Adam steps in the direction that reduces the gradient."""
        from pipeline.training.optimizers import Adam

        param = Parameter(
            data=np.array([1.0, 1.0]),
            grad=np.array([0.5, 0.5]),
            name="w",
        )
        adam = Adam(lr=0.1)
        adam.update(param)
        # Parameter should decrease (grad is positive → step is negative)
        assert param.data[0] < 1.0
        assert param.data[1] < 1.0

    def test_zero_gradient_no_change(self) -> None:
        """When grad is all zeros, data is unchanged."""
        from pipeline.training.optimizers import Adam

        param = Parameter(
            data=np.array([3.0, 4.0]),
            grad=np.array([0.0, 0.0]),
            name="w",
        )
        original = param.data.copy()
        adam = Adam(lr=0.01)
        adam.update(param)
        np.testing.assert_array_equal(param.data, original)

    def test_different_params_have_separate_momentum(self) -> None:
        """Each parameter gets its own m and v buffers."""
        from pipeline.training.optimizers import Adam

        p1 = Parameter(data=np.array([1.0]), grad=np.array([0.3]), name="w1")
        p2 = Parameter(data=np.array([2.0]), grad=np.array([0.7]), name="w2")
        adam = Adam(lr=0.1)
        adam.update(p1)
        adam.update(p2)
        # Both should have moved
        assert p1.data[0] != 1.0
        assert p2.data[0] != 2.0

    def test_negative_lr_raises(self) -> None:
        """Negative learning rate should raise ValueError."""
        from pipeline.training.optimizers import Adam

        with pytest.raises(ValueError, match="positive"):
            Adam(lr=-0.001)

    def test_missing_grad_raises_valueerror(self) -> None:
        """A param with grad=None raises ValueError (survives ``python -O``)."""
        from pipeline.training.optimizers import Adam

        param = Parameter(data=np.array([1.0, 2.0]), grad=None, name="w")
        with pytest.raises(ValueError, match="no gradient"):
            Adam(lr=0.1).update(param)

    def test_step_counter_is_per_parameter_not_inflated_by_count(self) -> None:
        """Regression: each param has its own step counter ``t``.

        ``NumpyOptimizer.step()`` calls ``update`` once per parameter, so one
        optimizer step over N params must leave every param at ``t == 1`` (not
        t == 1, 2, ..., N). Otherwise each param gets a different — and wrong —
        bias-correction factor within a single step.
        """
        from pipeline.training.optimizers import Adam

        lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8

        def adam_step_one(data: float, grad: float) -> float:
            """Correct Adam update from zero state at ``t == 1``."""
            m_hat = grad  # ((1 - beta1) * grad) / (1 - beta1)
            v_hat = grad**2  # ((1 - beta2) * grad**2) / (1 - beta2)
            return data - lr * m_hat / (np.sqrt(v_hat) + eps)

        p1 = Parameter(data=np.array([1.0]), grad=np.array([0.3]), name="w1")
        p2 = Parameter(data=np.array([2.0]), grad=np.array([0.7]), name="w2")
        adam = Adam(lr=lr, betas=(beta1, beta2), eps=eps)

        # One optimizer "step" == update every param exactly once.
        adam.update(p1)
        adam.update(p2)

        np.testing.assert_allclose(p1.data, np.array([adam_step_one(1.0, 0.3)]))
        np.testing.assert_allclose(p2.data, np.array([adam_step_one(2.0, 0.7)]))

        # White-box sanity: both counters advanced to exactly 1.
        assert adam._t[id(p1)] == 1
        assert adam._t[id(p2)] == 1

        # A param seen for the first time AFTER those two updates must still
        # start at t == 1 — the global update count (2) must not leak in.
        p3 = Parameter(data=np.array([5.0]), grad=np.array([0.4]), name="w3")
        adam.update(p3)
        np.testing.assert_allclose(p3.data, np.array([adam_step_one(5.0, 0.4)]))
        assert adam._t[id(p3)] == 1

    def test_per_param_counter_advances_once_per_step(self) -> None:
        """Over K optimizer steps a param's ``t`` reaches K, not K * n_params."""
        from pipeline.training.optimizers import Adam

        p1 = Parameter(data=np.array([1.0]), grad=np.array([0.3]), name="w1")
        p2 = Parameter(data=np.array([2.0]), grad=np.array([0.7]), name="w2")
        adam = Adam(lr=0.1)

        for _ in range(3):  # 3 optimizer steps over 2 params
            adam.update(p1)
            adam.update(p2)

        assert adam._t[id(p1)] == 3
        assert adam._t[id(p2)] == 3

