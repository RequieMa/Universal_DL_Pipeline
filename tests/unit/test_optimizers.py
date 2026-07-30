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
