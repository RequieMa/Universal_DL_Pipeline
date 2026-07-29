"""Unit tests for pipeline.protocols — data structures and abstract interfaces."""
from collections.abc import Iterator

import numpy as np
import pytest

from pipeline.protocols import (
    ArrayLike,
    Batch,
    DataStream,
    Loss,
    LossProtocol,
    ModelProtocol,
    OptimizerProtocol,
    Parameter,
)


class TestParameter:
    """Tests for Parameter dataclass."""

    def test_create_with_data_only(self):
        """Happy Path: Parameter with only data creates valid object."""
        data = np.array([1.0, 2.0, 3.0])
        p = Parameter(data=data)
        assert np.array_equal(p.data, data)
        assert p.grad is None
        assert p.name == ""

    def test_create_with_all_fields(self):
        """Happy Path: Parameter with all fields sets each correctly."""
        data = np.zeros((3, 4))
        grad = np.ones((3, 4))
        p = Parameter(data=data, grad=grad, name="weight")
        assert np.array_equal(p.data, data)
        assert np.array_equal(p.grad, grad)
        assert p.name == "weight"

    def test_equal_parameters_compare_equal(self):
        """Happy Path: Parameters with same values are equal."""
        data = np.array([1.0, 2.0])
        p1 = Parameter(data=data)
        p2 = Parameter(data=data.copy())
        assert p1 == p2

    def test_different_parameters_compare_unequal(self):
        """Boundary: Parameters with different data are not equal."""
        p1 = Parameter(data=np.array([1.0]))
        p2 = Parameter(data=np.array([2.0]))
        assert p1 != p2

    def test_none_grad_vs_zero_grad_unequal(self):
        """Boundary: Parameter with None grad differs from zero grad."""
        p1 = Parameter(data=np.array([1.0]), grad=None)
        p2 = Parameter(data=np.array([1.0]), grad=np.array([0.0]))
        assert p1 != p2


class TestBatch:
    """Tests for Batch dataclass."""

    def test_create_batch_with_arrays(self):
        """Happy Path: Batch with numpy arrays."""
        inputs = np.random.randn(32, 10)
        targets = np.random.randint(0, 2, size=(32,))
        batch = Batch(inputs=inputs, targets=targets)
        assert np.array_equal(batch.inputs, inputs)
        assert np.array_equal(batch.targets, targets)

    def test_batch_single_sample(self):
        """Boundary: Batch with batch_size=1 (single sample)."""
        inputs = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([0])
        batch = Batch(inputs=inputs, targets=targets)
        assert batch.inputs.shape == (1, 3)
        assert batch.targets.shape == (1,)

    def test_batch_with_lists_becomes_arraylike(self):
        """Type Error boundary: Batch accepts list inputs (duck-typed)."""
        batch = Batch(inputs=[[1.0], [2.0]], targets=[0, 1])
        assert batch.inputs is not None

    def test_batch_empty_arrays(self):
        """Empty boundary: Batch with zero-length arrays."""
        inputs = np.array([]).reshape(0, 10)
        targets = np.array([])
        batch = Batch(inputs=inputs, targets=targets)
        assert len(batch.inputs) == 0
        assert len(batch.targets) == 0


class TestArrayLike:
    """Tests for ArrayLike type alias compatibility."""

    def test_ndarray_is_arraylike(self):
        """Happy Path: numpy.ndarray is compatible with ArrayLike."""
        x: ArrayLike = np.array([1.0, 2.0])
        assert x is not None

    def test_parameter_data_is_arraylike(self):
        """Happy Path: Parameter.data is type-compatible with ArrayLike."""
        p = Parameter(data=np.ones(5))
        _data: ArrayLike = p.data
        assert _data is not None


class TestLoss:
    """Tests for the Loss value object."""

    def test_float_conversion(self):
        """Happy Path: float(loss) extracts the scalar value."""
        loss = Loss(value=0.5)
        assert float(loss) == 0.5

    def test_backward_noop_when_no_fn(self):
        """Happy Path: backward() is a no-op when _backward_fn is None."""
        loss = Loss(value=0.5)
        loss.backward()  # should not raise

    def test_backward_calls_fn(self):
        """Happy Path: backward() calls the stored function."""
        called = []
        loss = Loss(value=0.5, _backward_fn=lambda: called.append(1))
        loss.backward()
        assert called == [1]

    def test_backward_fn_receives_no_args(self):
        """Boundary: backward_fn receives zero arguments."""
        captured = None

        def _backward():
            nonlocal captured
            captured = "ran"

        loss = Loss(value=1.0, _backward_fn=_backward)
        loss.backward()
        assert captured == "ran"

    def test_negative_loss_value(self):
        """Boundary: negative loss values are preserved."""
        loss = Loss(value=-3.2)
        assert float(loss) == -3.2

    def test_zero_loss_value(self):
        """Boundary: zero loss value."""
        loss = Loss(value=0.0)
        assert float(loss) == 0.0


# ── Fake implementations for testing ABCs ────────────────────────────────
class FakeDataStream(DataStream):
    """Minimal DataStream implementation for testing."""

    def __init__(self, batches: list[Batch]):
        self._batches = batches

    def __iter__(self) -> Iterator[Batch]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


class FakeModel(ModelProtocol):
    """Minimal ModelProtocol implementation for testing."""

    def __init__(self):
        self._params = [Parameter(data=np.array([1.0]), name="w")]
        self._mode = "train"

    def forward(self, inputs: ArrayLike) -> ArrayLike:
        return np.asarray(inputs) * 2.0

    def parameters(self) -> Iterator[Parameter]:
        return iter(self._params)

    def train_mode(self) -> None:
        self._mode = "train"

    def eval_mode(self) -> None:
        self._mode = "eval"


class FakeLoss(LossProtocol):
    """Minimal LossProtocol implementation for testing."""

    def forward(self, predictions: ArrayLike, targets: ArrayLike) -> Loss:
        preds = np.asarray(predictions)
        targs = np.asarray(targets)
        return float(np.mean((preds - targs) ** 2))


class FakeOptimizer(OptimizerProtocol):
    """Minimal OptimizerProtocol implementation for testing."""

    def __init__(self):
        self.step_count = 0
        self.zero_count = 0

    def step(self) -> None:
        self.step_count += 1

    def zero_grad(self) -> None:
        self.zero_count += 1


# ── DataStream tests ─────────────────────────────────────────────────────
class TestDataStream:
    """Tests for DataStream ABC."""

    def test_cannot_instantiate_abc_directly(self):
        """Type Error: instantiating DataStream directly raises TypeError."""
        with pytest.raises(TypeError):
            DataStream()  # type: ignore[abstract]

    def test_iter_yields_batches(self):
        """Happy Path: DataStream iter yields Batch objects."""
        batch = Batch(inputs=np.array([1.0]), targets=np.array([2.0]))
        stream = FakeDataStream([batch])
        items = list(stream)
        assert len(items) == 1
        assert isinstance(items[0], Batch)

    def test_len_returns_count(self):
        """Happy Path: DataStream len returns batch count."""
        batches = [
            Batch(inputs=np.array([i]), targets=np.array([i]))
            for i in range(3)
        ]
        stream = FakeDataStream(batches)
        assert len(stream) == 3

    def test_empty_stream_len_zero(self):
        """Empty: DataStream with no batches has len=0."""
        stream = FakeDataStream([])
        assert len(stream) == 0

    def test_empty_stream_iter_empty(self):
        """Empty: iterating empty DataStream yields nothing."""
        stream = FakeDataStream([])
        items = list(stream)
        assert items == []

    def test_stress_many_batches(self):
        """Stress: DataStream with 1000 batches iterates correctly."""
        n = 1000
        batches = [
            Batch(inputs=np.array([float(i)]), targets=np.array([float(i)]))
            for i in range(n)
        ]
        stream = FakeDataStream(batches)
        assert len(stream) == n
        count = 0
        for _batch in stream:
            count += 1
        assert count == n


# ── ModelProtocol tests ───────────────────────────────────────────────────
class TestModelProtocol:
    """Tests for ModelProtocol ABC."""

    def test_cannot_instantiate_abc_directly(self):
        """Type Error: instantiating ModelProtocol directly raises TypeError."""
        with pytest.raises(TypeError):
            ModelProtocol()  # type: ignore[abstract]

    def test_forward_returns_arraylike(self):
        """Happy Path: forward returns array-like output."""
        model = FakeModel()
        inputs = np.array([1.0, 2.0, 3.0])
        output = model.forward(inputs)
        assert output is not None
        assert len(np.asarray(output)) == 3

    def test_parameters_returns_iterable(self):
        """Happy Path: parameters returns iterable of Parameter objects."""
        model = FakeModel()
        params = list(model.parameters())
        assert len(params) == 1
        assert isinstance(params[0], Parameter)

    def test_empty_parameters_is_valid(self):
        """Boundary: model with zero parameters (e.g., sklearn wrapper)
        returns empty iterable."""

        class NoParamModel(ModelProtocol):
            def forward(self, inputs):
                return np.asarray(inputs)

            def parameters(self):
                return iter([])

            def train_mode(self):
                pass

            def eval_mode(self):
                pass

        model = NoParamModel()
        params = list(model.parameters())
        assert params == []

    def test_train_mode_sets_training(self):
        """Happy Path: train_mode() switches to training mode."""
        model = FakeModel()
        model.train_mode()
        assert model._mode == "train"

    def test_eval_mode_sets_evaluation(self):
        """Happy Path: eval_mode() switches to evaluation mode."""
        model = FakeModel()
        model.eval_mode()
        assert model._mode == "eval"

    def test_toggle_mode_repeatedly(self):
        """Concurrency: toggling train/eval repeatedly works correctly."""
        model = FakeModel()
        for _ in range(10):
            model.train_mode()
            model.eval_mode()
        assert model._mode == "eval"


# ── LossProtocol tests ─────────────────────────────────────────────────────
class TestLossProtocol:
    """Tests for LossProtocol ABC."""

    def test_cannot_instantiate_abc_directly(self):
        """Type Error: instantiating LossProtocol directly raises TypeError."""
        with pytest.raises(TypeError):
            LossProtocol()  # type: ignore[abstract]

    def test_call_delegates_to_forward(self):
        """Happy Path: __call__ delegates to forward() and returns float."""
        loss_fn = FakeLoss()
        preds = np.array([1.0, 2.0])
        targs = np.array([0.5, 2.5])
        result = loss_fn(preds, targs)
        assert isinstance(result, float)

    def test_perfect_prediction_yields_zero_loss(self):
        """Boundary: perfect predictions produce near-zero loss."""
        loss_fn = FakeLoss()
        preds = np.array([1.0, 2.0, 3.0])
        targs = np.array([1.0, 2.0, 3.0])
        result = loss_fn(preds, targs)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_wrong_predictions_yield_positive_loss(self):
        """Happy Path: wrong predictions produce positive loss."""
        loss_fn = FakeLoss()
        preds = np.array([0.0, 0.0])
        targs = np.array([10.0, 10.0])
        result = loss_fn(preds, targs)
        assert result > 0.0

    def test_single_sample(self):
        """Boundary: loss computed on a single-sample batch."""
        loss_fn = FakeLoss()
        result = loss_fn(np.array([3.0]), np.array([1.0]))
        assert isinstance(result, float)
        assert result > 0.0


# ── OptimizerProtocol tests ─────────────────────────────────────────────────
class TestOptimizerProtocol:
    """Tests for OptimizerProtocol ABC."""

    def test_cannot_instantiate_abc_directly(self):
        """Type Error: instantiating OptimizerProtocol directly raises TypeError."""
        with pytest.raises(TypeError):
            OptimizerProtocol()  # type: ignore[abstract]

    def test_step_increments(self):
        """Happy Path: step() executes without error."""
        opt = FakeOptimizer()
        opt.step()
        assert opt.step_count == 1

    def test_zero_grad_increments(self):
        """Happy Path: zero_grad() executes without error."""
        opt = FakeOptimizer()
        opt.zero_grad()
        assert opt.zero_count == 1

    def test_multiple_steps_accumulate(self):
        """Concurrency: multiple step() calls accumulate correctly."""
        opt = FakeOptimizer()
        for _ in range(5):
            opt.step()
        assert opt.step_count == 5

    def test_zero_grad_then_step_independent(self):
        """Concurrency: zero_grad and step counts are independent."""
        opt = FakeOptimizer()
        opt.zero_grad()
        opt.step()
        opt.zero_grad()
        assert opt.zero_count == 2
        assert opt.step_count == 1

    def test_stress_many_optimizer_steps(self):
        """Stress: 1000 step() calls complete without memory issues."""
        opt = FakeOptimizer()
        for _ in range(1000):
            opt.step()
            opt.zero_grad()
        assert opt.step_count == 1000
        assert opt.zero_count == 1000
