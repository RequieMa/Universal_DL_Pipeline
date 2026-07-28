"""Unit tests for pipeline.protocols — data structures and abstract interfaces."""
import numpy as np

from pipeline.protocols import ArrayLike, Batch, Parameter


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
