"""Unit tests for pipeline.data.transforms — TransformedDataStream."""

from __future__ import annotations

from pathlib import Path  # noqa: F401

import numpy as np
import pytest  # noqa: F401

from pipeline.data.transforms import TransformedDataStream
from pipeline.protocols import Batch


class _IdentityStream:
    """Minimal DataStream for testing transforms (no batch_size logic)."""

    def __init__(self, batches: list[Batch]) -> None:
        self._batches = batches
        self._call_count = 0

    def __iter__(self):
        self._call_count += 1
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


class TestTransformedDataStream:
    """Tests for TransformedDataStream."""

    def test_identity_transform_passthrough(self) -> None:
        """Identity transform returns batches unchanged."""
        original = Batch(inputs=np.array([1.0, 2.0]), targets=np.array([0]))
        stream = _IdentityStream([original])
        transformed = TransformedDataStream(stream, lambda b: b)
        result = next(iter(transformed))
        assert result is original  # identity: same object

    def test_transform_modifies_inputs(self) -> None:
        """Transform can modify batch.inputs."""
        original = Batch(inputs=np.array([1.0, 2.0]), targets=np.array([0]))
        stream = _IdentityStream([original])

        def double_inputs(batch: Batch) -> Batch:
            return Batch(inputs=batch.inputs * 2, targets=batch.targets)

        transformed = TransformedDataStream(stream, double_inputs)
        result = next(iter(transformed))
        np.testing.assert_array_equal(result.inputs, np.array([2.0, 4.0]))

    def test_transform_can_modify_targets(self) -> None:
        """Transform can modify batch.targets (e.g., one-hot encoding)."""
        original = Batch(inputs=np.array([1.0]), targets=np.array([2]))
        stream = _IdentityStream([original])

        def one_hot(batch: Batch) -> Batch:
            num_classes = 5
            oh = np.zeros((len(batch.targets), num_classes))
            oh[np.arange(len(batch.targets)), batch.targets] = 1
            return Batch(inputs=batch.inputs, targets=oh)

        transformed = TransformedDataStream(stream, one_hot)
        result = next(iter(transformed))
        assert result.targets.shape == (1, 5)
        np.testing.assert_array_equal(result.targets, np.array([[0, 0, 1, 0, 0]]))

    def test_len_delegates_to_wrapped_stream(self) -> None:
        """__len__ returns the wrapped stream's length."""
        batches = [
            Batch(inputs=np.array([1.0]), targets=np.array([0])),
            Batch(inputs=np.array([2.0]), targets=np.array([1])),
        ]
        stream = _IdentityStream(batches)
        transformed = TransformedDataStream(stream, lambda b: b)
        assert len(transformed) == 2

    def test_pil_to_numpy_conversion(self) -> None:
        """Transform can convert PIL Images → numpy arrays for an image pipeline."""
        from PIL import Image

        img = Image.new("RGB", (2, 2), color=(128, 64, 32))
        original = Batch(inputs=[img], targets=np.array([0]))
        stream = _IdentityStream([original])

        def pil_to_numpy(batch: Batch) -> Batch:
            arrs = [np.asarray(img) for img in batch.inputs]
            return Batch(inputs=np.stack(arrs), targets=batch.targets)

        transformed = TransformedDataStream(stream, pil_to_numpy)
        result = next(iter(transformed))
        assert isinstance(result.inputs, np.ndarray)
        assert result.inputs.shape == (1, 2, 2, 3)  # (batch, H, W, C)

    def test_chained_transforms(self) -> None:
        """Two TransformedDataStreams can be chained for a pipeline."""
        original = Batch(inputs=np.array([1.0, 2.0, 3.0]), targets=np.array([0]))
        stream = _IdentityStream([original])

        t1 = TransformedDataStream(stream, lambda b: Batch(inputs=b.inputs + 1, targets=b.targets))
        t2 = TransformedDataStream(t1, lambda b: Batch(inputs=b.inputs * 2, targets=b.targets))

        result = next(iter(t2))
        # (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
        np.testing.assert_array_equal(result.inputs, np.array([4.0, 6.0, 8.0]))
