"""Transform decorator for data streams.

Provides :class:`TransformedDataStream`, a :class:`DataStream` wrapper
that applies a user-provided callable to each :class:`Batch` before
yielding it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from pipeline.protocols import Batch, DataStream


class TransformedDataStream(DataStream):
    """Decorator DataStream that transforms each batch.

    Wraps an existing :class:`DataStream` and applies ``transform``
    to every :class:`Batch` before yielding it. The transform is fully
    opaque — it can convert PIL images to tensors, apply augmentations,
    normalize pixel values, or any other per-batch operation.

    Chain multiple :class:`TransformedDataStream` instances to compose
    a sequence of preprocessing steps.

    Usage::

        raw = ImageFolderDataSource("data/train")
        to_tensor = transforms.Compose([...])   # torchvision

        def transform_fn(batch):
            return Batch(
                inputs=torch.stack([to_tensor(img) for img in batch.inputs]),
                targets=torch.as_tensor(batch.targets),
            )

        stream = TransformedDataStream(raw, transform_fn)
        for batch in stream:
            # batch.inputs is a torch.Tensor

    Args:
        stream: The wrapped :class:`DataStream` to read from.
        transform: A callable ``(Batch) -> Batch`` applied to each batch.
    """

    def __init__(self, stream: DataStream, transform: Callable[[Batch], Batch]) -> None:
        """Wrap a DataStream with a transform.

        Args:
            stream: The underlying :class:`DataStream`.
            transform: Callable receiving and returning a :class:`Batch`.
        """
        self._stream = stream
        self._transform = transform

    def __iter__(self) -> Iterator[Batch]:
        """Yield transformed batches.

        Iterates the wrapped stream and applies ``transform`` to each batch.
        """
        for batch in self._stream:
            yield self._transform(batch)

    def __len__(self) -> int:
        """Number of batches — delegated to the wrapped stream."""
        return len(self._stream)
