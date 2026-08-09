"""Image folder data source.

Provides :class:`ImageFolderDataSource`, a :class:`DataStream` implementation
that reads images from class-labeled folder trees and yields :class:`Batch`
objects with PIL Image inputs and integer-label targets.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from pipeline.protocols import Batch, DataStream


class ImageFolderDataSource(DataStream):
    """Image folder → Batch iterator.

    Scans a root directory where each subfolder is a class label.
    Subfolder names are sorted alphabetically to produce integer
    labels (0, 1, 2, ...). Each subfolder contains image files.

    Lazy-loading: the folder is scanned on the first call to
    :meth:`__iter__` or :meth:`__len__`, not during ``__init__``.
    This keeps construction cheap and defers filesystem I/O.

    Usage::

        stream = ImageFolderDataSource("data/train", batch_size=32)
        print(stream.class_names)  # -> ["bird", "cat", "dog"]
        for batch in stream:
            # batch.inputs = [PIL.Image, ...]
            # batch.targets = np.array([0, 1, ...])
            ...

    Attributes:
        root_dir: Path to the root image folder.
        batch_size: Number of images per batch.
        shuffle: Whether to shuffle images before batching.
        class_names: List of class folder names, sorted alphabetically.
        n_samples: Total number of images found.
    """

    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """Create an image folder data source.

        Args:
            root_dir: Path to a directory with one subfolder per class.
            batch_size: Number of images per batch.
            shuffle: If True, shuffle samples before batching.
            seed: Random seed for reproducible shuffling.
        """
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._samples: list[tuple[Path, int]] | None = None
        self._class_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> list[str]:
        """Class folder names, sorted alphabetically.

        Lazily scans the root directory on first access.
        """
        self._load()
        assert self._class_names is not None  # _load() always sets this
        return self._class_names

    @property
    def n_samples(self) -> int:
        """Total number of image files found."""
        self._load()
        assert self._samples is not None
        return len(self._samples)

    # ------------------------------------------------------------------
    # DataStream protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Batch]:
        """Yield one :class:`Batch` per batch window.

        Images are loaded from disk on each batch using PIL.
        If ``shuffle=True``, sample order is randomized on each call
        to ``__iter__``.
        """
        from PIL import Image

        self._load()
        assert self._samples is not None

        indices = np.arange(len(self._samples))
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(self._samples), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            images: list[Image.Image] = []
            labels: list[int] = []
            for idx in batch_indices:
                path, label = self._samples[idx]
                images.append(Image.open(path))
                labels.append(label)
            yield Batch(inputs=images, targets=np.array(labels, dtype=np.int64))

    def __len__(self) -> int:
        """Number of batches (``ceil(n_samples / batch_size)``)."""
        self._load()
        assert self._samples is not None
        return max(1, math.ceil(len(self._samples) / self.batch_size))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Scan the root directory and collect (path, label) pairs.

        Called lazily on first access. Caches results in ``_samples``
        and ``_class_names``.

        Raises:
            FileNotFoundError: If ``root_dir`` does not exist.
            ValueError: If no image files are found.
        """
        if self._samples is not None:
            return

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        samples: list[tuple[Path, int]] = []
        class_names: list[str] = []

        # Sort for deterministic class→label mapping
        subdirs = sorted(
            d for d in self.root_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

        if not subdirs:
            raise ValueError(f"No class subdirectories found in {self.root_dir}")

        for label_idx, subdir in enumerate(subdirs):
            class_names.append(subdir.name)
            # Collect image files (common extensions only). Glob both cases
            # (*.png and *.PNG); on case-insensitive filesystems (WSL2 DrvFs,
            # macOS) both patterns match the same file, so dedupe via a set
            # before sorting for deterministic order.
            matches: set[Path] = set()
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"):
                matches.update(subdir.glob(f"*{ext}"))
                matches.update(subdir.glob(f"*{ext.upper()}"))
            for img_path in sorted(matches):
                samples.append((img_path, label_idx))

        if not samples:
            raise ValueError(f"No image files found in {self.root_dir}")

        self._samples = samples
        self._class_names = class_names
