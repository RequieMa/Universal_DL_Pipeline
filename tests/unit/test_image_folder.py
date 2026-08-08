"""Unit tests for pipeline.data.image_folder — ImageFolderDataSource."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.data.image_folder import ImageFolderDataSource


class TestImageFolderConstruction:
    """Construction and property tests."""

    def test_construction_stores_params(self, tiny_image_folder: Path) -> None:
        """Construction stores batch_size, shuffle, seed without loading."""
        ds = ImageFolderDataSource(tiny_image_folder, batch_size=4, shuffle=False, seed=123)
        assert ds.batch_size == 4
        assert ds.shuffle is False

    def test_root_dir_missing_raises_on_load(self) -> None:
        """_load() raises FileNotFoundError when root_dir does not exist."""
        ds = ImageFolderDataSource("/nonexistent/path/12345")
        with pytest.raises(FileNotFoundError):
            _ = len(ds)

    def test_class_names_are_sorted(self, tiny_image_folder: Path) -> None:
        """class_names returns subfolder names sorted alphabetically."""
        ds = ImageFolderDataSource(tiny_image_folder)
        assert ds.class_names == ["bird", "cat", "dog"]

    def test_n_samples_counts_all_images(self, tiny_image_folder: Path) -> None:
        """n_samples equals total number of image files across all class dirs."""
        ds = ImageFolderDataSource(tiny_image_folder)
        assert ds.n_samples == 6  # 3 classes × 2 images

    def test_len_returns_batch_count(self, tiny_image_folder: Path) -> None:
        """__len__ returns ceil(n_samples / batch_size)."""
        ds = ImageFolderDataSource(tiny_image_folder, batch_size=4)
        assert len(ds) == 2  # ceil(6/4) = 2


class TestImageFolderIteration:
    """__iter__ tests."""

    def test_iter_yields_batches(self, tiny_image_folder: Path) -> None:
        """__iter__ yields Batch objects."""
        ds = ImageFolderDataSource(tiny_image_folder, batch_size=2, shuffle=False)
        batches = list(ds)
        assert len(batches) == 3  # 6/2 = 3 batches
        for batch in batches:
            assert len(batch.inputs) > 0
            assert len(batch.targets) > 0

    def test_batch_inputs_are_pil_images(self, tiny_image_folder: Path) -> None:
        """Batch.inputs contains PIL Image objects."""
        ds = ImageFolderDataSource(tiny_image_folder, batch_size=2, shuffle=False)
        batch = next(iter(ds))
        from PIL import Image

        for img in batch.inputs:
            assert isinstance(img, Image.Image)

    def test_batch_targets_are_int_labels(self, tiny_image_folder: Path) -> None:
        """Batch.targets are integer class labels (0, 1, 2)."""
        ds = ImageFolderDataSource(tiny_image_folder, batch_size=2, shuffle=False)
        batch = next(iter(ds))
        assert batch.targets.dtype in (np.int32, np.int64)
        assert all(0 <= lbl <= 2 for lbl in batch.targets)

    def test_no_shuffle_is_deterministic(self, tiny_image_folder: Path) -> None:
        """With shuffle=False, iteration order is deterministic."""
        ds1 = ImageFolderDataSource(tiny_image_folder, shuffle=False)
        ds2 = ImageFolderDataSource(tiny_image_folder, shuffle=False)
        labels1 = [int(batch.targets[0]) for batch in ds1]
        labels2 = [int(batch.targets[0]) for batch in ds2]
        assert labels1 == labels2

    def test_shuffle_same_seed_is_deterministic(self, tiny_image_folder: Path) -> None:
        """Same seed produces same shuffle order."""
        ds1 = ImageFolderDataSource(tiny_image_folder, shuffle=True, seed=42)
        ds2 = ImageFolderDataSource(tiny_image_folder, shuffle=True, seed=42)
        all_labels_1 = np.concatenate([batch.targets for batch in ds1])
        all_labels_2 = np.concatenate([batch.targets for batch in ds2])
        np.testing.assert_array_equal(all_labels_1, all_labels_2)

    def test_shuffle_different_seeds_produce_different_order(self, tiny_image_folder: Path) -> None:
        """Different seeds produce different order (very high probability)."""
        ds1 = ImageFolderDataSource(tiny_image_folder, shuffle=True, seed=42)
        ds2 = ImageFolderDataSource(tiny_image_folder, shuffle=True, seed=99)
        all_labels_1 = np.concatenate([batch.targets for batch in ds1])
        all_labels_2 = np.concatenate([batch.targets for batch in ds2])
        # With 6 images and 2 seeds, different order is near-certain
        assert not np.array_equal(all_labels_1, all_labels_2)

    def test_iter_multiple_times_produces_different_shuffles(self, tiny_image_folder: Path) -> None:
        """Each call to __iter__ reshuffles (new order)."""
        ds = ImageFolderDataSource(tiny_image_folder, shuffle=True)
        order1 = np.concatenate([batch.targets for batch in ds])
        order2 = np.concatenate([batch.targets for batch in ds])
        # Two reshuffles of 6 items are unlikely to be identical
        assert not np.array_equal(order1, order2) or len(order1) <= 1
