"""Unit tests for pipeline.utils.seed."""

import random

import numpy as np
import pytest

from pipeline.utils.seed import set_seed


class TestSetSeed:
    """Tests for set_seed()."""

    def test_sets_python_random(self):
        """Happy Path: set_seed makes random module reproducible."""
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_sets_numpy_random(self):
        """Happy Path: set_seed makes numpy reproducible."""
        set_seed(42)
        a = np.random.randn(3)
        set_seed(42)
        b = np.random.randn(3)
        assert np.array_equal(a, b)

    def test_different_seeds_produce_different_results(self):
        """Happy Path: different seeds → different random sequences."""
        set_seed(1)
        a = np.random.randn(5)
        set_seed(2)
        b = np.random.randn(5)
        assert not np.array_equal(a, b)

    def test_seed_zero_is_valid(self):
        """Boundary: seed=0 is valid (not the same as 'no seed')."""
        set_seed(0)
        set_seed(0)
        a = np.random.randn(3)
        set_seed(0)
        b = np.random.randn(3)
        assert np.array_equal(a, b)

    def test_negative_seed_is_valid(self):
        """Boundary: negative seeds are valid integers."""
        set_seed(-1)
        a = np.random.randn(3)
        set_seed(-1)
        b = np.random.randn(3)
        assert np.array_equal(a, b)

    def test_large_seed_is_valid(self):
        """Boundary: large seed value works correctly."""
        set_seed(2**31 - 1)
        a = np.random.randn(3)
        set_seed(2**31 - 1)
        b = np.random.randn(3)
        assert np.array_equal(a, b)

    def test_deterministic_flag_does_not_crash(self):
        """Happy Path: deterministic=True does not raise."""
        set_seed(42, deterministic=True)
        # Should not raise, even without CUDA

    def test_consecutive_calls_idempotent(self):
        """Concurrency: calling set_seed twice is fine."""
        set_seed(42)
        set_seed(42)
        # Should not raise


class TestSetSeedTorch:
    """Tests for set_seed torch integration."""

    def test_sets_torch_seed_if_available(self):
        """Happy Path: set_seed also sets torch manual seed.
        NOTE: Only tested when torch is installed."""
        try:
            import torch  # noqa: F811

            set_seed(42)
            a = torch.randn(3)
            set_seed(42)
            b = torch.randn(3)
            assert torch.allclose(a, b)
        except ImportError:
            pytest.skip("torch not installed")
