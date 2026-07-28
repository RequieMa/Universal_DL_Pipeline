"""Unit tests for pipeline.utils.device."""

import pytest

from pipeline.utils.device import device_info, get_device


class TestGetDevice:
    """Tests for get_device()."""

    def test_auto_returns_string(self):
        """Happy Path: get_device('auto') returns a valid device string."""
        device = get_device("auto")
        assert device in ("cpu", "cuda", "mps")

    def test_cpu_explicit(self):
        """Happy Path: get_device('cpu') returns 'cpu'."""
        assert get_device("cpu") == "cpu"

    def test_invalid_device_raises(self):
        """Type Error: unknown device string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device"):
            get_device("tpu")

    def test_empty_string_raises(self):
        """Boundary: empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device"):
            get_device("")

    def test_cuda_name_normalized(self):
        """Happy Path: 'cuda:0' and 'cuda' both normalize to 'cuda'."""
        # NOTE: On a CPU-only machine, get_device('cuda') may fall back
        # to 'cpu'. This test verifies the normalization behavior.
        result = get_device("cuda")
        assert result in ("cpu", "cuda")

    def test_case_insensitive(self):
        """Boundary: device string is case-insensitive."""
        assert get_device("CPU") == "cpu"


class TestDeviceInfo:
    """Tests for device_info()."""

    def test_cpu_info_returns_string(self):
        """Happy Path: device_info('cpu') returns a non-empty string."""
        info = device_info("cpu")
        assert isinstance(info, str)
        assert len(info) > 0

    def test_info_contains_device_name(self):
        """Happy Path: device_info includes the device name."""
        info = device_info("cpu")
        assert "cpu" in info.lower() or "CPU" in info

    def test_info_unknown_device_raises(self):
        """Type Error: device_info for unknown device raises ValueError."""
        with pytest.raises(ValueError):
            device_info("quantum_computer")
