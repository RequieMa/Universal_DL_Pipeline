"""Device detection and information.

Provides a single function :func:`get_device` that resolves device
preferences across CUDA, MPS, and CPU. Students never need to write
``if torch.cuda.is_available()`` by hand.
"""

from __future__ import annotations

import platform


def get_device(preference: str = "auto") -> str:
    """Resolve a device preference to an available device name.

    Priority for ``"auto"``: CUDA -> MPS -> CPU.

    Args:
        preference: One of ``"auto"``, ``"cpu"``, ``"cuda"``,
            ``"mps"``. Case-insensitive.

    Returns:
        ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Raises:
        ValueError: If ``preference`` is not recognized.
    """
    preference = preference.lower()

    if preference == "auto":
        return _detect_best_device()

    valid = {"cpu", "cuda", "mps"}
    if preference not in valid:
        raise ValueError(
            f"Unknown device: {preference!r}. "
            f"Choose from: {', '.join(sorted(valid))}"
        )

    # NOTE: For explicit requests (not "auto"), return as-is.
    # Availability is checked by the framework adapter (e.g.,
    # TorchAdapter verifies torch.cuda.is_available()).
    return preference


def device_info(device: str) -> str:
    """Return a human-readable description of the compute device.

    Args:
        device: Device name (``"cpu"``, ``"cuda"``, or ``"mps"``).

    Returns:
        A string like ``"CPU (x86_64)"`` or ``"CUDA (not checked)"``.

    Raises:
        ValueError: If ``device`` is not recognized.
    """
    device = device.lower()
    if device == "cpu":
        return f"CPU ({platform.machine()})"
    elif device == "cuda":
        # WHY: We don't import torch here. The framework adapter
        # provides GPU details when it initializes.
        return "CUDA (availability checked by framework adapter)"
    elif device == "mps":
        return f"MPS ({platform.machine()}) -- Apple Silicon"
    else:
        raise ValueError(f"Unknown device: {device!r}")


def _detect_best_device() -> str:
    """Probe the system for the best available device.

    Tries to import torch to check CUDA/MPS availability, falls
    back to CPU. This is deliberately lazy-imported so the module
    works without PyTorch installed.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    # NOTE: torch.backends.mps.is_available() may exist but MPS
    # support is version-dependent. Check both availability and
    # whether it's actually built.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
