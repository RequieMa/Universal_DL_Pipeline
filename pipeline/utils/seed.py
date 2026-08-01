"""Reproducibility via seed setting.

A single function :func:`set_seed` that seeds Python's ``random``,
``numpy``, and (if available) ``torch``. Students call it once at
the top of their script.
"""

from __future__ import annotations

import random


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Seeds Python's ``random`` module, ``numpy``, and (if available)
    ``torch``. Call once at the start of your experiment.

    Args:
        seed: Integer seed for all random number generators.
        deterministic: If ``True``, enable deterministic CUDA
            operations. WARNING: This makes training significantly
            slower. Only use when debugging reproducibility issues.

    NOTE: Deterministic mode configures ``torch.backends.cudnn``
    and sets ``CUBLAS_WORKSPACE_CONFIG``. It does NOT guarantee
    bit-for-bit reproducibility across different hardware or
    PyTorch versions.
    """
    random.seed(seed)

    import numpy as np

    # numpy's legacy RandomState.seed requires seed in [0, 2**32-1].
    # Mask to uint32 so negative seeds work; Python's random and torch
    # accept negatives without issue.
    np.random.seed(seed & 0xFFFFFFFF)

    # Try torch -- don't crash if not installed
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if deterministic and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # NOTE: CUBLAS workspace config required for deterministic
            # cuDNN convolution algorithms.
            import os

            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass
