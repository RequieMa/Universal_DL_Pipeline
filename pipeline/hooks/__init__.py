"""Hook system for cross-cutting concerns in the pipeline.

Hooks observe pipeline lifecycle events without modifying the pipeline
stages themselves. Students add one hook at a time to layer on logging,
checkpointing, and early stopping.

Usage::

    from pipeline.hooks import BaseHook, ProgressHook

    pipeline.add_hook(ProgressHook())
"""

from __future__ import annotations

from pipeline.hooks.base import BaseHook
from pipeline.hooks.checkpoint import CheckpointHook
from pipeline.hooks.early_stop import EarlyStopHook
from pipeline.hooks.progress import ProgressHook

__all__ = ["BaseHook", "CheckpointHook", "EarlyStopHook", "ProgressHook"]
